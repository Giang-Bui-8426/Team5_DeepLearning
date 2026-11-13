"""Validate a saved prediction bundle and export a Vietnamese PDF; no inference."""
import argparse
import csv
import json
import os
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, LongTable, PageBreak, Paragraph, SimpleDocTemplate, TableStyle

from data import CLASSES
from metrics import classification_metrics, file_hash

_INK = colors.HexColor('#1b283b')
_MUTED = colors.HexColor('#5b687b')
_NAVY = colors.HexColor('#1b283b')
_LINE = colors.HexColor('#dde4ed')
_PALE = colors.HexColor('#f2f6fb')


def _read_artifacts(report_dir):
    root = Path(report_dir)
    p = json.loads((root/'test_metrics.json').read_text(encoding='utf-8'))
    required = {'test_predictions.csv','test_classification_report.csv','confusion_matrix.png','roc_ovr.png','prediction_examples.png'}
    if set(p['artifact_hashes'])!=required or p['classes']!=CLASSES or p['dataset']!='CIFAR-10':
        raise ValueError('Invalid artifact manifest/classes/dataset')
    if p['split']!=('validation_smoke' if p['is_smoke'] else 'official_test'):
        raise ValueError('Split/smoke mismatch')
    for name,digest in p['artifact_hashes'].items():
        if file_hash(root/name)!=digest:
            raise ValueError(f'Artifact hash mismatch: {name}')
    with (root/'test_predictions.csv').open(encoding='utf-8',newline='') as f:
        rows = list(csv.DictReader(f))
    count = 32 if p['is_smoke'] else 10000
    if len(rows)!=count:
        raise ValueError('Prediction count mismatch')
    indices = [int(r['sample_index']) for r in rows]
    if len(set(indices))!=count or (not p['is_smoke'] and indices!=list(range(count))):
        raise ValueError('Invalid sample indices')
    y = np.asarray([int(r['true_label']) for r in rows])
    prob = np.asarray([[float(r[f'prob_{k}']) for k in range(10)] for r in rows])
    result = classification_metrics(y,prob,p['metrics']['loss'])
    for key,value in result['metrics'].items():
        if key=='loss':  # Loss came from logits; reusing it is not an independent check.
            continue
        saved = p['metrics'][key]
        if (value is None)!=(saved is None) or (value is not None and not np.isclose(value,saved,atol=1e-10,rtol=1e-10)):
            raise ValueError(f'JSON/CSV metric mismatch: {key}')
    if result['cm']!=p['confusion_matrix']:
        raise ValueError('Confusion matrix mismatch')
    if not p['is_smoke'] and not np.all(np.bincount(y,minlength=10)==1000):
        raise ValueError('Official test support mismatch')
    for i,r in enumerate(rows):
        pred = int(prob[i].argmax())
        if (int(r['pred_label'])!=pred or int(r['correct'])!=int(pred==y[i])
                or not np.isclose(float(r['confidence']),prob[i,pred],atol=1e-10,rtol=1e-10)):
            raise ValueError('Prediction/confidence mismatch')
    with (root/'test_classification_report.csv').open(encoding='utf-8',newline='') as f:
        per_class = list(csv.DictReader(f))
    if len(per_class)!=10 or len(p['per_class'])!=10:
        raise ValueError('Per-class report must have ten rows')
    for k,(r,computed,recorded) in enumerate(zip(per_class,result['per_class'],p['per_class'])):
        if int(r['class_id'])!=k or r['class_name']!=CLASSES[k] or recorded['class_name']!=CLASSES[k]:
            raise ValueError('Per-class order mismatch')
        for key,value in computed.items():
            csv_value = float(r[key]) if r[key] else None
            for saved in [csv_value,recorded[key]]:
                if (value is None)!=(saved is None) or (value is not None and not np.isclose(value,saved,atol=1e-10,rtol=1e-10)):
                    raise ValueError(f'Per-class mismatch: {key}')
    chosen = p['selected_indices']
    if len(chosen)!=len(set(chosen)) or not set(chosen)<=set(indices):
        raise ValueError('Invalid gallery indices')
    return p,rows


def _page_decoration(canvas, document, regular, bold):
    canvas.saveState()
    canvas.setFont(regular,8)
    canvas.setFillColor(_MUTED)
    label = 'VALIDATION SMOKE' if 'SMOKE' in document.title else 'TEST REPORT'
    canvas.drawString(20*mm,13*mm,'LAB2 / CIFAR-10 / '+label)
    canvas.drawRightString(190*mm,13*mm,f'Trang {document.page}')
    canvas.restoreState()


def export_prediction_pdf(*, report_dir, output_path, checkpoint=None, checkpoint_path=None):
    root,output = Path(report_dir),Path(output_path)
    if output.exists():
        raise FileExistsError(f'PDF already exists: {output}')
    p,rows = _read_artifacts(root)
    if checkpoint is not None and (checkpoint['run_id']!=p['run_id'] or checkpoint['best_epoch']!=p['best_epoch']):
        raise ValueError('Checkpoint identity mismatch')
    if checkpoint_path is not None and file_hash(checkpoint_path)!=p['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    regular,bold = _register_vietnamese_fonts()
    body = ParagraphStyle('body',fontName=regular,fontSize=9,leading=14,textColor=_INK,spaceAfter=8)
    heading = ParagraphStyle('heading',parent=body,fontName=bold,fontSize=16,leading=22,spaceAfter=12,keepWithNext=True)
    caption = ParagraphStyle('caption',parent=body,alignment=TA_CENTER,textColor=_MUTED)
    def paragraph(value,style=body):
        return Paragraph(escape(str(value)),style)
    def table(values,widths):
        cells = [[paragraph(v,ParagraphStyle('cell',parent=body,fontSize=8,leading=11,
                                            textColor=colors.white if i==0 else _INK)) for v in row]
                 for i,row in enumerate(values)]
        return _table(cells,widths,regular,bold,compact=True)
    title = 'VALIDATION SMOKE — KHÔNG PHẢI KẾT QUẢ TEST' if p['is_smoke'] else 'BÁO CÁO KIỂM THỬ CIFAR-10'
    story = [paragraph(title,heading),paragraph('LAB2 — Pretrained CNN. Chọn checkpoint bằng validation trước khi test.')]
    info = [['Thông tin','Giá trị']]+[[k,p[k]] for k in ['run_id','architecture','mode','best_epoch','checkpoint_sha256','test_run_id','split','device','dtype']]
    story += [table(info,[48*mm,122*mm]),paragraph('1. Validation và Test',heading)]
    metric_rows = [['Metric','Validation','Validation smoke' if p['is_smoke'] else 'Test']]
    for key in ['loss','accuracy','precision_macro','recall_macro','f1_macro','fpr_macro','auc_macro_ovr','auc_micro_ovr']:
        metric_rows.append([key]+['N/A' if v is None else f'{v:.6f}' for v in [p['validation_metrics'].get(key),p['metrics'].get(key)]])
    story += [table(metric_rows,[65*mm,52*mm,53*mm]),
              paragraph(f"Số mẫu: {p['metrics']['samples']:,}; đúng: {p['metrics']['correct_count']:,}; sai: {p['metrics']['incorrect_count']:,}."),
              paragraph(f"Forward FP32: {p['inference_seconds']:.3f}s; evaluation gồm dữ liệu/transfer/gom kết quả: {p['evaluation_seconds']:.3f}s. Không gồm tải model, vẽ hình hoặc PDF."),
              PageBreak(),paragraph('2. Cấu hình và nguồn checkpoint',heading)]
    keys = ['config','weights_enum','transform_spec','augmentation_spec','split_sha256','n_params','trainable_params',
            'parent_run_id','parent_checkpoint_sha256','parent_best_epoch','versions','hardware','inference_versions','inference_hardware','numeric_policy']
    story.append(table([['Thông số','Giá trị']]+[[k,_as_text(p.get(k))] for k in keys],[48*mm,122*mm]))
    story += [PageBreak(),paragraph('3. Chất lượng từng lớp',heading)]
    class_rows = [['Lớp','Precision','Recall','F1','Support','FPR','AUC']]
    for r in p['per_class']:
        class_rows.append([r['class_name']]+['N/A' if r[k] is None else str(r[k]) if k=='support' else f'{r[k]:.4f}'
                                          for k in ['precision','recall','f1','support','fpr','auc']])
    story.append(table(class_rows,[34*mm,24*mm,23*mm,23*mm,21*mm,22*mm,23*mm]))
    story.append(paragraph('4. Phân tích lỗi',heading))
    pairs = Counter((int(r['true_label']),int(r['pred_label'])) for r in rows if r['correct']=='0')
    for (true,pred),count in pairs.most_common(5):
        story.append(paragraph(f'{CLASSES[true]} → {CLASSES[pred]}: {count} ảnh.'))
    worst = min(p['per_class'],key=lambda r:r['recall'])
    story.append(paragraph(f"Recall thấp nhất: {worst['class_name']} ({worst['recall']:.2%}); chưa xác định nguyên nhân lỗi từ số đếm."))
    for title_image,name in [('5. Confusion matrix — hàng thật, cột dự đoán','confusion_matrix.png'),
                             ('6. ROC One-vs-Rest','roc_ovr.png'),('7. Ảnh RGB gốc — dự đoán đúng/sai','prediction_examples.png')]:
        story += [PageBreak(),paragraph(title_image,heading),_image_fit(root/name,170*mm,195*mm),paragraph(p['test_run_id'],caption)]
    story += [paragraph('8. Giới hạn kết luận',heading),
              paragraph('Một split/seed chưa xác nhận độ ổn định. Confidence Softmax chưa hiệu chỉnh; gallery chỉ là mẫu minh họa. Không chọn lại model từ test, không suy ra nguyên nhân lỗi hay mức cải thiện giữa các mạng từ một checkpoint.'),
              paragraph('Accuracy/F1/AUC/CM đã đối chiếu từ CSV. Loss được tính từ logits lúc inference; không được kiểm chứng độc lập bằng việc đọc lại loss JSON.')]
    if checkpoint_path is None:
        story.append(paragraph('Dựng lại từ bundle đã lưu: đã kiểm tra hash CSV/PNG; chưa đối chiếu lại file checkpoint gốc trong lần export này.'))
    output.parent.mkdir(parents=True,exist_ok=True)
    temp = output.with_suffix('.pdf.tmp')
    doc = SimpleDocTemplate(str(temp),pagesize=A4,leftMargin=20*mm,rightMargin=20*mm,topMargin=20*mm,bottomMargin=22*mm,title=title)
    doc.build(story,onFirstPage=lambda c,d:_page_decoration(c,d,regular,bold),onLaterPages=lambda c,d:_page_decoration(c,d,regular,bold))
    temp.replace(output)
    return output


def _register_vietnamese_fonts() -> tuple[str, str]:
    """Use local installed fonts; never embed a font file in the project package."""
    overrides = (os.environ.get("PREDICT_PDF_FONT"), os.environ.get("PREDICT_PDF_BOLD_FONT"))
    choices = []
    if all(overrides):
        choices.append((Path(overrides[0]), Path(overrides[1])))
    choices.extend((Path(a), Path(b)) for a, b in [
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        ("C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf"),
        ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ])
    for regular_path, bold_path in choices:
        if regular_path.is_file() and bold_path.is_file():
            if "PredictPDF-Regular" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("PredictPDF-Regular", str(regular_path)))
            if "PredictPDF-Bold" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("PredictPDF-Bold", str(bold_path)))
            pdfmetrics.registerFontFamily(
                "PredictPDF-Regular", normal="PredictPDF-Regular",
                bold="PredictPDF-Bold", italic="PredictPDF-Regular",
                boldItalic="PredictPDF-Bold",
            )
            return "PredictPDF-Regular", "PredictPDF-Bold"
    raise RuntimeError(
        "Không tìm thấy font Unicode hỗ trợ tiếng Việt. "
        "Hãy cài DejaVu Sans/Noto Sans/Arial, hoặc khai báo "
        "PREDICT_PDF_FONT và PREDICT_PDF_BOLD_FONT trỏ đến các file .ttf trên máy."
    )


def _as_text(value) -> str:
    if isinstance(value, (tuple, list)):
        return ", ".join(str(x) for x in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _image_fit(path: Path, max_w: float, max_h: float) -> Image:
    w, h = ImageReader(str(path)).getSize()
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale, kind="proportional")


def _table(rows: list[list], col_widths: list[float], regular: str, bold: str,
           header: bool = True, compact: bool = False) -> LongTable:
    t = LongTable(rows, colWidths=col_widths, repeatRows=1 if header else 0,
                  hAlign="LEFT", splitByRow=1)
    stylings = [
        ("FONTNAME", (0, 0), (-1, -1), regular),
        ("FONTSIZE", (0, 0), (-1, -1), 8.0 if compact else 9.2),
        ("LEADING", (0, 0), (-1, -1), 11 if compact else 13),
        ("TEXTCOLOR", (0, 0), (-1, -1), _INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5 if compact else 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5 if compact else 7),
        ("LINEBELOW", (0, -1), (-1, -1), 0.55, _LINE),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, _PALE]),
    ]
    if header:
        stylings += [
            ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
            ("FONTNAME", (0, 0), (-1, 0), bold),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]
    t.setStyle(TableStyle(stylings))
    return t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,default=Path(__file__).resolve().parent/'reports/test')
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'reports/Predict_Report.pdf')
    args = parser.parse_args()
    print(export_prediction_pdf(report_dir=args.report_dir,output_path=args.out))
