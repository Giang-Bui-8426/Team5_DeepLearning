"""Create a self-contained, Vietnamese PDF report from Predict.py's real test artifacts.

Requires: pip install reportlab
Does not change inference, preprocessing, checkpoint loading, or metric computation.
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, LongTable, PageBreak,
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

_NAVY = colors.HexColor("#17263e")
_INK = colors.HexColor("#1b283b")
_MUTED = colors.HexColor("#5b687b")
_BLUE = colors.HexColor("#2463a5")
_LINE = colors.HexColor("#dde4ed")
_PALE = colors.HexColor("#f2f6fb")


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


def _page_decoration(canvas, document, regular: str, bold: str) -> None:
    canvas.saveState()
    page_width, page_height = A4
    canvas.setStrokeColor(_LINE)
    canvas.setLineWidth(0.6)
    canvas.line(19 * mm, page_height - 17 * mm, page_width - 19 * mm, page_height - 17 * mm)
    canvas.setFont(bold, 8)
    canvas.setFillColor(_NAVY)
    canvas.drawString(20 * mm, page_height - 14 * mm, "LAB1  /  FASHION-MNIST  /  TEST REPORT")
    canvas.line(19 * mm, 17.5 * mm, page_width - 19 * mm, 17.5 * mm)
    canvas.setFont(regular, 8)
    canvas.setFillColor(_MUTED)
    canvas.drawString(20 * mm, 13 * mm, "Báo cáo kiểm thử - dữ liệu test độc lập")
    canvas.drawRightString(page_width - 20 * mm, 13 * mm, f"Trang {document.page}")
    canvas.restoreState()


def _fnum(x, places=4) -> str:
    return f"{float(x):.{places}f}"


def _pct(x) -> str:
    return f"{float(x) * 100:.2f}%"


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


def _read_artifacts(report_dir: Path) -> tuple[dict, list[dict], dict[str, Path]]:
    json_path = report_dir / "test_metrics.json"
    csv_path = report_dir / "test_predictions.csv"
    images = {
        "confusion_matrix": report_dir / "confusion_matrix.png",
        "roc": report_dir / "roc_ovr.png",
        "examples": report_dir / "prediction_examples.png",
    }
    missing = [p for p in (json_path, csv_path, *images.values()) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Thiếu kết quả Predict.py; không tạo PDF với dữ liệu suy đoán:\n"
            + "\n".join(str(p) for p in missing)
        )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    count = int(payload["metrics"]["samples"])
    if len(rows) != count:
        raise ValueError(f"Kết quả không khớp: JSON={count} ảnh, CSV={len(rows)} ảnh.")
    ncorrect = sum(int(r["correct"]) for r in rows)
    if abs(ncorrect / count - float(payload["metrics"]["accuracy"])) > 1e-6:
        raise ValueError("Accuracy trong test_metrics.json không khớp test_predictions.csv.")
    return payload, rows, images


def export_prediction_pdf(*, report_dir: Path, output_path: Path,
                          checkpoint: dict, checkpoint_path: Path) -> Path:
    """Generate the PDF only from metrics/figures that the unchanged Predict.py already saved.

    A report is not generated if required real test artifacts are absent/inconsistent.
    """
    report_dir = Path(report_dir)
    output_path = Path(output_path)
    payload, rows, figs = _read_artifacts(report_dir)
    regular, bold = _register_vietnamese_fonts()

    # Match the JSON produced by the current Predict.py, not a regenerated evaluation.
    metrics = payload["metrics"]
    class_report = payload["classification_report"]
    config = checkpoint.get("config", {})
    model_name = str(payload.get("model", checkpoint.get("model_name", "unknown")))
    if model_name != str(checkpoint.get("model_name", model_name)):
        raise ValueError("Tên model trong JSON và checkpoint không khớp.")
    if payload.get("checkpoint_epoch") != checkpoint.get("epoch"):
        raise ValueError("Epoch trong JSON và checkpoint không khớp.")

    page_w, page_h = A4
    margin = 20 * mm
    width = page_w - 2 * margin
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=24 * mm, bottomMargin=22 * mm,
        title="Báo cáo kiểm thử Fashion-MNIST - LAB1",
        author="LAB1 - Deep Learning",
        pageCompression=1,
    )
    body = ParagraphStyle("body", fontName=regular, fontSize=9.5,
                          leading=15, textColor=_INK, spaceAfter=9)
    subtitle = ParagraphStyle("subtitle", parent=body, fontSize=8.5,
                              leading=13, textColor=_MUTED, spaceAfter=12)
    heading = ParagraphStyle("heading", parent=body, fontName=bold,
                             fontSize=14, leading=20, textColor=_NAVY,
                             spaceBefore=14, spaceAfter=12, keepWithNext=True)
    top_heading = ParagraphStyle("title", parent=heading, fontSize=21,
                                 leading=28, spaceBefore=12, spaceAfter=15)
    caption = ParagraphStyle("caption", parent=subtitle, alignment=TA_CENTER,
                             spaceBefore=7, spaceAfter=10)
    story = []

    def p(text: str, style=body):
        return Paragraph(escape(str(text)), style)

    def h(text: str):
        story.append(p(text, heading))

    def fig_page(title: str, image_key: str, caption_text: str):
        story.append(PageBreak())
        h(title)
        story.append(_image_fit(figs[image_key], width, 205 * mm))
        story.append(p(caption_text, caption))

    # Cover / overview
    story.append(p("BÁO CÁO KẾT QUẢ KIỂM THỬ", top_heading))
    story.append(p("LAB1 - Phân loại Fashion-MNIST bằng mạng MLP", subtitle))
    story.append(HRFlowable(width="100%", thickness=1.25, color=_BLUE,
                            spaceBefore=2, spaceAfter=15))
    info = [
        ["Thông tin", "Giá trị"],
        ["Mô hình được chọn", model_name],
        ["Checkpoint", str(checkpoint_path.name)],
        ["Epoch checkpoint", str(payload.get("checkpoint_epoch", "-"))],
        ["Tập kiểm thử", str(payload.get("dataset", "-"))],
        ["Số ảnh test", f"{int(metrics['samples']):,}"],
        ["Thời điểm tạo báo cáo", datetime.now().strftime("%d/%m/%Y %H:%M")],
    ]
    story.append(_table(info, [56 * mm, width - 56 * mm], regular, bold))
    h("1. Cấu hình mô hình")
    if not config:
        story.append(p("Checkpoint không cung cấp cấu hình.", subtitle))
    else:
        crows = [["Tham số", "Giá trị"]]
        for key, value in config.items():
            val = _as_text(value)
            crows.append([p(key), p(val)])
        story.append(_table(crows, [56 * mm, width - 56 * mm], regular, bold, compact=True))

    h("2. Chỉ số đánh giá trên tập test")
    labels = [
        ("samples", "Số mẫu test", lambda v: f"{int(v):,}"),
        ("loss", "Cross-entropy loss", _fnum),
        ("accuracy", "Accuracy", _pct),
        ("precision_macro", "Precision (macro)", _pct),
        ("recall_macro", "Recall (macro)", _pct),
        ("f1_macro", "F1 (macro)", _pct),
        ("fpr_macro", "FPR (macro)", _pct),
        ("auc_macro_ovr", "ROC-AUC (macro, OvR)", _fnum),
        ("auc_micro_ovr", "ROC-AUC (micro, OvR)", _fnum),
    ]
    metric_rows = [["Metric", "Kết quả"]]
    for key, label, fmt in labels:
        if key in metrics:
            metric_rows.append([label, fmt(metrics[key])])
    story.append(_table(metric_rows, [100 * mm, width - 100 * mm], regular, bold))
    story.append(p("Các chỉ số được trích từ test_metrics.json do Predict.py tạo; "
                   "không sử dụng accuracy trên validation để thay thế.", subtitle))

    story.append(PageBreak())
    h("3. Classification report theo từng lớp")
    header = ["Lớp", "Precision", "Recall", "F1", "Support"]
    class_rows = [header]
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
        "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    for name in class_names:
        if name not in class_report:
            raise ValueError(f"Không có classification report của lớp {name}.")
        m = class_report[name]
        class_rows.append([name, _fnum(m["precision"]), _fnum(m["recall"]),
                           _fnum(m["f1-score"]), f"{int(m['support']):,}"])
    for name, label in [("macro avg", "Macro avg"), ("weighted avg", "Weighted avg")]:
        if name in class_report:
            m = class_report[name]
            class_rows.append([label, _fnum(m["precision"]), _fnum(m["recall"]),
                               _fnum(m["f1-score"]), f"{int(m['support']):,}"])
    story.append(_table(class_rows, [49*mm, 30*mm, 29*mm, 29*mm, width - 137*mm],
                        regular, bold, compact=True))
    story.append(p("Support là số ảnh thực tế của từng lớp trên tập test.", subtitle))

    # Error analysis from the actual saved predictions, without re-running inference.
    h("4. Phân tích lỗi từ kết quả test")
    n = len(rows)
    ncorrect = sum(int(row["correct"]) for row in rows)
    wrong = n - ncorrect
    story.append(p(f"Tổng số ảnh: {n:,}. Dự đoán đúng: {ncorrect:,}. "
                   f"Dự đoán sai: {wrong:,}. Accuracy: {_pct(ncorrect/n)}."))
    low_name = min(class_names, key=lambda name: float(class_report[name]["recall"]))
    story.append(p(f"Lớp có recall thấp nhất: {low_name} "
                   f"({_pct(class_report[low_name]['recall'])}); "
                   "nên xem ma trận nhầm lẫn để nhận diện các kiểu sai của lớp này."))
    confusions = Counter((r["actual"], r["predicted"]) for r in rows
                         if r["actual"] != r["predicted"])
    if confusions:
        (actual, predicted), count = confusions.most_common(1)[0]
        story.append(p(f"Kiểu nhầm lẫn phổ biến nhất: {actual} → {predicted} "
                       f"({count:,} ảnh)."))
    else:
        story.append(p("Không có ảnh nào bị dự đoán sai trong kết quả test được lưu."))
    story.append(p("Nhận xét chỉ dựa trên test_predictions.csv và test_metrics.json; "
                   "không suy đoán nguyên nhân hoặc kết quả chưa đo được.", subtitle))

    # Figures: original exported images embedded into the PDF.
    fig_page("5. Confusion matrix - Tập test", "confusion_matrix",
             "Hình 1. Ma trận nhầm lẫn trên tập test độc lập (số lượng ảnh mỗi ô).")
    fig_page("6. ROC curves - One-vs-Rest", "roc",
             "Hình 2. Đường cong ROC trên tập test; AUC từng lớp nằm trong chú giải của biểu đồ.")
    fig_page("7. Ví dụ dự đoán đúng và sai", "examples",
             "Hình 3. Các ảnh dự đoán minh họa, kèm nhãn thật, nhãn dự đoán và độ tin cậy.")
    h("8. Kết luận")
    story.append(p(
        f"Mô hình {model_name} được đánh giá trên {n:,} ảnh thuộc tập test chính thức. "
        f"Accuracy đạt {_pct(metrics['accuracy'])}, macro-F1 đạt {_pct(metrics['f1_macro'])}, "
        f"và macro ROC-AUC OvR đạt {_fnum(metrics['auc_macro_ovr'])}. "
        "Các chỉ số và hình minh họa trong báo cáo được xuất từ lần kiểm thử này."
    ))
    story.append(p("Tệp dữ liệu gốc được giữ nguyên tại reports/test/ "
                   "để thuận tiện đối chiếu và phân tích tiếp.", subtitle))

    # Fail visibly if writing the PDF is unsuccessful; no fake report is generated.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=lambda canv, d: _page_decoration(canv, d, regular, bold),
              onLaterPages=lambda canv, d: _page_decoration(canv, d, regular, bold))
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Tạo PDF không thành công: {output_path}")
    return output_path


if __name__ == "__main__":
    # Recreate the PDF from existing test artifacts, without running inference again.
    import argparse
    import torch

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Xuất PDF từ kết quả Predict.py đã lưu")
    parser.add_argument("--report-dir", type=Path, default=root / "reports" / "test")
    parser.add_argument("--checkpoint", type=Path,
                        default=root / "checkpoints" / "checkpoint_best_global.pt")
    parser.add_argument("--out", type=Path, default=root / "reports" / "Predict_Report.pdf")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Không tìm thấy checkpoint: {args.checkpoint}")
    info = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dest = export_prediction_pdf(report_dir=args.report_dir, output_path=args.out,
                                 checkpoint=info, checkpoint_path=args.checkpoint)
    print(f"[PredictPDF] Đã xuất: {dest}")
