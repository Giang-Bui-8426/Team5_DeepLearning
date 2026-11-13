"""Evaluate one selected checkpoint, save one test bundle, then build its PDF."""
import argparse
import importlib.metadata
import json
import os
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, roc_curve, auc
from torch.utils.data import DataLoader, Subset

from data import BASE_DIR, CLASSES, DATA_DIR, WEIGHTS, check_metadata, load_dataset, load_split
from metrics import classification_metrics, file_hash, save_json, write_csv
from model import CNN


def load_model(path, device, smoke=False):
    cp = torch.load(path, map_location='cpu', weights_only=True)
    check_metadata(cp, smoke=smoke)
    if cp['epoch'] != cp['best_epoch']:
        raise ValueError('Inference requires a best checkpoint')
    if not smoke:
        selection = json.loads((path.parent.parent/'reports/selection.json').read_text(encoding='utf-8'))
        if (selection['run_id'] != cp['run_id'] or selection['best_epoch'] != cp['best_epoch']
                or selection['checkpoint_sha256'] != file_hash(path)
                or selection['split_sha256'] != cp['split_sha256']
                or selection['run_status'] not in ['completed','early_stopped']):
            raise ValueError('Checkpoint differs from validation selection')
    model = CNN(cp['architecture'], pretrained=False, fine_tune=cp['config']['fine_tune'])
    model.load_state_dict(cp['model_state_dict'], strict=True)
    return model.to(device).eval(), cp


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    loss_sum, forward_seconds = 0.0, 0.0
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        forward_start = time.perf_counter()
        logits = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - forward_start
        if logits.shape != (len(y),10) or not torch.isfinite(logits).all():
            raise FloatingPointError('Invalid inference logits')
        loss_sum += torch.nn.functional.cross_entropy(logits,y,reduction='sum').item()
        labels.append(y.cpu().numpy())
        probabilities.append(logits.softmax(1).cpu().numpy())
    elapsed = time.perf_counter() - start
    y, p = np.concatenate(labels), np.concatenate(probabilities)
    result = classification_metrics(y,p,loss_sum/len(y))
    result.update(labels=y, probabilities=p, predictions=p.argmax(1),
                  inference_seconds=forward_seconds,evaluation_seconds=elapsed)
    return result


def plot_results(result, raw_images, indices, output, num_images=12, seed=42, smoke=False):
    title = 'CIFAR-10 / VALIDATION SMOKE' if smoke else 'CIFAR-10 / Official test'
    cm = np.asarray(result['cm'])
    fig, ax = plt.subplots(figsize=(10,9))
    image = ax.imshow(cm,cmap='Blues')
    fig.colorbar(image,ax=ax,label='Number of images')
    ax.set(xticks=range(10),yticks=range(10),xlabel='Predicted class',ylabel='True class',title=title+' / Confusion matrix')
    ax.set_xticklabels(CLASSES,rotation=45,ha='right')
    ax.set_yticklabels(CLASSES)
    for i in range(10):
        for j in range(10):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout()
    fig.savefig(output/'confusion_matrix.png',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,7))
    for k,name in enumerate(CLASSES):
        binary = result['labels']==k
        if binary.any() and not binary.all():
            x,y,_ = roc_curve(binary,result['probabilities'][:,k])
            ax.plot(x,y,label=f'{name}: {auc(x,y):.3f}')
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlim=(0,1),ylim=(0,1.01),xlabel='False positive rate',ylabel='True positive rate',title=title+' / ROC OvR')
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left')
    fig.tight_layout()
    fig.savefig(output/'roc_ovr.png',dpi=220,bbox_inches='tight')
    plt.close(fig)
    correct = result['labels']==result['predictions']
    rng = np.random.default_rng(seed)
    selected = np.concatenate([rng.choice(np.flatnonzero(correct),min((num_images+1)//2,int(correct.sum())),replace=False),
                               rng.choice(np.flatnonzero(~correct),min(num_images//2,int((~correct).sum())),replace=False)])
    rng.shuffle(selected)
    rows = max(1,(len(selected)+3)//4)
    fig, axes = plt.subplots(rows,4,figsize=(12,rows*3.2),squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax,i in zip(axes.flat,selected):
        true,pred = result['labels'][i],result['predictions'][i]
        ax.imshow(raw_images[i],interpolation='nearest')
        ax.set_title(f'True: {CLASSES[true]}\nPred: {CLASSES[pred]} ({result["probabilities"][i,pred]:.1%})\n'
                     f'{"Đúng" if correct[i] else "Sai"} / index {indices[i]}',color='forestgreen' if correct[i] else 'firebrick')
    fig.suptitle(title+f' / Original RGB 32x32, enlarged / selected {len(selected)} of {num_images}')
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(output/'prediction_examples.png',dpi=220,bbox_inches='tight')
    plt.close(fig)
    return [int(indices[i]) for i in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=BASE_DIR/'checkpoints/checkpoint_best_global.pt')
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
    parser.add_argument('--output-dir',type=Path,default=BASE_DIR/'reports/test')
    parser.add_argument('--pdf-output',type=Path,default=BASE_DIR/'reports/Predict_Report.pdf')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--num-images',type=int,default=12)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--smoke',action='store_true',help='32 validation images; isolated checkpoint/output only')
    args = parser.parse_args()
    if args.batch_size<=0 or args.workers<0 or args.num_images<=0:
        parser.error('Invalid batch size/workers/gallery size')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output,pdf,checkpoint_path = args.output_dir.resolve(),args.pdf_output.resolve(),args.checkpoint.resolve()
    if args.smoke and (output==BASE_DIR/'reports/test' or pdf==BASE_DIR/'reports/Predict_Report.pdf'):
        parser.error('Smoke requires separate output directory and PDF path')
    artifacts = ['test_predictions.csv','test_classification_report.csv','confusion_matrix.png','roc_ovr.png','prediction_examples.png']
    if pdf.exists() or any((output/n).exists() for n in artifacts+['test_metrics.json','test_summary.md']):
        raise FileExistsError('Output exists; choose new paths')
    model,cp = load_model(checkpoint_path,device,args.smoke)
    ds = load_dataset(args.data_dir,train=args.smoke,transform=WEIGHTS[cp['architecture']].transforms())
    indices = load_split()[1][:32] if args.smoke else np.arange(10000)
    loader = DataLoader(Subset(ds,indices.tolist()),batch_size=args.batch_size,num_workers=args.workers,
                        shuffle=False,drop_last=False,pin_memory=device.type=='cuda')
    result = evaluate(model,loader,device)
    if result['metrics']['samples']!=len(indices):
        raise ValueError('Inference omitted samples')
    output.mkdir(parents=True,exist_ok=True)
    rows = [dict(sample_index=int(index),true_label=int(y),pred_label=int(pred),confidence=float(p[pred]),correct=int(y==pred),
                 **{f'prob_{k}':float(p[k]) for k in range(10)})
            for index,y,pred,p in zip(indices,result['labels'],result['predictions'],result['probabilities'])]
    write_csv(output/'test_predictions.csv',rows)
    per_class = [dict(p,class_name=CLASSES[p['class_id']]) for p in result['per_class']]
    write_csv(output/'test_classification_report.csv',per_class)
    chosen = plot_results(result,ds.data[indices],indices,output,args.num_images,args.seed,args.smoke)
    keys = ['run_id','architecture','mode','config','best_epoch','classes','transform_spec','augmentation_spec',
            'weights_enum','split_sha256','n_params','trainable_params','hardware','versions',
            'parent_run_id','parent_checkpoint_sha256','parent_best_epoch','parent_metrics']
    payload = {k:cp[k] for k in keys}
    payload.update(test_run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8],
                   dataset='CIFAR-10',split='validation_smoke' if args.smoke else 'official_test',is_smoke=args.smoke,
                   checkpoint_path=str(checkpoint_path),checkpoint_sha256=file_hash(checkpoint_path),
                   validation_metrics=cp['best_metrics'],metrics=result['metrics'],confusion_matrix=result['cm'],
                   classification_report=classification_report(result['labels'],result['predictions'],labels=range(10),
                                                                 target_names=CLASSES,output_dict=True,zero_division=0),
                   per_class=per_class,undefined_classes=result['undefined_classes'],selected_indices=chosen,seed=args.seed,num_images=args.num_images,
                   device=str(device),dtype='float32',batch_size=args.batch_size,workers=args.workers,
                   inference_hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
                   inference_versions={k:importlib.metadata.version(k) for k in ['torch','torchvision','numpy','scikit-learn']},
                   numeric_policy=dict(cudnn_deterministic=True,cudnn_allow_tf32=False,matmul_allow_tf32=False),
                   inference_seconds=result['inference_seconds'],evaluation_seconds=result['evaluation_seconds'],
                   artifact_hashes={n:file_hash(output/n) for n in artifacts})
    save_json(output/'test_metrics.json',payload)
    cm = np.asarray(result['cm'])
    pairs = sorted([(int(cm[i,j]),i,j) for i in range(10) for j in range(10) if i!=j and cm[i,j]],reverse=True)[:5]
    lines = ['# LAB2 — '+('VALIDATION SMOKE' if args.smoke else 'Official test'),'',
             f'Run `{cp["run_id"]}` · epoch {cp["best_epoch"]} · checkpoint SHA-256 `{payload["checkpoint_sha256"]}`','',
             '| Metric | Value |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in result['metrics'].items()]
    lines += ['', 'Validation metrics được lưu riêng trong JSON. Confidence chưa hiệu chỉnh. Gallery chỉ là minh họa.','']
    lines += [f'- {CLASSES[i]} → {CLASSES[j]}: {count} ảnh.' for count,i,j in pairs]
    lines += ['', '![Confusion matrix](confusion_matrix.png)','![ROC](roc_ovr.png)',
              '![Examples](prediction_examples.png)',f'[PDF]({os.path.relpath(pdf,output).replace(chr(92),"/")})']
    (output/'test_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    from PredictPDF import export_prediction_pdf
    export_prediction_pdf(report_dir=output,output_path=pdf,checkpoint=cp,checkpoint_path=checkpoint_path)
    print(json.dumps(result['metrics'],indent=2))
    print('PDF:',pdf)


if __name__=='__main__':
    main()
