"""Verify six best checkpoints on validation and export evidence for evaluation.md."""
import argparse
import csv
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import BASE_DIR, CLASSES, DATA_DIR, WEIGHTS, check_metadata, load_dataset, load_split
from metrics import file_hash, save_json, write_csv
from model import CNN, parameter_statistics
from Predict import evaluate
from PredictPDF import _read_artifacts
from TrainReport import load_runs


def collect(root, data_dir, device):
    runs = sorted(load_runs(root/'reports'),key=lambda r:r['name'])
    expected = {(a,mode) for a in WEIGHTS for mode in ['feature_extract','fine_tune']}
    if len(runs)!=6 or {(r.get('architecture'),r.get('mode')) for r in runs}!=expected:
        raise ValueError('Evaluation requires exactly the six core runs')
    selection = json.loads((root/'reports/selection.json').read_text(encoding='utf-8'))
    test,_ = _read_artifacts(root/'reports/test')
    if test['is_smoke'] or test['metrics']['samples']!=10000:
        raise ValueError('Official train and test must finish before reporting')
    global_hash = file_hash(root/'checkpoints/checkpoint_best_global.pt')
    if test['checkpoint_sha256']!=global_hash or selection['checkpoint_sha256']!=global_hash:
        raise ValueError('Test and selection use different checkpoints')
    ds = load_dataset(data_dir,train=True)
    _,indices,split_hash = load_split()
    rows,resources,verification = [],[],[]
    results = {}
    for run in runs:
        name,epochs = run['name'],run['epochs']
        if run['status'] not in ['completed','early_stopped'] or run.get('is_smoke'):
            raise ValueError(f'{name}: incomplete or smoke run')
        if [e['epoch'] for e in epochs]!=list(range(1,len(epochs)+1)):
            raise ValueError(f'{name}: non-contiguous history')
        if any(e['train_images']!=45000 or e['val_images']!=5000 for e in epochs):
            raise ValueError(f'{name}: incorrect epoch sample counts')
        path = root/'checkpoints'/name/'checkpoint_best.pt'
        cp = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        check_metadata(cp)
        if (cp['run_id']!=name or cp['config']!=run['config'] or cp['epoch']!=run['best_epoch']
                or cp['epoch']!=cp['best_epoch'] or cp['best_metrics']!=run['best_metrics']):
            raise ValueError(f'{name}: checkpoint/history identity mismatch')
        if any(file_hash(BASE_DIR/n)!=h for n,h in cp['code_sha256'].items()):
            raise ValueError(f'{name}: training source changed')
        if cp['parent_run_id']:
            parent_path = root/'checkpoints'/cp['parent_run_id']/'checkpoint_best.pt'
            if file_hash(parent_path)!=cp['parent_checkpoint_sha256']:
                raise ValueError(f'{name}: parent checkpoint changed')
        model = CNN(cp['architecture'],pretrained=False,fine_tune=cp['config']['fine_tune'])
        counts = parameter_statistics(model)
        if any(counts[k]!=cp[k] for k in counts):
            raise ValueError(f'{name}: parameter counts mismatch')
        model.load_state_dict(cp['model_state_dict'],strict=True)
        ds.transform = WEIGHTS[cp['architecture']].transforms()
        loader = DataLoader(Subset(ds,indices.tolist()),batch_size=cp['config']['val_batch_size'],
                            shuffle=False,num_workers=0,drop_last=False,pin_memory=device.type=='cuda')
        result = evaluate(model.to(device).eval(),loader,device)
        for key,value in result['metrics'].items():
            if not np.isclose(value,cp['best_metrics'][key],atol=1e-5 if key=='loss' else 1e-8,rtol=0):
                raise ValueError(f'{name}: recomputed {key} differs from checkpoint')
        with np.load(root/'reports'/name/'best_validation.npz',allow_pickle=False) as z:
            if not np.array_equal(z['y_true'],result['labels']) or not np.allclose(z['y_prob'],result['probabilities'],atol=1e-7,rtol=1e-5):
                raise ValueError(f'{name}: saved best predictions mismatch')
            max_difference = float(np.max(np.abs(z['y_prob']-result['probabilities'])))
        best = next(e for e in epochs if e['epoch']==run['best_epoch'])
        minimum = min(epochs,key=lambda e:e['val_loss'])
        row = dict(run_id=name,architecture=cp['architecture'],mode=cp['mode'],status=run['status'],
                   best_epoch=cp['best_epoch'],epochs_trained=len(epochs),**result['metrics'],
                   val_loss_at_best=result['metrics']['loss'],min_val_loss=minimum['val_loss'],
                   epoch_of_min_val_loss=minimum['epoch'],train_loss_at_best=best['train_loss'],
                   train_accuracy_at_best=best['train_acc'],final_val_loss=epochs[-1]['val_loss'],
                   final_val_accuracy=epochs[-1]['val_acc'],checkpoint_sha256=file_hash(path),
                   parent_run_id=cp['parent_run_id'],parent_checkpoint_sha256=cp['parent_checkpoint_sha256'])
        rows.append(row)
        resource = dict(run_id=name,total_params=cp['n_params'],trainable_params=cp['trainable_params'],
                        trainable_pct=100*cp['trainable_params']/cp['n_params'],epochs_trained=len(epochs),
                        train_seconds_total=sum(e['train_time'] for e in epochs),
                        val_seconds_total=sum(e['val_time'] for e in epochs),
                        train_seconds_per_epoch_median=float(np.median([e['train_time'] for e in epochs])),
                        val_seconds_per_epoch_median=float(np.median([e['val_time'] for e in epochs])),
                        device=cp['device'],amp_dtype=cp['precision'],train_batch_size=cp['config']['batch_size'],
                        val_batch_size=cp['config']['val_batch_size'],workers=cp['config']['workers'])
        for phase in ['train','val']:
            resource[f'{phase}_images_per_second'] = sum(e[f'{phase}_images'] for e in epochs)/resource[f'{phase}_seconds_total']
            for key in ['peak_cuda_allocated_mib','peak_cuda_reserved_mib','cpu_rss_observed_peak_mib']:
                values = [e[f'{phase}_{key}'] for e in epochs if e.get(f'{phase}_{key}') is not None]
                resource[f'{phase}_{key}'] = max(values) if values else None
        resource['cpu_rss_observed_peak_mib'] = max(
            (resource[f'{p}_cpu_rss_observed_peak_mib'] for p in ['train','val']
             if resource[f'{p}_cpu_rss_observed_peak_mib'] is not None),default=None)
        resources.append(resource)
        verification.append(dict(run_id=name,checkpoint_sha256=row['checkpoint_sha256'],metrics=result['metrics'],
                                 max_probability_difference=max_difference,verification_seconds=result['evaluation_seconds']))
        results[name] = result
        print(f'{name}: validation verified ({result["metrics"]["accuracy"]:.2%})',flush=True)
        del model,cp,loader
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    for r,resource in zip(rows,resources):
        parent = next((p for p in resources if p['run_id']==r['parent_run_id']),None)
        resource['stage_seconds_total'] = resource['train_seconds_total']+resource['val_seconds_total']
        resource['chain_seconds_total'] = resource['stage_seconds_total'] + (
            parent['train_seconds_total']+parent['val_seconds_total'] if parent else 0)
        resource['chain_epochs'] = resource['epochs_trained']+(parent['epochs_trained'] if parent else 0)
    ranking = sorted(rows,key=lambda r:(-r['accuracy'],-r['f1_macro'],r['loss'],r['run_id']))
    if ranking[0]['run_id']!=selection['run_id'] or test['run_id']!=selection['run_id']:
        raise ValueError('Recomputed validation ranking disagrees with selection/test')
    # Check CSV and TensorBoard independently of the stored checkpoint metrics.
    with (root/'reports/run_summary.csv').open(encoding='utf-8',newline='') as f:
        summary = {r['run_id']:r for r in csv.DictReader(f)}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    for r,run in zip(rows,runs):
        if int(summary[r['run_id']]['best_epoch'])!=r['best_epoch']:
            raise ValueError('Summary best epoch mismatch')
        events = EventAccumulator(str(root/'logs'/r['run_id'])).Reload()
        for tag,key in [('Loss/train_epoch','train_loss'),('Loss/validation','val_loss'),
                        ('Validation/accuracy','val_acc'),('Validation/f1_macro','f1')]:
            values = events.Scalars(tag)
            if len(values)!=len(run['epochs']) or any(v.step!=e['epoch'] or not np.isclose(v.value,e[key],atol=1e-6,rtol=1e-6)
                                                        for v,e in zip(values,run['epochs'])):
                raise ValueError(f'{r["run_id"]}: TensorBoard/history mismatch')
    return runs,rows,resources,results,dict(split_sha256=split_hash,checkpoints=verification,
               selected_run_id=selection['run_id'],selected_checkpoint_sha256=global_hash,
               test_run_id=test['test_run_id'],tensorboard_verified=True,test_bundle_verified=True)


def plot_evidence(runs,rows,resources,results,folder,selected):
    folder.mkdir(parents=True,exist_ok=True)
    events = folder.parent/'execution_events.json'
    has_sleep = events.is_file() and bool(json.loads(events.read_text(encoding='utf-8')).get('power_events'))
    plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':0.2})
    labels = [r['architecture'].replace('densenet121','Dense121').replace('resnet18','ResNet18').replace('vgg16','VGG16')+
              (' FT' if r['mode']=='fine_tune' else ' FE') for r in rows]
    colors = ['#2563eb','#2563eb','#f97316','#f97316','#16a34a','#16a34a']
    def save(fig,name):
        timing_note = has_sleep and name.startswith(('04_','05_'))
        if timing_note:
            fig.text(.5,.005,'Recorded elapsed times include system sleep; see execution_events.json.',ha='center',fontsize=9)
        fig.suptitle('CIFAR-10 / VALIDATION SMOKE' if any(r.get('is_smoke') for r in runs)
                     else 'CIFAR-10 / six official runs',fontsize=12)
        fig.tight_layout(rect=(0,(.18 if timing_note else .14) if fig.legends else .07 if timing_note else 0,1,.94))
        fig.savefig(folder/name,dpi=220,bbox_inches='tight')
        plt.close(fig)
    def curves(keys,titles,name,percent=False):
        fig,axes = plt.subplots(1,2,figsize=(13,5))
        for ax,key,title in zip(axes,keys,titles):
            for i,r in enumerate(runs):
                e = r['epochs']
                ax.plot([x['epoch'] for x in e],[(100 if percent else 1)*x[key] for x in e],
                        color=colors[i],linestyle='--' if i%2 else '-',label=labels[i],marker='s' if i%2 else 'o',markersize=3)
                if percent:
                    best=next(x for x in e if x['epoch']==r['best_epoch'])
                    ax.scatter(best['epoch'],100*best[key],color=colors[i],marker='*',s=130,zorder=5)
            ax.set(title=title,xlabel='Epoch within stage',ylabel='Percent' if percent else 'Cross-entropy')
            ax.legend(fontsize=9)
        save(fig,name)
    curves(['val_acc','f1'],['Validation accuracy (* best epoch)','Validation macro F1 (* best accuracy epoch)'],
           '01_validation_learning_curves.png',True)
    curves(['train_loss','val_loss'],['Training loss (augmented images)','Validation loss (deterministic preprocessing)'],
           '02_loss_curves.png')
    fig,axes = plt.subplots(1,2,figsize=(12,5));x=np.arange(3)
    for ax,key,title in zip(axes,['accuracy','f1_macro'],['Validation accuracy','Validation macro F1']):
        for mode,offset,color in [(0,-.18,'#2563eb'),(1,.18,'#f97316')]:
            bars=ax.bar(x+offset,[100*rows[i][key] for i in range(mode,6,2)],width=.36,label='FE' if mode==0 else 'FT',color=color)
            ax.bar_label(bars,fmt='%.2f',padding=3)
        ax.set(xticks=x,xticklabels=['ResNet18','VGG16','DenseNet121'],ylabel='Percent',title=title)
        ax.set_ylim(0,min(103,max(100*r[key] for r in rows)+10));ax.legend()
    save(fig,'03_transfer_strategy_comparison.png')
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    for i,(r,c) in enumerate(zip(rows,resources)):
        for ax,value in zip(axes,[c['trainable_params']/1e6,c['chain_seconds_total']/60]):
            ax.scatter(value,100*r['accuracy'],color=colors[i],marker='s' if i%2 else 'o',s=65,label=labels[i])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.035),ncol=3,fontsize=10)
    axes[0].set_xscale('log')
    axes[0].set(xlabel='Trainable parameters (millions, log scale)',ylabel='Validation accuracy (%)',title='Quality and trainable size')
    axes[1].set(xlabel='Recorded chain minutes (FE alone / FE + FT)',ylabel='Validation accuracy (%)',title='Quality and total chain cost')
    save(fig,'04_accuracy_vs_cost.png')
    fig,axes = plt.subplots(1,2,figsize=(13,5));y=np.arange(6)
    for ax,key1,key2,title,scale in [(axes[0],'train_seconds_total','val_seconds_total','Total stage time (minutes)',60),
                                    (axes[1],'train_seconds_per_epoch_median','val_seconds_per_epoch_median','Median phase seconds/epoch (all epochs)',1)]:
        a=np.array([c[key1]/scale for c in resources]);b=np.array([c[key2]/scale for c in resources])
        ax.barh(y,a,label='Train',color='#2563eb');ax.barh(y,b,left=a,label='Validation',color='#f97316')
        ax.set(yticks=y,yticklabels=labels,title=title);ax.invert_yaxis();ax.legend()
    save(fig,'05_runtime_comparison.png')
    fig,axes = plt.subplots(1,2,figsize=(13,5))
    for ax,key,title in [(axes[0],'peak_cuda_allocated_mib','PyTorch peak CUDA allocated (MiB)'),
                         (axes[1],'cpu_rss_observed_peak_mib','Main-process observed RSS peak (MiB)')]:
        values=[max([c[f'{p}_{key}'] for p in ['train','val'] if c[f'{p}_{key}'] is not None],default=None) for c in resources]
        for i,v in enumerate(values):
            if v is None:ax.text(0,i,'N/A',va='center')
            else:ax.barh(i,v,color=colors[i])
        ax.set(yticks=y,yticklabels=labels,title=title);ax.invert_yaxis()
    save(fig,'06_memory_comparison.png')
    cm=np.asarray(results[selected]['cm']);fig,ax=plt.subplots(figsize=(10,9))
    image=ax.imshow(cm,cmap='Blues');fig.colorbar(image,ax=ax,label='Number of validation images')
    ax.grid(False);ax.set(xticks=range(10),yticks=range(10),xlabel='Predicted class',ylabel='True class',
                         title=f'Selected checkpoint / validation ({cm.sum()} images)')
    ax.set_xticklabels(CLASSES,rotation=45,ha='right');ax.set_yticklabels(CLASSES)
    for i in range(10):
        for j in range(10):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=10,color='white' if cm[i,j]>cm.max()/2 else 'black')
    save(fig,'07_best_validation_confusion_matrix.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=BASE_DIR)
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    root=args.output_root.resolve()
    runs,rows,resources,results,verification=collect(root,args.data_dir,device)
    write_csv(root/'reports/validation_best_metrics.csv',rows)
    write_csv(root/'reports/resource_comparison.csv',resources)
    save_json(root/'reports/validation_verification.json',verification)
    plot_evidence(runs,rows,resources,results,root/'reports/figures',verification['selected_run_id'])
    print('Validation/test evidence verified; tables and seven figures exported.')


if __name__=='__main__':
    main()
