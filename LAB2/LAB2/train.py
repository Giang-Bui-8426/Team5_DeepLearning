"""Six sequential CIFAR-10 runs; TensorBoard, checkpoints and one offline HTML."""
import argparse
import copy
import gc
import json
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from data import (AUGMENTATION, BASE_DIR, CLASSES, DATA_DIR, augmentation_preview,
                  check_metadata, training_loaders, transform_spec)
from metrics import classification_metrics, file_hash, save_json, write_csv
from model import CNN, WEIGHTS, architecture_text, parameter_statistics
from TrainReport import RunRecorder, build_report, load_runs

Config_parameter = []
for index, architecture in enumerate(WEIGHTS):
    for fine_tune in [False, True]:
        Config_parameter.append(dict(
            name=f'{2*index+1+int(fine_tune):02d}_{architecture}_'+('fine_tune' if fine_tune else 'feature_extract'),
            architecture=architecture, fine_tune=fine_tune,
            warm_start_from=f'{2*index+1:02d}_{architecture}_feature_extract' if fine_tune else None,
            lr_head=1e-4 if fine_tune else 1e-3, lr_backbone=1e-5 if fine_tune else None,
            weight_decay=1e-4, epochs=10, batch_size=16, val_batch_size=32,
            workers=4, seed=42, patience=3, min_delta=1e-4, max_grad_norm=1.0,
            scheduler='cosine', eta_min=1e-6))


def choose_device(value):
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    if value == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return torch.device(value)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state(generator):
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), n[2], n[3], n[4]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    generator.set_state(state['loader'].cpu())


def atomic_checkpoint(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def save_validation(value, path):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, y_true=value['y_true'].cpu().numpy(), y_prob=value['y_prob'].cpu().numpy())
    temp.replace(path)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_epoch(model, loader, device, optimizer=None, scaler=None, max_grad_norm=1.0):
    training = optimizer is not None
    model.train(training)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    rss, images, correct, loss_sum, skipped = 0, 0, 0, 0.0, 0
    norms, labels, probabilities = [], [], []
    with (torch.enable_grad() if training else torch.inference_mode()):
        for batch_index,(x,y) in enumerate(loader,1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=training and device.type == 'cuda', dtype=torch.float16):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
            if logits.shape != (len(y), 10) or not torch.isfinite(loss):
                raise FloatingPointError('Invalid logits or nonfinite loss')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_grad_norm))
                if not np.isfinite(norm) and not scaler.is_enabled():
                    raise FloatingPointError('Nonfinite FP32 gradient')
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += scaler.get_scale() < previous_scale
                if np.isfinite(norm):
                    norms.append(norm)
            else:
                labels.append(y.cpu())
                probabilities.append(logits.softmax(1).cpu())
            loss_sum += loss.detach().float().item() * len(y)
            correct += int((logits.argmax(1) == y).sum())
            images += len(y)
            rss = max(rss, psutil.Process().memory_info().rss)
            if training and batch_index%500==0:
                print(f'  train batch {batch_index}/{len(loader)}',flush=True)
    if not images or (training and skipped == len(loader)):
        raise FloatingPointError('Empty epoch or AMP skipped every update')
    synchronize(device)
    seconds = time.perf_counter() - start
    result = dict(loss=loss_sum/images, accuracy=correct/images, images=images, seconds=seconds,
                  grad_norm_mean=float(np.mean(norms)) if norms else None,
                  grad_norm_max=max(norms) if norms else None,
                  clip_frac=sum(n > max_grad_norm for n in norms)/len(norms) if norms else None,
                  skipped_amp_updates=int(skipped), resources=dict(cpu_rss_observed_peak_mib=rss/2**20,
                      peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None,
                      peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20 if device.type == 'cuda' else None))
    if not training:
        result.update(labels=torch.cat(labels).numpy(), probabilities=torch.cat(probabilities).numpy())
        result['classification'] = classification_metrics(result['labels'], result['probabilities'], result['loss'])
    return result


def train_run(config, root, data_dir, device, *, resume=False, smoke=False):
    name = config['name']
    folder, reports = root/'checkpoints'/name, root/'reports'
    last, best, history_path = folder/'last_checkpoint.pt', folder/'checkpoint_best.pt', reports/name/'history.json'
    if not resume and any(p.exists() for p in [last, best, history_path]):
        raise FileExistsError(f'{name} already exists; resume or choose another output root')
    if resume and not last.is_file():
        raise FileNotFoundError(f'Resume checkpoint missing: {last}')
    state = torch.load(last, map_location='cpu', weights_only=True) if resume else None
    hashes = {n:file_hash(BASE_DIR/n) for n in ['model.py','data.py','metrics.py','train.py']}
    if state:
        check_metadata(state, smoke=smoke)
        if state['config'] != config or state['code_sha256'] != hashes:
            raise ValueError('Resume config/source mismatch; use a new experiment')
    seed_everything(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    metadata = dict(run_id=name, architecture=config['architecture'], config=config, classes=CLASSES,
                    mode='fine_tune' if config['fine_tune'] else 'feature_extract',
                    dataset='CIFAR-10', is_smoke=smoke, code_sha256=hashes)
    recorder = RunRecorder(metadata, reports, copy.deepcopy(state['history']) if state else None)
    writer = None
    try:
        train_loader, val_loader, split_hash = training_loaders(config, data_dir, device, generator, smoke)
        parent = None
        parent_path = root/'checkpoints'/str(config['warm_start_from'])/'checkpoint_best.pt'
        if config['fine_tune'] and not state:
            parent = torch.load(parent_path, map_location='cpu', weights_only=True)
            check_metadata(parent, smoke=smoke)
            parent_history = json.loads((reports/config['warm_start_from']/'history.json').read_text(encoding='utf-8'))
            if (parent['run_id'] != config['warm_start_from'] or parent['architecture'] != config['architecture']
                    or parent['config']['fine_tune'] or parent['code_sha256'] != hashes
                    or parent_history['status'] not in ['completed','early_stopped']
                    or parent_history['best_epoch'] != parent['epoch']):
                raise ValueError('Parent is incomplete or incompatible')
        model = CNN(config['architecture'], pretrained=not (smoke or state or config['fine_tune']),
                    fine_tune=config['fine_tune'])
        if state or parent:
            source = (state or parent)['model_state_dict']
            model.load_state_dict(source, strict=True)
            if parent and not all(torch.equal(v, source[k]) for k,v in model.state_dict().items()):
                raise AssertionError('Fine-tuning changed parent weights before first update')
        metadata.update(parameter_statistics(model), split_sha256=split_hash,
                        transform_spec=transform_spec(config['architecture']), augmentation_spec=AUGMENTATION,
                        weights_enum=str(WEIGHTS[config['architecture']]), device=str(device),
                        hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                        precision='fp16_amp' if device.type == 'cuda' else 'fp32', validation_precision='fp32',
                        versions=dict(python=platform.python_version(), torch=str(torch.__version__), torchvision=str(torchvision.__version__)),
                        deterministic=dict(cudnn_benchmark=False,cudnn_deterministic=True),
                        init_source='random_smoke' if smoke and not config['fine_tune'] else 'parent_best_checkpoint' if config['fine_tune'] else 'ImageNet',
                        parent_run_id=config['warm_start_from'], parent_checkpoint_path=str(parent_path) if parent else None,
                        parent_checkpoint_sha256=file_hash(parent_path) if parent else None,
                        parent_best_epoch=parent['best_epoch'] if parent else None,
                        parent_metrics=parent['best_metrics'] if parent else None)
        if state:
            metadata.update({k:v for k,v in state.items() if k.startswith('parent_')})
        recorder.data.update(metadata)
        (reports/name/'architecture.txt').write_text(architecture_text(model), encoding='utf-8')
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameter_groups(config['lr_head'],config['lr_backbone']),
                                      weight_decay=config['weight_decay'], betas=(0.9,0.999),eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['epochs'],eta_min=config['eta_min'])
        scaler = torch.amp.GradScaler('cuda',enabled=device.type=='cuda',init_scale=1024)
        start, best_accuracy, best_epoch, best_metrics, best_loss, counter, best_validation = 1,-1.0,0,None,float('inf'),0,None
        if state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])
            scaler.load_state_dict(state['scaler_state_dict'])
            start, best_epoch = state['epoch']+1,state['best_epoch']
            best_accuracy, best_metrics = state['best_accuracy'],state['best_metrics']
            best_loss, counter, best_validation = state['best_val_loss'],state['early_stopping_counter'],state['best_validation']
            # Last is saved first; repair a best/NPZ write interrupted immediately after it.
            if state['epoch'] == best_epoch:
                atomic_checkpoint(state,best)
                save_validation(best_validation,reports/name/'best_validation.npz')
            elif not best.is_file():
                raise FileNotFoundError('Earlier best checkpoint is missing')
            restore_rng(state['rng_state'],generator)
        writer = SummaryWriter(str(root/'logs'/name),purge_step=start if resume else None)
        writer.add_text('Config',json.dumps(metadata,ensure_ascii=False,indent=2),start)
        del parent, state
        print(f'{name}: {device}, epoch {start}/{config["epochs"]}',flush=True)
        status = 'early_stopped' if counter >= config['patience'] else 'completed'
        for epoch in range(start,config['epochs']+1):
            if counter >= config['patience']:
                break
            lr = {g['name']:g['lr'] for g in optimizer.param_groups}
            train = run_epoch(model,train_loader,device,optimizer,scaler,config['max_grad_norm'])
            val = run_epoch(model,val_loader,device)
            metrics = val['classification']['metrics']
            improved = metrics['accuracy'] > best_accuracy
            if improved:
                best_accuracy,best_epoch,best_metrics = metrics['accuracy'],epoch,metrics
                best_validation = dict(y_true=torch.from_numpy(val['labels']),y_prob=torch.from_numpy(val['probabilities']))
            if metrics['loss'] < best_loss-config['min_delta']:
                best_loss,counter = metrics['loss'],0
            else:
                counter += 1
            e = recorder.log_epoch(epoch,train,val,lr,improved)
            scalars = {'Loss/train_epoch':train['loss'],'Train/accuracy':train['accuracy'],
                       'Loss/validation':metrics['loss'],
                       **{f'Validation/{k}':metrics[k] for k in ['accuracy','precision_macro','recall_macro','f1_macro']},
                       **{f'Optimization/lr_{k}':v for k,v in lr.items()},
                       **{f'Optimization/{k}':train[k] for k in ['grad_norm_mean','grad_norm_max','clip_frac','skipped_amp_updates']}}
            for phase,result in [('train',train),('val',val)]:
                scalars.update({f'Resources/{phase}_{k}':v for k,v in result['resources'].items()})
                scalars.update({f'Performance/{phase}_seconds':result['seconds'],
                                f'Performance/{phase}_images_per_second':result['images']/result['seconds']})
            for tag,value in scalars.items():
                if value is not None:
                    writer.add_scalar(tag,value,epoch)
            writer.flush()
            scheduler.step()
            status = 'early_stopped' if counter>=config['patience'] else 'completed' if epoch==config['epochs'] else 'running'
            recorder.data.update(status=status,best_metrics=best_metrics)
            payload = dict(metadata,epoch=epoch,best_epoch=best_epoch,best_accuracy=best_accuracy,best_metrics=best_metrics,
                           best_val_loss=best_loss,early_stopping_counter=counter,status=status,
                           model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                           scheduler_state_dict=scheduler.state_dict(),scaler_state_dict=scaler.state_dict(),
                           rng_state=rng_state(generator),history=recorder.data,best_validation=best_validation)
            atomic_checkpoint(payload,last)
            if improved:
                atomic_checkpoint(payload,best)
                save_validation(best_validation,reports/name/'best_validation.npz')
            recorder.save()
            print(f'  epoch {epoch}: train {train["loss"]:.4f}, val {metrics["loss"]:.4f}, '
                  f'accuracy {metrics["accuracy"]:.2%}, best {best_epoch}',flush=True)
        recorder.finish(status)
    except BaseException as exc:
        recorder.finish('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',repr(exc))
        raise
    finally:
        if writer:
            writer.close()
        gc.collect()


def summarize(root, configs):
    runs = load_runs(root/'reports',[c['name'] for c in configs])
    rows = []
    for r in runs:
        epochs = r.get('epochs',[])
        best = next((e for e in epochs if e['epoch']==r.get('best_epoch')), {})
        parent = next((p for p in runs if p['name']==r.get('parent_run_id')),None)
        seconds = sum(e['epoch_time'] for e in epochs) if epochs else None
        row = dict(run_id=r['name'],status=r['status'],architecture=r.get('architecture'),mode=r.get('mode'),
                   best_epoch=r.get('best_epoch'),epochs_run=len(epochs),n_params=r.get('n_params'),
                   trainable_params=r.get('trainable_params'),config=json.dumps(r.get('config',{})),
                   device=r.get('device'),precision=r.get('precision'),parent_run_id=r.get('parent_run_id'),
                   parent_checkpoint_sha256=r.get('parent_checkpoint_sha256'),stage_seconds=seconds,
                   chain_seconds=(seconds+sum(e['epoch_time'] for e in parent['epochs'])
                                  if seconds is not None and parent and parent.get('epochs')
                                  else seconds if not r.get('parent_run_id') else None))
        row.update(best.get('metrics',{}))
        for phase in ['train','val']:
            total = sum(e[f'{phase}_time'] for e in epochs)
            row[f'{phase}_images_per_second'] = sum(e[f'{phase}_images'] for e in epochs)/total if total else None
            for key in ['peak_cuda_allocated_mib','peak_cuda_reserved_mib','cpu_rss_observed_peak_mib']:
                values = [e[f'{phase}_{key}'] for e in epochs if e.get(f'{phase}_{key}') is not None]
                row[f'{phase}_{key}'] = max(values) if values else None
        rows.append(row)
    # Pending and completed rows share one schema.
    fields = list(dict.fromkeys(k for row in rows for k in row))
    root.joinpath('reports').mkdir(parents=True,exist_ok=True)
    write_csv(root/'reports/run_summary.csv',[{k:r.get(k) for k in fields} for r in rows])
    build_report(runs,root/'reports/report_all_models.html')
    if any(r.get('is_smoke') for r in runs) or not all(r['status'] in ['completed','early_stopped'] for r in runs):
        return
    for r in runs:
        cp = torch.load(root/'checkpoints'/r['name']/'checkpoint_best.pt',map_location='cpu',weights_only=True,mmap=True)
        check_metadata(cp)
        if cp['epoch']!=r['best_epoch'] or cp['best_metrics']!=r['best_metrics']:
            raise ValueError('Best checkpoint/history mismatch')
        del cp
    rank = sorted(runs,key=lambda r:(-r['best_metrics']['accuracy'],-r['best_metrics']['f1_macro'],r['best_metrics']['loss'],r['name']))
    selected = rank[0]
    target = root/'checkpoints/checkpoint_best_global.pt'
    shutil.copyfile(root/'checkpoints'/selected['name']/'checkpoint_best.pt',target.with_suffix('.tmp'))
    target.with_suffix('.tmp').replace(target)
    save_json(root/'reports/selection.json',dict(run_id=selected['name'],best_epoch=selected['best_epoch'],
              run_status=selected['status'],checkpoint_sha256=file_hash(target),split_sha256=selected['split_sha256'],
              rule='accuracy desc, macro F1 desc, loss asc, run ID asc',ranked_runs=[r['name'] for r in rank]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='all',choices=['all']+[c['name'] for c in Config_parameter])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='isolated functional check; random initialization, small train/validation subsets')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--output-root',type=Path,default=BASE_DIR)
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
    for key in ['epochs','batch-size','val-batch-size','workers']:
        parser.add_argument('--'+key,type=int)
    args = parser.parse_args()
    configs = copy.deepcopy(Config_parameter)
    for cfg in configs:
        for key in ['epochs','batch_size','val_batch_size','workers']:
            value = getattr(args,key)
            if value is not None:
                if value < (0 if key=='workers' else 1):
                    parser.error(f'Invalid {key}')
                cfg[key] = value
        if args.smoke:
            cfg['epochs'] = 2
    root = args.output_root.resolve()
    if args.smoke and root==BASE_DIR:
        root = BASE_DIR/'smoke'/args.device
    if not args.smoke and BASE_DIR/'smoke' in root.parents:
        parser.error('Official training cannot use the smoke directory')
    selected = configs if args.run=='all' else [c for c in configs if c['name']==args.run]
    device = choose_device(args.device)
    torch.set_num_threads(4)
    # Refused launches must not regenerate reports in an existing experiment.
    hashes = {n:file_hash(BASE_DIR/n) for n in ['model.py','data.py','metrics.py','train.py']}
    for cfg in selected:
        history = root/'reports'/cfg['name']/'history.json'
        last = root/'checkpoints'/cfg['name']/'last_checkpoint.pt'
        best = last.with_name('checkpoint_best.pt')
        if not args.resume and any(p.exists() for p in [history,last,best]):
            raise FileExistsError('Run output exists; resume or choose another output root')
        if args.resume and history.exists():
            cp = torch.load(last,map_location='cpu',weights_only=True,mmap=True)
            check_metadata(cp,smoke=args.smoke)
            if cp['config']!=cfg or cp['code_sha256']!=hashes:
                raise ValueError('Resume config/source mismatch')
            del cp
    try:
        preview = root/'reports/augmentation_preview.png'
        if not preview.exists():
            augmentation_preview(preview,args.data_dir.resolve())
        for cfg in selected:
            history = root/'reports'/cfg['name']/'history.json'
            if args.resume and history.is_file():
                previous = json.loads(history.read_text(encoding='utf-8'))
                if previous['status'] in ['completed','early_stopped']:
                    continue
            train_run(cfg,root,args.data_dir.resolve(),device,resume=args.resume and history.exists(),smoke=args.smoke)
            summarize(root,configs)
    finally:
        import sys
        active_error = sys.exc_info()[0] is not None
        try:
            summarize(root,configs)
        except Exception as exc:
            if not active_error:
                raise
            print(f'Report export also failed: {exc}',flush=True)


if __name__ == '__main__':
    main()
