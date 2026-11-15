"""Functional checks only. Run: python LAB2/tests.py; no downloads or official test inference."""
import gc
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train
from data import BASE_DIR, AUGMENTATION, check_metadata, load_split, training_loaders
from metrics import classification_metrics, file_hash, save_json, write_csv
from model import CNN, parameter_statistics
from TrainReport import build_report, load_runs


class TinyCNN(nn.Module):
    """Exercise the real checkpoint/optimizer/RNG loop without expensive convolutions."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4,12),nn.Dropout(0.3),nn.Linear(12,10))

    def forward(self,x):
        return self.network(x)

    def parameter_groups(self,lr_head,lr_backbone=None):
        return [dict(name='head',params=list(self.parameters()),lr=lr_head)]


def small_loaders(config,root,device,generator,smoke):
    ds = TensorDataset(torch.arange(48,dtype=torch.float32).reshape(12,4)/48,torch.arange(12)%10)
    return (DataLoader(ds,batch_size=4,shuffle=True,generator=generator),
            DataLoader(ds,batch_size=5),load_split()[2])


def frozen_digest(model):
    h = hashlib.sha256()
    for p in model.parameters():
        if not p.requires_grad:
            h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        (BASE_DIR/'smoke').mkdir(exist_ok=True)

    def temporary_root(self):
        temp = tempfile.TemporaryDirectory(prefix='unit_',dir=BASE_DIR/'smoke')
        root = Path(temp.name).resolve()
        assert root.parent==(BASE_DIR/'smoke').resolve()
        self.addCleanup(temp.cleanup)
        return root

    def test_model_counts_and_modes(self):
        counts = {'resnet18':(11181642,5130,8398858),'vgg16':(134301514,40970,7120394),
                  'densenet121':(6964106,10250,2170378)}
        for name,(total,fe,ft) in counts.items():
            for fine_tune,expected in [(False,fe),(True,ft)]:
                with torch.device('meta'):
                    model = CNN(name,pretrained=False,fine_tune=fine_tune)
                    self.assertEqual(tuple(model(torch.empty(1,3,224,224)).shape),(1,10))
                self.assertEqual(parameter_statistics(model),dict(n_params=total,trainable_params=expected))
        with self.assertRaises(ValueError): CNN('custom')
        with self.assertRaises(ValueError): CNN(num_classes=True)

    def test_real_gradients_freezing_and_stage_transition(self):
        for name in ['resnet18','vgg16','densenet121']:
            with self.subTest(architecture=name):
                parent = CNN(name,pretrained=False)
                x,y = torch.randn(1,3,224,224),torch.tensor([2])
                frozen = frozen_digest(parent)
                optimizer = torch.optim.SGD(parent.parameter_groups(0.01),lr=0.01)
                parent.train()
                nn.functional.cross_entropy(parent(x),y).backward()
                self.assertTrue(all(p.grad is None for p in parent.parameters() if not p.requires_grad))
                optimizer.step()
                self.assertEqual(frozen,frozen_digest(parent))
                parent.eval()
                with torch.no_grad(): before = parent(x)
                child = CNN(name,pretrained=False,fine_tune=True)
                child.load_state_dict(parent.state_dict(),strict=True)
                self.assertTrue(all(torch.equal(v,parent.state_dict()[k]) for k,v in child.state_dict().items()))
                with torch.no_grad():
                    child.eval()
                    torch.testing.assert_close(child(x),before,rtol=0,atol=0)
                del parent,optimizer
                child.train().eval().train()
                self.assertTrue(all(not m.training for m in child.modules() if isinstance(m,(nn.BatchNorm2d,nn.Dropout))))
                buffers = {k:v.clone() for k,v in child.named_buffers()}
                frozen = frozen_digest(child)
                groups = child.parameter_groups(0.01,0.001)
                self.assertEqual(len(set(id(p) for g in groups for p in g['params'])),sum(p.requires_grad for p in child.parameters()))
                optimizer = torch.optim.SGD(groups,lr=0.01)
                # A second target avoids a saturated loss after fitting one synthetic sample.
                nn.functional.cross_entropy(child(x),(y+1)%10).backward()
                for group in groups:
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in group['params']))
                changed = [[(p,p.detach().clone()) for p in g['params']] for g in groups]
                optimizer.step()
                self.assertTrue(all(any(not torch.equal(p,old) for p,old in group) for group in changed))
                self.assertEqual(frozen,frozen_digest(child))
                self.assertTrue(all(torch.equal(v,buffers[k]) for k,v in child.named_buffers()))
                del child,optimizer,groups,changed
                gc.collect()

    def test_data_and_metadata(self):
        cfg = dict(train.Config_parameter[0],batch_size=2,workers=0)
        a,b,identity = training_loaders(cfg,BASE_DIR/'data/prepared',torch.device('cpu'),torch.Generator().manual_seed(42),True)
        x,y = next(iter(a))
        self.assertEqual(tuple(x.shape),(2,3,224,224))
        self.assertEqual(y.dtype,torch.long)
        self.assertTrue(torch.isfinite(x).all())
        torch.testing.assert_close(b.dataset[0][0],b.dataset[0][0],rtol=0,atol=0)
        self.assertEqual(identity,'8e505038999c282a5d13b0649f54f63aabb1e3f985f3a17522f5121f6fcb751a')
        self.assertEqual(len(b.dataset),32)

    def test_metrics_undefined_and_invalid_inputs(self):
        y = np.arange(10)
        p = np.eye(10)*0.9+0.01
        result = classification_metrics(y,p,0.1)
        self.assertEqual(result['metrics']['accuracy'],1)
        self.assertEqual(result['metrics']['f1_macro'],1)
        self.assertEqual(result['metrics']['fpr_macro'],0)
        result = classification_metrics(np.zeros(3,dtype=int),np.ones((3,10))/10,1)
        self.assertIsNone(result['metrics']['auc_macro_ovr'])
        self.assertTrue(all(r is None for r in result['roc']['tpr']))
        with self.assertRaises(ValueError): classification_metrics(y,p*2,1)
        with self.assertRaises(ValueError): classification_metrics(y.astype(float),p,1)

    def test_four_workers_reproduce_augmented_batches(self):
        cfg = dict(train.Config_parameter[0],batch_size=2,workers=4)
        batches = []
        indices = load_split()[0]
        for _ in range(2):
            loader,validation,_ = training_loaders(cfg,BASE_DIR/'data/prepared',torch.device('cpu'),
                                                   torch.Generator().manual_seed(42),True)
            loader.dataset.indices = indices[:8].tolist()
            values = list(loader)
            batches.append((torch.cat([x for x,y in values]),torch.cat([y for x,y in values])))
            del loader,validation,values
            gc.collect()
        torch.testing.assert_close(batches[0][0],batches[1][0],rtol=0,atol=0)
        self.assertTrue(torch.equal(batches[0][1],batches[1][1]))

    def test_resume_matches_uninterrupted_training(self):
        root = self.temporary_root()
        cfg = dict(train.Config_parameter[0],epochs=3,patience=10,batch_size=4,val_batch_size=5)
        device = torch.device('cpu')
        with patch.object(train,'CNN',TinyCNN),patch.object(train,'training_loaders',small_loaders):
            train.train_run(cfg,root/'continuous',BASE_DIR/'data/prepared',device,smoke=True)
            original,calls = train.run_epoch,0
            def interrupt(*args,**kwargs):
                nonlocal calls
                calls += 1
                if calls==3: raise KeyboardInterrupt('intentional unit interruption')
                return original(*args,**kwargs)
            with patch.object(train,'run_epoch',interrupt),self.assertRaises(KeyboardInterrupt):
                train.train_run(cfg,root/'resumed',BASE_DIR/'data/prepared',device,smoke=True)
            last = root/'resumed/checkpoints'/cfg['name']/'last_checkpoint.pt'
            original_hash = file_hash(last)
            with self.assertRaises(ValueError):
                train.train_run(dict(cfg,batch_size=8),root/'resumed',BASE_DIR/'data/prepared',device,resume=True,smoke=True)
            self.assertEqual(file_hash(last),original_hash)
            train.train_run(cfg,root/'resumed',BASE_DIR/'data/prepared',device,resume=True,smoke=True)
        def load(name):
            return torch.load(root/name/'checkpoints'/cfg['name']/'last_checkpoint.pt',weights_only=True)
        a,b = load('continuous'),load('resumed')
        self.assertEqual(a['best_epoch'],b['best_epoch'])
        self.assertEqual(a['scheduler_state_dict'],b['scheduler_state_dict'])
        self.assertTrue(all(torch.equal(v,b['model_state_dict'][k]) for k,v in a['model_state_dict'].items()))
        self.assertEqual([e['epoch'] for e in b['history']['epochs']],[1,2,3])
        self.assertEqual([e['metrics'] for e in a['history']['epochs']],[e['metrics'] for e in b['history']['epochs']])
        check_metadata(b,smoke=True)
        with self.assertRaises(ValueError): check_metadata(b,smoke=False)
        b['augmentation_spec'] = dict(AUGMENTATION,flip_p=0)
        with self.assertRaises(ValueError): check_metadata(b,smoke=True)

    def test_empty_report_preserves_pending_runs(self):
        root = self.temporary_root()
        runs = load_runs(root,[c['name'] for c in train.Config_parameter])
        page = build_report(runs,root/'report.html').read_text(encoding='utf-8')
        self.assertTrue(all(c['name'] in page for c in train.Config_parameter))
        self.assertNotIn('const D=',page)
        self.assertEqual(len(runs),6)

    def test_missing_parent_and_nonfinite_loss_fail_without_updates(self):
        root = self.temporary_root()
        cfg = dict(train.Config_parameter[1],batch_size=4)
        with patch.object(train,'training_loaders',small_loaders),self.assertRaises(FileNotFoundError):
            train.train_run(cfg,root,BASE_DIR/'data/prepared',torch.device('cpu'),smoke=True)
        history = json.loads((root/'reports'/cfg['name']/'history.json').read_text(encoding='utf-8'))
        self.assertEqual(history['status'],'failed')
        self.assertFalse((root/'checkpoints'/cfg['name']/'last_checkpoint.pt').exists())
        model = TinyCNN()
        optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
        original = {k:v.clone() for k,v in model.state_dict().items()}
        with patch.object(model,'forward',return_value=torch.full((4,10),float('nan'))),self.assertRaises(FloatingPointError):
            train.run_epoch(model,[(torch.zeros(4,4),torch.zeros(4,dtype=torch.long))],torch.device('cpu'),
                            optimizer,torch.amp.GradScaler('cuda',enabled=False))
        self.assertTrue(all(torch.equal(v,original[k]) for k,v in model.state_dict().items()))

    def test_pdf_bundle_rejects_changed_values_even_with_updated_hash(self):
        from data import CLASSES
        from PredictPDF import _read_artifacts
        from PIL import Image
        root = self.temporary_root()
        labels = np.arange(32)%10
        probabilities = np.eye(10)[labels]*0.9+0.01
        result = classification_metrics(labels,probabilities,0.1)
        predictions = [dict(sample_index=i,true_label=int(y),pred_label=int(y),confidence=float(p[y]),correct=1,
                            **{f'prob_{k}':float(p[k]) for k in range(10)}) for i,(y,p) in enumerate(zip(labels,probabilities))]
        write_csv(root/'test_predictions.csv',predictions)
        per_class = [dict(r,class_name=CLASSES[r['class_id']]) for r in result['per_class']]
        write_csv(root/'test_classification_report.csv',per_class)
        for name in ['confusion_matrix.png','roc_ovr.png','prediction_examples.png']:
            Image.new('RGB',(4,4)).save(root/name)
        payload = dict(dataset='CIFAR-10',classes=CLASSES,is_smoke=True,split='validation_smoke',metrics=result['metrics'],
                       confusion_matrix=result['cm'],per_class=per_class,selected_indices=[],
                       artifact_hashes={p.name:file_hash(p) for p in root.iterdir()})
        save_json(root/'test_metrics.json',payload)
        self.assertEqual(len(_read_artifacts(root)[1]),32)
        predictions[0]['confidence'] = 0.01
        write_csv(root/'test_predictions.csv',predictions)
        payload['artifact_hashes']['test_predictions.csv'] = file_hash(root/'test_predictions.csv')
        save_json(root/'test_metrics.json',payload)
        with self.assertRaisesRegex(ValueError,'confidence'):
            _read_artifacts(root)


if __name__=='__main__':
    unittest.main(verbosity=2)
