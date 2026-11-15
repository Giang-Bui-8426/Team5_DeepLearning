
"""
TrainReport.py - ghi lại lịch sử huấn luyện và xuất báo cáo HTML (tự chứa, không cần internet).

Cách dùng trong Train.py:
    from TrainReport import RunRecorder, build_report, load_runs, fig_confusion, fig_roc

    recorder = RunRecorder(name, config, classes, REPORT_DIR, n_params=..., device=device,
                           resume_from=start_epoch)
    recorder.log_batch(epoch + 1, global_step, loss)            # (tuỳ chọn) loss theo batch
    e = recorder.log_epoch(epoch, train_dict, val_dict, y_true, y_prob,
                           lr=..., weight_norm=..., train_time=..., val_time=..., is_best=...)
    recorder.save()                                             # ghi history.json sau mỗi epoch
    recorder.finish(early_stopped=False)                        # ghi report.html của model đó

    build_report(load_runs(REPORT_DIR, names), REPORT_DIR / "report_all_models.html")

Dựng lại báo cáo từ dữ liệu đã lưu (không cần train lại):
    python TrainReport.py ./reports
"""
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import auc as _auc, roc_curve as _roc_curve

# Lưới FPR dùng chung để lưu đường ROC (dày hơn ở vùng FPR nhỏ, nơi đường cong thay đổi nhanh)
_GRID = np.unique(np.concatenate([np.linspace(0, 0.1, 60), np.linspace(0.1, 1, 100)]))


# ----------------------------------------------------------------------------
# Tiện ích
# ----------------------------------------------------------------------------
def _clean(o):
    """Chuyển numpy/NaN về kiểu JSON an toàn, làm tròn 9 chữ số có nghĩa."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return float("%.9g" % v) if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _confusion(y_true, y_pred, n):
    return np.bincount(y_true * n + y_pred, minlength=n * n).reshape(n, n)


def _per_class(cm):
    tp = np.diag(cm).astype(float)
    row, col = cm.sum(axis=1).astype(float), cm.sum(axis=0).astype(float)
    precision = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    recall = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, row


def _roc_pack(y_true, y_prob, n):
    onehot = np.eye(n)[y_true]
    tprs, aucs = [], []
    for i in range(n):
        fpr, tpr, _ = _roc_curve(onehot[:, i], y_prob[:, i])
        tprs.append(np.interp(_GRID, fpr, tpr))
        aucs.append(_auc(fpr, tpr))
    return {
        "tpr": np.round(np.array(tprs), 4),
        "macro": np.round(np.mean(tprs, axis=0), 4),
        "auc": np.array(aucs),
    }


# ----------------------------------------------------------------------------
# Ghi lịch sử cho 1 lần chạy (1 model)
# ----------------------------------------------------------------------------
class RunRecorder:
    def __init__(self, name, config, classes, report_dir, n_params=None,
                 device="", resume_from=0):
        self.name = name
        self.config = dict(config)
        self.classes = list(classes)
        self.n_params = n_params
        self.device = str(device)
        self.dir = Path(report_dir) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "history.json"

        self.epochs, self.batches = [], []
        self.best, self.last = None, None
        self.best_epoch = 0
        self.status = "running"
        self.started = time.time()
        self.finished = None
        rel_offset = 0.0

        # Tiếp tục huấn luyện: nạp lại lịch sử cũ, bỏ phần từ epoch resume_from trở đi
        if resume_from > 0 and self.path.exists():
            try:
                old = json.loads(self.path.read_text(encoding="utf-8"))
                self.epochs = [e for e in old.get("epochs", []) if e["epoch"] <= resume_from]
                self.batches = [b for b in old.get("batches", []) if b[0] <= resume_from]
                self.started = old.get("started", self.started)
                self.best_epoch = old.get("best_epoch", 0)
                if self.best_epoch > resume_from:
                    self.best_epoch = 0
                if old.get("best") and old["best"].get("epoch", 0) <= resume_from:
                    self.best = old["best"]
                if self.epochs:
                    rel_offset = self.epochs[-1]["rel"]
            except Exception as exc:  # file hỏng -> bắt đầu lại lịch sử
                print(f"[TrainReport] Không đọc được {self.path}: {exc}")
        self._t0 = time.time() - rel_offset      # 'rel' = thời gian chạy thực tế, bỏ qua lúc tạm dừng

    # -- ghi theo batch ------------------------------------------------------
    def log_batch(self, epoch, step, loss):
        now = time.time()
        self.batches.append([int(epoch), int(step), float(loss), now, now - self._t0])

    # -- ghi cuối mỗi epoch --------------------------------------------------
    def log_epoch(self, epoch, train, val, y_true, y_prob, lr, weight_norm,
                  train_time, val_time, is_best=False):
        n = len(self.classes)
        y_true = np.asarray(y_true, dtype=np.int64)
        y_prob = np.asarray(y_prob, dtype=np.float64)
        cm = _confusion(y_true, y_prob.argmax(axis=1), n)
        precision, recall, f1, support = _per_class(cm)
        roc = _roc_pack(y_true, y_prob, n)
        now = time.time()
        images = train.get("images", 0)
        e = {
            "epoch": int(epoch) + 1,
            "wall": now,
            "rel": now - self._t0,
            "train_time": train_time, "val_time": val_time,
            "epoch_time": train_time + val_time,
            "img_per_sec": images / train_time if train_time > 0 else None,
            "train_loss": train["loss"], "train_acc": train["accuracy"],
            "val_loss": val["loss"], "val_acc": val["accuracy"],
            "precision": val["precision"], "recall": val["recall"], "f1": val["f1"],
            "fpr": val["fpr"], "auc": val["auc"],
            "lr": lr, "weight_norm": weight_norm,
            "grad_norm": train.get("grad_norm_mean"),
            "grad_norm_max": train.get("grad_norm_max"),
            "clip_frac": train.get("clip_frac"),
            "cm": cm,
            "pc": {"precision": precision, "recall": recall, "f1": f1,
                   "auc": roc["auc"], "support": support},
        }
        e = _clean(e)
        self.epochs = [x for x in self.epochs if x["epoch"] != e["epoch"]] + [e]
        self.last = {"epoch": e["epoch"], "roc": _clean(roc)}
        if is_best or self.best is None:
            self.best, self.best_epoch = self.last, e["epoch"]
        return e

    # -- xuất ----------------------------------------------------------------
    def to_dict(self):
        return _clean({
            "name": self.name, "config": self.config, "classes": self.classes,
            "n_params": self.n_params, "device": self.device,
            "status": self.status, "started": self.started, "finished": self.finished,
            "best_epoch": self.best_epoch, "roc_grid": np.round(_GRID, 4),
            "epochs": self.epochs, "batches": self.batches,
            "best": self.best, "last": self.last,
        })

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")
        tmp.replace(self.path)

    def finish(self, early_stopped=False):
        self.status = "early_stopped" if early_stopped else "completed"
        self.finished = time.time()
        self.save()
        return self.write_report()

    def write_report(self):
        return build_report([self.to_dict()], self.dir / "report.html",
                            title=f"Báo cáo huấn luyện - {self.name}")


def load_runs(report_dir, names=None):
    """Đọc history.json đã lưu. names=None -> lấy tất cả thư mục con có history.json."""
    report_dir = Path(report_dir)
    if names is None:
        names = sorted(p.parent.name for p in report_dir.glob("*/history.json"))
    runs = []
    for name in names:
        p = report_dir / name / "history.json"
        if p.exists():
            runs.append(json.loads(p.read_text(encoding="utf-8")))
    return runs


# ----------------------------------------------------------------------------
# Hình matplotlib cho TensorBoard (add_figure)
# ----------------------------------------------------------------------------
def fig_confusion(cm, classes, epoch):
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), classes, rotation=40, ha="right")
    ax.set_yticks(range(n), classes)
    ax.set_xlabel("Dự đoán (predicted)")
    ax.set_ylabel("Nhãn thật (true)")
    ax.set_title(f"Confusion matrix - Epoch {epoch} (màu = recall theo hàng, số = số ảnh)",
                 fontsize=11, fontweight="bold")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    return fig


def fig_roc(roc, classes, epoch):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, name in enumerate(classes):
        ax.plot(_GRID, roc["tpr"][i], lw=1.5, alpha=0.7, label=f"{name} (AUC={roc['auc'][i]:.3f})")
    ax.plot(_GRID, roc["macro"], color="black", lw=2.2, label=f"macro (AUC={np.mean(roc['auc']):.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"ROC Curve - Epoch {epoch}", fontsize=14, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Xuất báo cáo HTML
# ----------------------------------------------------------------------------
def build_report(runs, path, title="Báo cáo huấn luyện"):
    runs = [r for r in runs if r.get("epochs")]
    if not runs:
        print("[TrainReport] Chưa có dữ liệu epoch nào để xuất báo cáo.")
        return None
    data = {"title": title, "generated": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "runs": _clean(runs)}
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\!--")
    page = _TEMPLATE.replace("__TITLE__", title.replace("<", "&lt;")).replace("__DATA__", blob)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    print(f"[TrainReport] Đã ghi báo cáo: {path}")
    return path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f4f6f9;--card:#fff;--text:#1f2430;--muted:#6b7280;--line:#e5e7eb;--accent:#2563eb;--tip:#111827;--tipt:#f9fafb;--hl:rgba(37,99,235,.08)}
[data-theme=dark]{--bg:#0e1117;--card:#171b24;--text:#e6e9ef;--muted:#9aa3b2;--line:#2a3140;--accent:#6ea8ff;--tip:#05070a;--tipt:#e6e9ef;--hl:rgba(110,168,255,.12)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 "Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header.top{position:sticky;top:0;z-index:20;background:var(--card);border-bottom:1px solid var(--line);padding:10px 20px;display:flex;flex-wrap:wrap;gap:12px 24px;align-items:center;justify-content:space-between}
header h1{font-size:17px;margin:0}
header .meta{color:var(--muted);font-size:12px}
.ctrls{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;font-size:12.5px}
.ctrls label{display:flex;align-items:center;gap:6px;color:var(--muted)}
.ctrls input[type=range]{width:130px}
select,button.b,.seg button{font:inherit;font-size:12px;border:1px solid var(--line);background:var(--card);color:var(--text);border-radius:7px;padding:4px 9px;cursor:pointer}
button.b:hover,.seg button:hover{border-color:var(--accent)}
.seg{display:inline-flex}.seg button{border-radius:0;margin-left:-1px}.seg button:first-child{border-radius:7px 0 0 7px;margin:0}.seg button:last-child{border-radius:0 7px 7px 0}
.seg button.on{background:var(--accent);border-color:var(--accent);color:#fff}
nav#tabs{display:flex;gap:6px;flex-wrap:wrap;padding:12px 20px 0}
nav#tabs button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--text);padding:7px 14px;border-radius:9px 9px 0 0;cursor:pointer;border-bottom:none}
nav#tabs button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
nav#tabs i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}nav#tabs button.on i{box-shadow:0 0 0 2px #fff}
main{padding:16px 20px 60px;max-width:1700px;margin:0 auto}
h2{font-size:15px;margin:26px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;box-shadow:0 1px 2px rgba(0,0,0,.04);min-width:0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(500px,1fr));gap:14px;align-items:start}
.ch{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;margin-bottom:6px}
.ch h3{margin:0;font-size:14px}.sub{color:var(--muted);font-size:12px}
.acts{display:flex;gap:5px;flex-shrink:0}
.legend{display:flex;flex-wrap:wrap;gap:4px 10px;margin:2px 0 4px;font-size:12px}
.chip{cursor:pointer;user-select:none;display:inline-flex;align-items:center;gap:5px;padding:1px 4px;border-radius:5px}
.chip i,.dot{display:inline-block;width:10px;height:10px;border-radius:3px}
.chip.off{opacity:.35;text-decoration:line-through}
.mini{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:6px;margin-bottom:2px}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin:6px 0 4px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.kl{font-size:11.5px;color:var(--muted)}.kv{font-size:21px;font-weight:700;margin-top:2px}.ks{font-size:11.5px;color:var(--muted)}
table.tbl{border-collapse:collapse;width:100%;font-size:13px}
.tbl th,.tbl td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
.tbl th:first-child,.tbl td:first-child,.tbl td.l,.tbl th.l{text-align:left}
.tbl th{color:var(--muted);font-weight:600;font-size:12px;position:sticky;top:0;background:var(--card)}
.tbl tr.dim td{opacity:.45}
.tbl td.best{font-weight:700;box-shadow:inset 0 0 0 1.5px #16a34a}
.tbl a{color:var(--accent);text-decoration:none;font-weight:600}
.scroll{overflow:auto;max-height:520px}
.badge{display:inline-block;font-size:11px;padding:1px 7px;border-radius:99px;background:#16a34a;color:#fff;margin-left:6px}
.badge.w{background:#d97706}.badge.g{background:#6b7280}
.cmwrap{display:flex;flex-wrap:wrap;gap:18px;align-items:flex-start}
.cmwrap>.left{flex:0 1 640px;min-width:320px}.cmwrap>.right{flex:1 1 380px;min-width:300px}
.pair{display:grid;grid-template-columns:minmax(150px,1.3fr) 2fr 44px 52px;gap:8px;align-items:center;font-size:12.5px;padding:2px 0}
.pair .bar{height:9px;background:var(--line);border-radius:5px;overflow:hidden}.pair .bar i{display:block;height:100%;background:#dc2626}
.pair small{color:var(--muted)}
.hint{color:var(--muted);font-size:12px;margin:4px 0 8px}
#tip{position:fixed;z-index:50;display:none;background:var(--tip);color:var(--tipt);border-radius:8px;padding:8px 10px;font-size:12px;pointer-events:none;box-shadow:0 6px 24px rgba(0,0,0,.3);max-width:640px}
#tip table{border-collapse:collapse}#tip td,#tip th{padding:2px 8px;text-align:left;white-space:nowrap}
#tip th{font-weight:600;opacity:.65;font-size:11px}#tip i{display:inline-block;width:9px;height:9px;border-radius:50%}
details{margin-top:8px}summary{cursor:pointer;font-weight:600;padding:6px 0}
.cfg td:first-child{color:var(--muted)}
</style>
</head>
<body>
<header class="top">
  <div><h1 id="ttl"></h1><div class="meta" id="meta"></div></div>
  <div class="ctrls">
    <label>Làm mượt <input type="range" id="smooth" min="0" max="0.99" step="0.01" value="0"><b id="smv">0</b></label>
    <label>Trục X <select id="xmode">
      <option value="step">Epoch / Step</option>
      <option value="rel">Thời gian chạy (Relative)</option>
      <option value="wall">Thời gian thực (Wall)</option></select></label>
    <label><input type="checkbox" id="raw" checked> Đường gốc mờ</label>
    <button class="b" id="theme">Sáng / Tối</button>
  </div>
</header>
<nav id="tabs"></nav>
<main id="view"></main>
<div id="tip"></div>
<script>
'use strict';
const D=__DATA__;
const RUNS=D.runs;
const PAL=['#2563eb','#f97316','#16a34a','#dc2626','#9333ea','#0891b2','#ca8a04','#db2777','#475569','#65a30d'];
const C10=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
const LIGHT={bg:'#ffffff',text:'#1f2430',muted:'#6b7280',grid:'#e5e7eb',axis:'#9ca3af',dark:false};
const DARK={bg:'#171b24',text:'#e6e9ef',muted:'#9aa3b2',grid:'#2a3140',axis:'#4b5563',dark:true};
const th=()=>document.documentElement.dataset.theme==='dark'?DARK:LIGHT;
const $=(s,r)=>(r||document).querySelector(s);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fin=v=>v!=null&&isFinite(v);
const fmt=(v,d)=>fin(v)?(+v).toFixed(d==null?4:d):'-';
const pct=(v,d)=>fin(v)?(v*100).toFixed(d==null?2:d)+'%':'-';
const nf=new Intl.NumberFormat('vi-VN');
function fmtDur(s){if(!fin(s))return'-';if(s<60)return s.toFixed(1)+'s';const h=Math.floor(s/3600),m=Math.floor(s%3600/60),c=Math.floor(s%60),p=n=>String(n).padStart(2,'0');return h?`${h}h ${p(m)}m ${p(c)}s`:`${m}m ${p(c)}s`}
function fmtWall(w){if(!fin(w))return'-';const d=new Date(w*1000),p=n=>String(n).padStart(2,'0');return`${p(d.getDate())}/${p(d.getMonth()+1)}/${d.getFullYear()} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`}
function ext(a){let lo=Infinity,hi=-Infinity;for(const v of a){if(v<lo)lo=v;if(v>hi)hi=v}return[lo,hi]}
function ema(y,w){if(!w)return y.slice();const o=[];let last=0,n=0;for(const v of y){if(!fin(v)){o.push(null);continue}last=last*w+(1-w)*v;n++;o.push(last/(1-Math.pow(w,n)))}return o}
function niceNum(x,round){const e=Math.floor(Math.log10(x)),f=x/Math.pow(10,e);let n;if(round)n=f<1.5?1:f<3?2:f<7?5:10;else n=f<=1?1:f<=2?2:f<=5?5:10;return n*Math.pow(10,e)}
function ticks(lo,hi,n){if(!(hi>lo)){hi=lo+1}const st=niceNum(niceNum(hi-lo,false)/(n-1),true),o=[];for(let v=Math.ceil(lo/st-1e-9)*st;v<=hi+st*1e-9;v+=st)o.push(+v.toFixed(12));o.step=st;return o}
function logTicks(lo,hi){const o=[],a=Math.floor(Math.log10(lo)),b=Math.ceil(Math.log10(hi)),many=(b-a)>3;for(let k=a;k<=b;k++)for(const m of(many?[1]:[1,2,5])){const v=m*Math.pow(10,k);if(v>=lo&&v<=hi)o.push(v)}if(o.length<2){o.length=0;o.push(lo,hi)}return o}
const TST=[1,2,5,10,15,30,60,120,300,600,900,1800,3600,7200,10800,21600,43200,86400,172800,432000,864000];
function timeTicks(lo,hi,n,wall){const st=TST.find(s=>(hi-lo)/s<=n)||TST[TST.length-1],off=wall?-new Date(lo*1000).getTimezoneOffset()*60:0,o=[];for(let v=Math.ceil((lo+off)/st)*st-off;v<=hi;v+=st)o.push(v);o.step=st;return o}
function stepTicks(lo,hi,n){if(hi-lo<=n*1.5){const o=[];for(let v=Math.ceil(lo);v<=hi;v++)o.push(v);return o}return ticks(lo,hi,n).filter(Number.isInteger)}
function relLbl(v){if(v<60)return Math.round(v)+'s';if(v<3600)return(v%60?Math.floor(v/60)+'m'+Math.round(v%60)+'s':v/60+'m');const h=Math.floor(v/3600),m=Math.round(v%3600/60);return m?h+'h'+m+'m':h+'h'}
function wallLbl(v,step,span){const d=new Date(v*1000),p=n=>String(n).padStart(2,'0'),hm=p(d.getHours())+':'+p(d.getMinutes())+(step<60?':'+p(d.getSeconds()):'');return span>86400?p(d.getDate())+'/'+p(d.getMonth()+1)+' '+hm:hm}
function yLbl(v,step){if(v===0)return'0';const a=Math.abs(v);if(a>=1e6||(a<1e-4))return v.toExponential(1);const dec=step>=1?0:Math.min(7,Math.ceil(-Math.log10(step)-1e-9));return v.toFixed(dec)}
function mixc(bg,fg,t){return`rgb(${bg.map((b,k)=>Math.round(b+(fg[k]-b)*t)).join(',')})`}
function dl(name,blob){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;document.body.appendChild(a);a.click();setTimeout(()=>{URL.revokeObjectURL(a.href);a.remove()},500)}
function slug(s){return String(s).normalize('NFD').replace(/[\u0300-\u036f]/g,'').replace(/[^a-zA-Z0-9]+/g,'_').replace(/^_|_$/g,'').toLowerCase()}
function exportChart(r,name,f){
  if(f==='svg'){dl(slug(name)+'.svg',new Blob([r.svg],{type:'image/svg+xml'}));return}
  const img=new Image(),u=URL.createObjectURL(new Blob([r.svg],{type:'image/svg+xml;charset=utf-8'})),k=3;
  img.onload=()=>{const c=document.createElement('canvas');c.width=r.W*k;c.height=r.H*k;const x=c.getContext('2d');x.fillStyle='#fff';x.fillRect(0,0,c.width,c.height);x.drawImage(img,0,0,c.width,c.height);URL.revokeObjectURL(u);c.toBlob(b=>dl(slug(name)+'.png',b))};
  img.src=u}
const tip=document.getElementById('tip');
function showTip(h,e){tip.innerHTML=h;tip.style.display='block';const w=tip.offsetWidth,hh=tip.offsetHeight;let x=e.clientX+16,y=e.clientY+14;if(x+w>innerWidth-8)x=Math.max(8,e.clientX-w-16);if(y+hh>innerHeight-8)y=Math.max(8,innerHeight-hh-8);tip.style.left=x+'px';tip.style.top=y+'px'}
function hideTip(){tip.style.display='none'}

/* ---------- chuẩn bị dữ liệu ---------- */
RUNS.forEach((r,i)=>{
  r.idx=i;r.color=PAL[i%PAL.length];r.E=r.epochs||[];r.byEp={};r.E.forEach(e=>{r.byEp[e.epoch]=e});r.cls=r.classes||[];
  let be=r.best_epoch;if(!r.byEp[be])be=r.E.reduce((a,e)=>e.val_acc>a.val_acc?e:a,r.E[0]).epoch;
  r.bestEp=be;r.bm=r.byEp[be];r.totalTime=r.E[r.E.length-1].rel;
  r.ips=r.E.reduce((a,e)=>a+(e.img_per_sec||0),0)/r.E.length;
  r.bestPack=(r.best&&r.best.roc)?r.best:null;r.lastPack=(r.last&&r.last.roc)?r.last:null;
});
const S={view:RUNS.length>1?'cmp':0,smooth:0,xmode:'step',raw:true,sel:new Set(RUNS.map(r=>r.name)),rank:'val_acc',barZero:true,cmMode:'count',cmEp:{},rocWhich:'best',rocOff:{},rocZoom:false};
let UID=0,CHARTS=[];
const MET=[
 {k:'val_acc',n:'Accuracy',up:1,f:v=>pct(v)},{k:'precision',n:'Precision',up:1,f:v=>pct(v)},
 {k:'recall',n:'Recall',up:1,f:v=>pct(v)},{k:'f1',n:'F1',up:1,f:v=>pct(v)},
 {k:'auc',n:'AUC',up:1,f:v=>fmt(v,4)},{k:'fpr',n:'FPR',up:0,f:v=>pct(v,3)},{k:'val_loss',n:'Val loss',up:0,f:v=>fmt(v,4)}];

/* ---------- biểu đồ đường (kiểu TensorBoard) ---------- */
function legendLayout(names,W,ml){let x=ml,y=0,o=[];names.forEach(n=>{const w=26+n.length*6.3;if(x+w>W-10&&x>ml){x=ml;y+=18}o.push({n,x,y});x+=w+8});return{items:o,rows:y/18+1}}
function build(spec,st,W,H,T,ex){
  const mode=S.xmode,sm=spec.ownSmooth!=null?st.smooth:S.smooth,showRaw=S.raw&&sm>0,log=!!st.log;
  const ser=spec.series.filter(s=>!st.hidden.has(s.name)).map(s=>{
    const xs=s[mode],ys=ema(s.y,sm),pts=[];
    for(let i=0;i<xs.length;i++){const y=ys[i];if(y==null||xs[i]==null)continue;if(log&&!(y>0&&s.y[i]>0))continue;pts.push({i,x:xs[i],y,r:s.y[i]})}
    return{s,pts}});
  const lg=ex?legendLayout(spec.series.map(s=>s.name),W,60):null;
  const ml=60,mr=16,mt=ex?46+lg.rows*18:12,mb=44,pw=W-ml-mr,ph=H-mt-mb;
  const head=`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="'Segoe UI',Roboto,Helvetica,Arial,sans-serif" font-size="11"><defs><clipPath id="cp${spec.id}"><rect x="${ml}" y="${mt}" width="${pw}" height="${ph}"/></clipPath></defs><rect width="${W}" height="${H}" fill="${T.bg}"/>`;
  const all=[].concat(...ser.map(o=>o.pts));
  if(!all.length)return{svg:head+`<text x="${W/2}" y="${H/2}" text-anchor="middle" fill="${T.muted}">Chưa có dữ liệu</text></svg>`,W,H,pw:0};
  let x0,x1;if(st.zoom){x0=st.zoom[0];x1=st.zoom[1]}else{[x0,x1]=ext(all.map(p=>p.x))}
  if(!(x1>x0)){x0-=1;x1+=1}else if(!st.zoom){const p=(x1-x0)*.02;x0-=p;x1+=p}
  const vis=all.filter(p=>p.x>=x0&&p.x<=x1),base=vis.length?vis:all,yv=[];
  base.forEach(p=>{yv.push(p.y);if(showRaw&&(!log||p.r>0))yv.push(p.r)});
  let[y0,y1]=ext(yv),ty;
  if(log){const a=Math.log10(y0),b=Math.log10(y1),p=(b-a)*.08||.2;ty=Math.log10;y0=Math.pow(10,a-p);y1=Math.pow(10,b+p)}
  else{if(spec.zero)y0=Math.min(0,y0);let p=(y1-y0)*.08;if(!(p>0))p=Math.abs(y0)*.05||.5;y1+=p;y0=spec.zero&&y0<=0?0:y0-p;ty=v=>v}
  const l0=ty(y0),l1=ty(y1),px=x=>ml+(x-x0)/(x1-x0)*pw,py=y=>mt+(1-(ty(y)-l0)/(l1-l0))*ph,inv=m=>x0+(m-ml)/pw*(x1-x0);
  const nx=Math.max(3,Math.floor(pw/95));
  const yt=log?logTicks(y0,y1):ticks(y0,y1,6);
  const xt=mode==='step'?stepTicks(x0,x1,nx):timeTicks(x0,x1,nx,mode==='wall');
  const xl=v=>mode==='step'?nf.format(Math.round(v)):mode==='rel'?relLbl(v):wallLbl(v,xt.step||60,x1-x0);
  const yl=spec.yaf||(log?(v=>{const a=Math.abs(v);return(a<1e-3||a>=1e5)?v.toExponential(0):String(+v.toPrecision(2))}):(v=>yLbl(v,yt.step||1)));
  let s=head;
  yt.forEach(v=>{const y=py(v);s+=`<line x1="${ml}" x2="${ml+pw}" y1="${y}" y2="${y}" stroke="${T.grid}"/><text x="${ml-8}" y="${y+4}" text-anchor="end" fill="${T.muted}">${yl(v)}</text>`});
  xt.forEach(v=>{const x=px(v);s+=`<line x1="${x}" x2="${x}" y1="${mt}" y2="${mt+ph}" stroke="${T.grid}" stroke-opacity=".6"/><text x="${x}" y="${mt+ph+16}" text-anchor="middle" fill="${T.muted}">${xl(v)}</text>`});
  s+=`<line x1="${ml}" x2="${ml+pw}" y1="${mt+ph}" y2="${mt+ph}" stroke="${T.axis}"/><line x1="${ml}" x2="${ml}" y1="${mt}" y2="${mt+ph}" stroke="${T.axis}"/>`;
  const xtitle=mode==='step'?(spec.stepName||'Epoch')+' (kết thúc)':mode==='rel'?'Thời gian chạy (relative)':'Thời gian thực (wall)';
  s+=`<text x="${ml+pw/2}" y="${H-6}" text-anchor="middle" fill="${T.muted}">${xtitle}</text>`;
  if(spec.yTitle)s+=`<text transform="rotate(-90)" x="${-(mt+ph/2)}" y="13" text-anchor="middle" fill="${T.muted}">${esc(spec.yTitle)}</text>`;
  if(spec.mark&&ser[0]){const idx=ser[0].s.step.indexOf(spec.mark);if(idx>=0){const mx=px(ser[0].s[mode][idx]);if(mx>=ml&&mx<=ml+pw)s+=`<line x1="${mx}" x2="${mx}" y1="${mt}" y2="${mt+ph}" stroke="#16a34a" stroke-dasharray="4 3"/><text x="${mx>ml+pw*.7?mx-4:mx+4}" y="${mt+11}" text-anchor="${mx>ml+pw*.7?'end':'start'}" fill="#16a34a" font-weight="700">best (epoch ${spec.mark})</text>`}}
  const path=(pts,f)=>pts.map((p,k)=>(k?'L':'M')+px(p.x).toFixed(1)+' '+py(f(p)).toFixed(1)).join('');
  s+=`<g clip-path="url(#cp${spec.id})">`;
  ser.forEach(o=>{const c=o.s.color,dash=o.s.dash?' stroke-dasharray="6 4"':'';
    if(showRaw)s+=`<path d="${path(o.pts,p=>p.r)}" fill="none" stroke="${c}" stroke-opacity=".28" stroke-width="1.2"/>`;
    s+=`<path d="${path(o.pts,p=>p.y)}" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round"${dash}/>`;
    if(o.pts.length<=80)o.pts.forEach(p=>{s+=`<circle cx="${px(p.x).toFixed(1)}" cy="${py(p.y).toFixed(1)}" r="2.4" fill="${c}"/>`})});
  s+='</g>';
  if(ex){s+=`<text x="${ml}" y="24" font-size="15" font-weight="700" fill="${T.text}">${esc(ex.title)}</text>`;
    lg.items.forEach(it=>{const sr=spec.series.find(q=>q.name===it.n);s+=`<rect x="${it.x}" y="${34+it.y}" width="14" height="4" rx="2" fill="${sr.color}"/><text x="${it.x+20}" y="${40+it.y}" fill="${T.text}">${esc(it.n)}</text>`})}
  else s+=`<g class="hov" pointer-events="none"></g><rect class="brush" y="${mt}" height="${ph}" fill="rgba(37,99,235,.15)" stroke="#2563eb" style="display:none" pointer-events="none"/><rect class="ov" x="${ml}" y="${mt}" width="${pw}" height="${ph}" fill="transparent" style="cursor:crosshair"/>`;
  return{svg:s+'</svg>',W,H,ml,mt,pw,ph,px,py,inv,ser,sm,T,spec};
}
function nearest(pts,x){let lo=0,hi=pts.length-1;while(hi-lo>1){const m=(lo+hi)>>1;if(pts[m].x<x)lo=m;else hi=m}return Math.abs(pts[lo].x-x)<=Math.abs(pts[hi].x-x)?pts[lo]:pts[hi]}
function lineTip(g,rows){
  const sp=g.spec,yf=sp.yf||(v=>fmt(v,5)),sm=g.sm>0;
  let h=`<table><thead><tr><th></th><th>Tên</th>${sm?'<th>Làm mượt</th>':''}<th>Giá trị</th><th>${sp.stepName||'Epoch'}</th><th>Kết thúc lúc</th><th>Đã chạy</th></tr></thead><tbody>`;
  rows.forEach(({o,p})=>{const s=o.s,i=p.i;h+=`<tr><td><i style="background:${s.color}"></i></td><td>${esc(s.name)}</td>${sm?`<td>${yf(p.y)}</td>`:''}<td>${yf(p.r)}</td><td>${sp.stepLabel?sp.stepLabel(s,i):s.step[i]}</td><td>${fmtWall(s.wall[i])}</td><td>${fmtDur(s.rel[i])}</td></tr>`});
  return h+'</tbody></table>'}
function LineChart(spec){
  const id='c'+(UID++);spec.id=id;
  const st={zoom:null,hidden:new Set(),smooth:spec.ownSmooth!=null?spec.ownSmooth:0,log:false};
  const html=`<section class="card" id="${id}"><div class="ch"><div><h3>${spec.title}</h3>${spec.sub?`<div class="sub">${spec.sub}</div>`:''}</div><div class="acts"><button class="b rz" hidden>Reset zoom</button>${spec.canLog?'<button class="b lg">log Y</button>':''}<button class="b ex" data-f="svg">SVG</button><button class="b ex" data-f="png">PNG</button></div></div>${spec.ownSmooth!=null?`<label class="mini">Làm mượt <input type="range" class="os" min="0" max="0.99" step="0.01" value="${st.smooth}"><b class="osv">${st.smooth}</b></label>`:''}<div class="legend"></div><div class="plot"></div></section>`;
  const self={html,
    resetZoom(){st.zoom=null},
    init(){const el=document.getElementById(id);
      el.querySelector('.rz').onclick=()=>{st.zoom=null;self.draw()};
      const lg=el.querySelector('.lg');if(lg)lg.onclick=()=>{st.log=!st.log;lg.classList.toggle('on',st.log);lg.style.background=st.log?'var(--hl)':'';self.draw()};
      el.querySelectorAll('.ex').forEach(b=>b.onclick=()=>exportChart(build(spec,st,1100,640,LIGHT,{title:(spec.runName?spec.runName+' - ':'')+spec.title.replace(/<[^>]+>/g,'')}),spec.file||spec.title,b.dataset.f));
      const os=el.querySelector('.os');if(os)os.oninput=()=>{st.smooth=+os.value;el.querySelector('.osv').textContent=os.value;self.draw()};
      el.querySelector('.legend').onclick=e=>{const c=e.target.closest('.chip');if(!c)return;const n=c.dataset.n;st.hidden.has(n)?st.hidden.delete(n):st.hidden.add(n);self.draw()}},
    draw(){const el=document.getElementById(id);if(!el)return;const box=el.querySelector('.plot'),W=Math.max(320,box.clientWidth),H=spec.h||280,T=th();
      const g=build(spec,st,W,H,T,null);box.innerHTML=g.svg;
      el.querySelector('.legend').innerHTML=spec.series.map(s=>`<span class="chip${st.hidden.has(s.name)?' off':''}" data-n="${esc(s.name)}"><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('');
      el.querySelector('.rz').hidden=!st.zoom;
      if(!g.pw)return;
      const svg=box.querySelector('svg'),ov=svg.querySelector('.ov'),hv=svg.querySelector('.hov'),br=svg.querySelector('.brush');
      const mxOf=e=>{const r=svg.getBoundingClientRect();return(e.clientX-r.left)*(g.W/r.width)};let drag=null;
      ov.addEventListener('mousemove',e=>{const mx=mxOf(e);
        if(drag){drag.x1=mx;const a=Math.min(drag.x0,mx),b=Math.max(drag.x0,mx);br.setAttribute('x',a);br.setAttribute('width',b-a);br.style.display='';hideTip();return}
        const xv=g.inv(mx),rows=[];let best=null,bd=1e18;
        g.ser.forEach(o=>{if(!o.pts.length)return;const p=nearest(o.pts,xv);rows.push({o,p});const d=Math.abs(g.px(p.x)-mx);if(d<bd){bd=d;best=p}});
        if(!best)return;const cx=g.px(best.x);
        let h=`<line x1="${cx}" x2="${cx}" y1="${g.mt}" y2="${g.mt+g.ph}" stroke="${g.T.axis}" stroke-dasharray="3 3"/>`;
        rows.forEach(({o,p})=>{h+=`<circle cx="${g.px(p.x)}" cy="${g.py(p.y)}" r="4.5" fill="${o.s.color}" stroke="#fff" stroke-width="1.5"/>`});
        hv.innerHTML=h;showTip(lineTip(g,rows),e)});
      ov.addEventListener('mouseleave',()=>{hv.innerHTML='';hideTip();drag=null;br.style.display='none'});
      ov.addEventListener('mousedown',e=>{const m=mxOf(e);drag={x0:m,x1:m};e.preventDefault()});
      ov.addEventListener('mouseup',()=>{if(!drag)return;const a=Math.min(drag.x0,drag.x1),b=Math.max(drag.x0,drag.x1);drag=null;br.style.display='none';if(b-a>8){st.zoom=[g.inv(a),g.inv(b)];self.draw()}});
      ov.addEventListener('dblclick',()=>{st.zoom=null;self.draw()})}};
  return self}

/* ---------- panel SVG chung (bar, scatter, ROC...) ---------- */
function Panel(o){
  const id='p'+(UID++);
  const html=`<section class="card ${o.cls||''}" id="${id}"><div class="ch"><div><h3>${o.title}</h3>${o.sub?`<div class="sub">${o.sub}</div>`:''}</div><div class="acts">${o.acts||''}<button class="b ex" data-f="svg">SVG</button><button class="b ex" data-f="png">PNG</button></div></div>${o.controls||''}<div class="plot"></div></section>`;
  const self={html,
    init(){const el=document.getElementById(id);el.querySelectorAll('.ex').forEach(b=>b.onclick=()=>exportChart(o.render(o.exW||960,LIGHT,true),o.file||o.title,b.dataset.f));o.init&&o.init(el,self)},
    draw(){const el=document.getElementById(id);if(!el)return;const box=el.querySelector('.plot'),r=o.render(Math.max(320,box.clientWidth),th(),false);box.innerHTML=r.svg;o.bind&&o.bind(box,r);o.after&&o.after(el,r)}};
  return self}
function svgHead(W,H,T){return`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="'Segoe UI',Roboto,Helvetica,Arial,sans-serif" font-size="11"><rect width="${W}" height="${H}" fill="${T.bg}"/>`}

function barRender(items,metric,zero,W,T,ex){
  const ml=Math.min(190,Math.max(110,Math.max(...items.map(i=>i.name.length))*6.6+14)),mr=70,mt=ex?40:6,rh=30,H=mt+items.length*rh+32;
  const vals=items.map(i=>i.v),hi=Math.max(...vals),lo=Math.min(...vals);
  let a=zero?0:Math.max(0,lo-(hi-lo)*.6-1e-9),b=hi+(hi-a)*.06;if(!(b>a))b=a+1;
  const pw=W-ml-mr,px=v=>ml+(v-a)/(b-a)*pw;let s=svgHead(W,H,T);
  if(ex)s+=`<text x="${ml}" y="24" font-size="15" font-weight="700" fill="${T.text}">${esc(metric.n)} - tốt nhất mỗi mô hình (validation)</text>`;
  ticks(a,b,5).forEach(v=>{const x=px(v);s+=`<line x1="${x}" x2="${x}" y1="${mt}" y2="${H-26}" stroke="${T.grid}"/><text x="${x}" y="${H-10}" text-anchor="middle" fill="${T.muted}">${metric.f===undefined?v:(metric.k==='val_loss'||metric.k==='auc'?fmt(v,3):pct(v,1))}</text>`});
  items.forEach((it,i)=>{const y=mt+i*rh+4,w=Math.max(2,px(it.v)-ml);
    s+=`<text x="${ml-8}" y="${y+rh/2-1}" text-anchor="end" fill="${T.text}">${esc(it.name)}</text><rect x="${ml}" y="${y}" width="${w}" height="${rh-10}" rx="4" fill="${it.color}"/><text x="${ml+w+6}" y="${y+rh/2-1}" fill="${T.text}" font-weight="700">${metric.f(it.v)}</text>`});
  return{svg:s+'</svg>',W,H}}

function scatterRender(runs,W,T,ex){
  const ml=64,mr=20,mt=ex?44:10,mb=44,H=Math.round(Math.min(480,W*.6)),pw=W-ml-mr,ph=H-mt-mb;
  const xs=runs.map(r=>r.bm.val_loss),ys=runs.map(r=>r.bm.val_acc);let[x0,x1]=ext(xs),[y0,y1]=ext(ys);
  let px_=(x1-x0)*.2||.01,py_=(y1-y0)*.25||.01;x0-=px_;x1+=px_;y0-=py_;y1+=py_;
  const px=x=>ml+(x-x0)/(x1-x0)*pw,py=y=>mt+(1-(y-y0)/(y1-y0))*ph;let s=svgHead(W,H,T);
  if(ex)s+=`<text x="${ml}" y="24" font-size="15" font-weight="700" fill="${T.text}">Accuracy so với Validation loss</text>`;
  const yt=ticks(y0,y1,6),xt=ticks(x0,x1,6);
  yt.forEach(v=>{s+=`<line x1="${ml}" x2="${ml+pw}" y1="${py(v)}" y2="${py(v)}" stroke="${T.grid}"/><text x="${ml-8}" y="${py(v)+4}" text-anchor="end" fill="${T.muted}">${pct(v,1)}</text>`});
  xt.forEach(v=>{s+=`<line x1="${px(v)}" x2="${px(v)}" y1="${mt}" y2="${mt+ph}" stroke="${T.grid}"/><text x="${px(v)}" y="${mt+ph+16}" text-anchor="middle" fill="${T.muted}">${fmt(v,3)}</text>`});
  s+=`<text x="${ml+pw/2}" y="${H-6}" text-anchor="middle" fill="${T.muted}">Validation loss (thấp hơn là tốt hơn)</text><text transform="rotate(-90)" x="${-(mt+ph/2)}" y="14" text-anchor="middle" fill="${T.muted}">Accuracy (cao hơn là tốt hơn)</text>`;
  runs.forEach(r=>{const x=px(r.bm.val_loss),y=py(r.bm.val_acc);s+=`<circle cx="${x}" cy="${y}" r="7" fill="${r.color}" stroke="${T.bg}" stroke-width="2"><title>${esc(r.name)}</title></circle><text x="${x+10}" y="${y-8}" fill="${T.text}">${esc(r.name)}</text>`});
  return{svg:s+'</svg>',W,H}}

function rocRender(r,which,W,T,ex){
  const pk=which==='best'?r.bestPack:r.lastPack;
  if(!pk)return{svg:svgHead(W,200,T)+`<text x="${W/2}" y="100" text-anchor="middle" fill="${T.muted}">Không có dữ liệu ROC</text></svg>`,W,H:200};
  const ml=56,mr=ex?170:16,mt=ex?44:12,mb=44,pw=W-ml-mr,H=Math.round(Math.min(560,pw*.78)+mt+mb),ph=H-mt-mb,g=r.roc_grid,hid=S.rocOff[r.name]||new Set();
  const zm=S.rocZoom,xr=zm?[0,.1]:[0,1],yr=zm?[.8,1]:[0,1],pxm=x=>ml+(x-xr[0])/(xr[1]-xr[0])*pw,pym=y=>mt+(1-(y-yr[0])/(yr[1]-yr[0]))*ph;
  const xs=zm?[0,.02,.04,.06,.08,.1]:[0,.2,.4,.6,.8,1],ys=zm?[.8,.84,.88,.92,.96,1]:xs,dg=zm?2:1;
  let s=svgHead(W,H,T)+`<defs><clipPath id="rc"><rect x="${ml}" y="${mt}" width="${pw}" height="${ph}"/></clipPath></defs>`;
  if(ex)s+=`<text x="${ml}" y="24" font-size="15" font-weight="700" fill="${T.text}">${esc(r.name)} - ROC một-với-tất-cả (epoch ${pk.epoch})${zm?' - phóng góc trên-trái':''}</text>`;
  ys.forEach(v=>{s+=`<line x1="${ml}" x2="${ml+pw}" y1="${pym(v)}" y2="${pym(v)}" stroke="${T.grid}"/><text x="${ml-8}" y="${pym(v)+4}" text-anchor="end" fill="${T.muted}">${v.toFixed(dg)}</text>`});
  xs.forEach(v=>{s+=`<line x1="${pxm(v)}" x2="${pxm(v)}" y1="${mt}" y2="${mt+ph}" stroke="${T.grid}"/><text x="${pxm(v)}" y="${mt+ph+16}" text-anchor="middle" fill="${T.muted}">${v.toFixed(dg)}</text>`});
  s+=`<g clip-path="url(#rc)"><line x1="${pxm(0)}" y1="${pym(0)}" x2="${pxm(1)}" y2="${pym(1)}" stroke="${T.axis}" stroke-dasharray="5 4"/></g><text x="${ml+pw/2}" y="${H-6}" text-anchor="middle" fill="${T.muted}">False Positive Rate</text><text transform="rotate(-90)" x="${-(mt+ph/2)}" y="14" text-anchor="middle" fill="${T.muted}">True Positive Rate</text>`;
  const line=(arr,c,w,op)=>`<path clip-path="url(#rc)" d="${arr.map((v,k)=>(k?'L':'M')+pxm(g[k]).toFixed(1)+' '+pym(v).toFixed(1)).join('')}" fill="none" stroke="${c}" stroke-width="${w}" stroke-opacity="${op}" stroke-linejoin="round"/>`;
  pk.roc.tpr.forEach((t,i)=>{if(!hid.has(i))s+=line(t,C10[i%10],1.6,.85)});
  if(!hid.has('m'))s+=line(pk.roc.macro,T.dark?'#fff':'#000',2.8,1);
  if(ex){r.cls.forEach((n,i)=>{s+=`<rect x="${ml+pw+14}" y="${mt+i*18}" width="12" height="4" fill="${C10[i%10]}"/><text x="${ml+pw+32}" y="${mt+i*18+6}" fill="${T.text}">${esc(n)} ${fmt(pk.roc.auc[i],3)}</text>`});s+=`<rect x="${ml+pw+14}" y="${mt+r.cls.length*18}" width="12" height="4" fill="#000"/><text x="${ml+pw+32}" y="${mt+r.cls.length*18+6}" fill="${T.text}">macro ${fmt(pk.roc.auc.reduce((a,b)=>a+b,0)/pk.roc.auc.length,3)}</text>`}
  return{svg:s+'</svg>',W,H,pk}}

/* ---------- ma trận nhầm lẫn ---------- */
function cmRender(cm,names,mode,W,T,ex,title){
  const n=cm.length,ml=ex?124:108,mt=ex?50:10,mr=14,mb=ex?104:92,cell=Math.max(26,Math.floor((W-ml-mr)/n)),size=cell*n,Wt=ml+size+mr,H=mt+size+mb;
  const rs=cm.map(r=>r.reduce((a,b)=>a+b,0)),cs=cm[0].map((_,j)=>cm.reduce((a,r)=>a+r[j],0));
  let mx=0,mo=0;cm.forEach((r,i)=>r.forEach((v,j)=>{mx=Math.max(mx,v);if(i!==j)mo=Math.max(mo,v)}));
  const base=mode==='errors'?[220,38,38]:[37,99,235],bg=T.dark?[23,27,36]:[255,255,255];
  let s=svgHead(Wt,H,T);
  if(ex)s+=`<text x="${ml}" y="24" font-size="15" font-weight="700" fill="${T.text}">${esc(title)}</text>`;
  cm.forEach((row,i)=>row.forEach((v,j)=>{
    let t,txt;
    if(mode==='count'){t=mx?v/mx:0;txt=v}
    else if(mode==='recall'){t=rs[i]?v/rs[i]:0;txt=v?(t*100).toFixed(1)+'%':''}
    else if(mode==='precision'){t=cs[j]?v/cs[j]:0;txt=v?(t*100).toFixed(1)+'%':''}
    else{if(i===j){t=0;txt='-'}else{t=mo?v/mo:0;txt=v||''}}
    const fill=(mode==='errors'&&i===j)?T.grid:mixc(bg,base,Math.min(1,t)),tc=t>.55&&!(mode==='errors'&&i===j)?'#fff':T.text;
    s+=`<rect x="${ml+j*cell}" y="${mt+i*cell}" width="${cell}" height="${cell}" fill="${fill}" stroke="${T.bg}" stroke-width="1" data-i="${i}" data-j="${j}"/><text x="${ml+j*cell+cell/2}" y="${mt+i*cell+cell/2+4}" text-anchor="middle" fill="${tc}" font-size="${cell<40?9.5:11}" pointer-events="none">${txt}</text>`}));
  names.forEach((nm,i)=>{s+=`<text x="${ml-8}" y="${mt+i*cell+cell/2+4}" text-anchor="end" fill="${T.text}">${esc(nm)}</text><text transform="translate(${ml+i*cell+cell/2+4},${mt+size+12}) rotate(-40)" text-anchor="end" fill="${T.text}">${esc(nm)}</text>`});
  s+=`<text x="${ml+size/2}" y="${H-6}" text-anchor="middle" fill="${T.muted}" font-weight="600">Dự đoán (predicted)</text><text transform="rotate(-90)" x="${-(mt+size/2)}" y="14" text-anchor="middle" fill="${T.muted}" font-weight="600">Nhãn thật (true)</text>`;
  return{svg:s+'</svg>',W:Wt,H,cm,rs,cs,names}}

function CMSection(r){
  const id='cm'+(UID++),names=r.cls;
  if(S.cmEp[r.name]==null)S.cmEp[r.name]=r.bestEp;
  const mk=r.E.length;
  const html=`<section class="card" id="${id}"><div class="ch"><div><h3>Ma trận nhầm lẫn &amp; lỗi theo lớp</h3><div class="sub">Tập validation - chọn epoch bằng thanh trượt (mặc định = epoch tốt nhất)</div></div><div class="acts"><button class="b ex" data-f="svg">SVG</button><button class="b ex" data-f="png">PNG</button></div></div>
  <div class="ctrls" style="margin:4px 0 10px"><label>Epoch <input type="range" class="ep" min="1" max="${mk}" step="1"><b class="epv"></b></label>
  <span class="seg"><button data-m="count">Số lượng</button><button data-m="recall">% theo hàng (recall)</button><button data-m="precision">% theo cột (precision)</button><button data-m="errors">Chỉ các lỗi</button></span></div>
  <div class="cmwrap"><div class="left plot"></div><div class="right"><div class="side"></div></div></div></section>`;
  const cur=()=>r.byEp[S.cmEp[r.name]]||r.E[r.E.length-1];
  const self={html,
    init(){const el=document.getElementById(id);
      el.querySelector('.ep').oninput=e=>{S.cmEp[r.name]=+e.target.value;self.draw()};
      el.querySelectorAll('.seg button').forEach(b=>b.onclick=()=>{S.cmMode=b.dataset.m;self.draw()});
      el.querySelectorAll('.ex').forEach(b=>b.onclick=()=>{const e=cur();exportChart(cmRender(e.cm,names,S.cmMode,760,LIGHT,true,`${r.name} - Confusion matrix (epoch ${e.epoch})`),`${r.name}_confusion_epoch${e.epoch}`,b.dataset.f)})},
    draw(){const el=document.getElementById(id);if(!el)return;const e=cur(),box=el.querySelector('.plot'),T=th();
      el.querySelector('.ep').value=e.epoch;el.querySelector('.epv').innerHTML=`${e.epoch}${e.epoch===r.bestEp?' (best)':''}`;
      el.querySelectorAll('.seg button').forEach(b=>b.classList.toggle('on',b.dataset.m===S.cmMode));
      const g=cmRender(e.cm,names,S.cmMode,Math.min(box.clientWidth||640,720),T,false);box.innerHTML=g.svg;
      box.querySelector('svg').addEventListener('mousemove',ev=>{const t=ev.target.closest('rect[data-i]');if(!t){hideTip();return}
        const i=+t.dataset.i,j=+t.dataset.j,v=g.cm[i][j];
        showTip(`<b>Thật:</b> ${esc(names[i])}<br><b>Dự đoán:</b> ${esc(names[j])}<br><b>${nf.format(v)}</b> ảnh &middot; ${pct(g.rs[i]?v/g.rs[i]:0,1)} của lớp thật &middot; ${pct(g.cs[j]?v/g.cs[j]:0,1)} của lớp dự đoán`,ev)});
      box.querySelector('svg').addEventListener('mouseleave',hideTip);
      const pairs=[];e.cm.forEach((row,i)=>row.forEach((v,j)=>{if(i!==j&&v>0)pairs.push({i,j,v})}));pairs.sort((a,b)=>b.v-a.v);
      const top=pairs.slice(0,10),mxp=top.length?top[0].v:1,rs=g.rs;
      let h=`<h3 style="margin:0 0 6px;font-size:13px">Top 10 cặp bị nhầm nhiều nhất (epoch ${e.epoch})</h3>`;
      top.forEach(p=>{h+=`<div class="pair"><span>${esc(names[p.i])} &rarr; ${esc(names[p.j])}</span><div class="bar"><i style="width:${p.v/mxp*100}%"></i></div><b>${p.v}</b><small>${pct(rs[p.i]?p.v/rs[p.i]:0,1)}</small></div>`});
      h+=`<div class="hint">Cột phải = % số ảnh thật của lớp nguồn bị dự đoán sai thành lớp đích.</div><h3 style="margin:12px 0 6px;font-size:13px">Chỉ số theo lớp</h3><div class="scroll" style="max-height:360px"><table class="tbl"><thead><tr><th>Lớp</th><th>Số ảnh</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th><th class="l">Hay bị nhầm thành</th></tr></thead><tbody>`;
      const pc=e.pc;names.forEach((nm,i)=>{let bj=-1,bv=0;e.cm[i].forEach((v,j)=>{if(j!==i&&v>bv){bv=v;bj=j}});
        h+=`<tr><td>${esc(nm)}</td><td>${nf.format(pc.support[i])}</td><td>${pct(pc.precision[i],1)}</td><td>${pct(pc.recall[i],1)}</td><td>${pct(pc.f1[i],1)}</td><td>${fmt(pc.auc[i],4)}</td><td class="l">${bj>=0?esc(names[bj])+' ('+bv+')':'-'}</td></tr>`});
      el.querySelector('.side').innerHTML=h+'</tbody></table></div>'}};
  return self}

/* ---------- view: một mô hình ---------- */
function serOf(r,key,name,color,x){return Object.assign({name,color,y:r.E.map(typeof key==='function'?key:e=>e[key]),step:r.E.map(e=>e.epoch),rel:r.E.map(e=>e.rel),wall:r.E.map(e=>e.wall)},x||{})}
function KPI(l,v,s){return`<div class="kpi"><div class="kl">${l}</div><div class="kv">${v}</div>${s?`<div class="ks">${s}</div>`:''}</div>`}
function csvOf(rows){return'\ufeff'+rows.map(r=>r.map(v=>{v=v==null?'':String(v);return/[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v}).join(',')).join('\n')}
function viewRun(r){
  const add=c=>{CHARTS.push(c);return c.html},T=r.name,bm=r.bm,C=(o)=>{o.runName=r.name;o.file=r.name+'_'+slug(o.title);return LineChart(o)};
  const stBadge=r.status==='completed'?'<span class="badge">Hoàn tất</span>':r.status==='early_stopped'?'<span class="badge w">Dừng sớm</span>':'<span class="badge g">Đang chạy / chưa xong</span>';
  let h=`<h2 style="margin-top:6px">${esc(r.name)} ${stBadge}</h2><div class="hint">Thiết bị: ${esc(r.device||'-')} &middot; bắt đầu ${fmtWall(r.started)}${r.finished?' &middot; kết thúc '+fmtWall(r.finished):''}</div>`;
  h+=`<div class="kpis">${KPI('Epoch tốt nhất','#'+r.bestEp,`đã chạy ${r.E.length} epoch`)}${KPI('Accuracy (val)',pct(bm.val_acc),'train '+pct(bm.train_acc))}${KPI('F1 macro',pct(bm.f1))}${KPI('AUC macro',fmt(bm.auc,4))}${KPI('Val loss',fmt(bm.val_loss,4),'train '+fmt(bm.train_loss,4))}${KPI('FPR macro',pct(bm.fpr,3))}${KPI('Tổng thời gian',fmtDur(r.totalTime),'TB '+fmtDur(r.totalTime/r.E.length)+'/epoch')}${KPI('Tham số',r.n_params!=null?nf.format(r.n_params):'-',nf.format(Math.round(r.ips))+' ảnh/giây')}</div>`;
  const grid=a=>`<div class="grid">${a.join('')}</div>`,pf=v=>pct(v,3),yp=v=>pct(v,1);
  h+=`<h2>Loss &amp; độ chính xác</h2>`+grid([
    add(C({title:'Loss: Train vs Validation',sub:'Cross-entropy; đường nét đứt xanh lá đánh dấu epoch tốt nhất',canLog:true,mark:r.bestEp,yTitle:'loss',series:[serOf(r,'train_loss','Train','#2563eb'),serOf(r,'val_loss','Validation','#f97316')]})),
    add(C({title:'Accuracy: Train vs Validation',sub:'Train = trung bình trong lúc train (dropout bật)',mark:r.bestEp,yf:pf,yaf:yp,series:[serOf(r,'train_acc','Train','#2563eb'),serOf(r,'val_acc','Validation','#f97316')]})),
    add(C({title:'Loss train theo batch',sub:'Mỗi điểm = một lần ghi trong epoch (có thanh làm mượt riêng)',stepName:'Iteration',canLog:true,ownSmooth:.9,stepLabel:(s,i)=>`${nf.format(s.step[i])} (epoch ${s.ep[i]})`,series:[{name:'Train batch loss',color:'#2563eb',step:r.batches.map(b=>b[1]),ep:r.batches.map(b=>b[0]),y:r.batches.map(b=>b[2]),wall:r.batches.map(b=>b[3]),rel:r.batches.map(b=>b[4])}]})),
    add(C({title:'Khoảng cách tổng quát hoá (overfitting)',sub:'Val loss - Train loss (dương = có dấu hiệu overfit)',mark:r.bestEp,series:[serOf(r,e=>e.val_loss-e.train_loss,'Val loss - Train loss','#dc2626'),serOf(r,e=>e.train_acc-e.val_acc,'Train acc - Val acc','#9333ea',{dash:true})]}))]);
  h+=`<h2>Chỉ số phân loại (validation, macro)</h2>`+grid([
    add(C({title:'Precision / Recall / F1',mark:r.bestEp,yf:pf,yaf:yp,series:[serOf(r,'precision','Precision','#2563eb'),serOf(r,'recall','Recall','#16a34a'),serOf(r,'f1','F1','#f97316')]})),
    add(C({title:'AUC (ROC, one-vs-rest)',mark:r.bestEp,yf:v=>fmt(v,5),series:[serOf(r,'auc','AUC macro','#9333ea')]})),
    add(C({title:'FPR (False Positive Rate)',sub:'Thấp hơn là tốt hơn',mark:r.bestEp,yf:pf,yaf:v=>pct(v,2),canLog:true,series:[serOf(r,'fpr','FPR macro','#dc2626')]})),
    add(C({title:'F1 theo từng lớp',mark:r.bestEp,yf:pf,yaf:yp,series:r.cls.map((n,i)=>serOf(r,e=>e.pc.f1[i],n,C10[i%10]))})),
    add(C({title:'Recall theo từng lớp',mark:r.bestEp,yf:pf,yaf:yp,series:r.cls.map((n,i)=>serOf(r,e=>e.pc.recall[i],n,C10[i%10]))})),
    add(C({title:'Precision theo từng lớp',mark:r.bestEp,yf:pf,yaf:yp,series:r.cls.map((n,i)=>serOf(r,e=>e.pc.precision[i],n,C10[i%10]))}))]);
  h+=`<h2>Trình tối ưu hoá</h2>`+grid([
    add(C({title:'Learning rate',sub:'Giá trị dùng trong epoch đó',yf:v=>v.toExponential(3),yaf:v=>v===0?'0':v.toExponential(1),series:[serOf(r,'lr','LR','#2563eb')]})),
    add(C({title:'Gradient norm (trước clip)',sub:'Trung bình và lớn nhất trong epoch',canLog:true,yf:v=>fmt(v,4),series:[serOf(r,'grad_norm','Trung bình','#16a34a'),serOf(r,'grad_norm_max','Lớn nhất','#dc2626',{dash:true})]})),
    add(C({title:'Tỷ lệ batch bị clip gradient',sub:'Phần trăm batch có grad norm vượt ngưỡng',yf:v=>pct(v,2),yaf:yp,zero:true,series:[serOf(r,'clip_frac','Clip fraction','#9333ea')]})),
    add(C({title:'Weight norm (L2 toàn mô hình)',yf:v=>fmt(v,3),series:[serOf(r,'weight_norm','||w||','#0891b2')]}))]);
  h+=`<h2>Thời gian</h2>`+grid([
    add(C({title:'Thời gian mỗi epoch',sub:'Giây',zero:true,yf:v=>fmt(v,2)+'s',yaf:v=>v.toFixed(0)+'s',series:[serOf(r,'train_time','Train','#2563eb'),serOf(r,'val_time','Validate','#f97316'),serOf(r,'epoch_time','Tổng','#475569',{dash:true})]})),
    add(C({title:'Tốc độ huấn luyện',sub:'Ảnh / giây',zero:true,yf:v=>nf.format(Math.round(v)),yaf:v=>nf.format(Math.round(v)),series:[serOf(r,'img_per_sec','img/s','#16a34a')]}))]);
  h+=`<h2>Nhầm lẫn giữa các lớp</h2>`;const cm=CMSection(r);CHARTS.push(cm);h+=cm.html;
  if(r.bestPack||r.lastPack){
    h+=`<h2>Đường cong ROC</h2>`;
    const roc=Panel({title:'ROC từng lớp (one-vs-rest)',sub:'Đường đen đậm = trung bình macro; bấm tên lớp để ẩn/hiện',file:r.name+'_roc',exW:980,
      acts:`<span class="seg sw"><button data-w="best">Epoch tốt nhất</button><button data-w="last">Epoch cuối</button></span><span class="seg sz"><button data-z="0">Toàn bộ</button><button data-z="1">Phóng góc trên-trái</button></span>`,
      controls:'<div class="legend roc-l"></div>',
      render:(W,Tm,ex)=>rocRender(r,S.rocWhich,Math.min(W,820),Tm,ex),
      init:(el,self)=>{el.querySelectorAll('.sw button').forEach(b=>b.onclick=()=>{S.rocWhich=b.dataset.w;self.draw()});el.querySelectorAll('.sz button').forEach(b=>b.onclick=()=>{S.rocZoom=b.dataset.z==='1';self.draw()});
        el.querySelector('.roc-l').onclick=ev=>{const c=ev.target.closest('.chip');if(!c)return;const k=c.dataset.k==='m'?'m':+c.dataset.k,hs=S.rocOff[r.name]||(S.rocOff[r.name]=new Set());hs.has(k)?hs.delete(k):hs.add(k);self.draw()}},
      after:(el,res)=>{el.querySelectorAll('.sw button').forEach(b=>b.classList.toggle('on',b.dataset.w===S.rocWhich));el.querySelectorAll('.sz button').forEach(b=>b.classList.toggle('on',(b.dataset.z==='1')===S.rocZoom));const pk=res.pk;if(!pk)return;const hs=S.rocOff[r.name]||new Set();
        el.querySelector('.roc-l').innerHTML=r.cls.map((n,i)=>`<span class="chip${hs.has(i)?' off':''}" data-k="${i}"><i style="background:${C10[i%10]}"></i>${esc(n)} ${fmt(pk.roc.auc[i],3)}</span>`).join('')+`<span class="chip${hs.has('m')?' off':''}" data-k="m"><i style="background:${th().dark?'#fff':'#000'}"></i><b>macro ${fmt(pk.roc.auc.reduce((a,b)=>a+b,0)/pk.roc.auc.length,4)}</b></span><span class="sub">&nbsp;(epoch ${pk.epoch})</span>`}});
    CHARTS.push(roc);h+=roc.html}
  const keys=[['epoch','Epoch'],['wall','Kết thúc lúc'],['rel','Đã chạy (s)'],['train_loss','Train loss'],['val_loss','Val loss'],['train_acc','Train acc'],['val_acc','Val acc'],['precision','Precision'],['recall','Recall'],['f1','F1'],['auc','AUC'],['fpr','FPR'],['lr','LR'],['grad_norm','Grad norm'],['weight_norm','Weight norm'],['epoch_time','Thời gian epoch (s)']];
  h+=`<h2>Số liệu từng epoch &amp; cấu hình</h2><div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(420px,1fr))"><section class="card"><div class="ch"><h3>Bảng số liệu từng epoch</h3><div class="acts"><button class="b" id="csvEp">Tải CSV</button></div></div><div class="scroll"><table class="tbl"><thead><tr>${keys.map(k=>`<th>${k[1]}</th>`).join('')}</tr></thead><tbody>${r.E.map(e=>`<tr${e.epoch===r.bestEp?' style="background:var(--hl)"':''}>${keys.map(([k])=>`<td>${k==='wall'?fmtWall(e[k]):k==='epoch'?e[k]+(e.epoch===r.bestEp?' ★':''):k==='rel'||k==='epoch_time'?fmt(e[k],1):k==='lr'?(e[k]==null?'-':e[k].toExponential(2)):fmt(e[k],4)}</td>`).join('')}</tr>`).join('')}</tbody></table></div></section>
  <section class="card"><h3 style="margin:0 0 8px">Cấu hình</h3><table class="tbl cfg"><tbody>${Object.entries(r.config||{}).map(([k,v])=>`<tr><td class="l">${esc(k)}</td><td class="l">${esc(JSON.stringify(v))}</td></tr>`).join('')}</tbody></table></section></div>`;
  return{html:h,after(){const b=document.getElementById('csvEp');if(b)b.onclick=()=>dl(slug(r.name)+'_epochs.csv',new Blob([csvOf([keys.map(k=>k[0])].concat(r.E.map(e=>keys.map(([k])=>e[k]))))],{type:'text/csv'}))}}}

/* ---------- view: so sánh ---------- */
function viewCmp(){
  const add=c=>{CHARTS.push(c);return c.html},m=MET.find(x=>x.k===S.rank),selRuns=RUNS.filter(r=>S.sel.has(r.name));
  const cmpv=(a,b)=>{const d=m.up?b.bm[m.k]-a.bm[m.k]:a.bm[m.k]-b.bm[m.k];return d||a.bm.val_loss-b.bm.val_loss};
  const sorted=RUNS.slice().sort((a,b)=>(S.sel.has(b.name)-S.sel.has(a.name))||cmpv(a,b));
  const rng={};MET.forEach(q=>{const v=selRuns.map(r=>r.bm[q.k]);rng[q.k]=v.length?ext(v):[0,0]});
  const extra=[['bestEp','Epoch tốt nhất',r=>'#'+r.bestEp],['n','Epoch đã chạy',r=>r.E.length],['np','Tham số',r=>r.n_params!=null?nf.format(r.n_params):'-'],['tt','Tổng thời gian',r=>fmtDur(r.totalTime)],['ips','Ảnh/giây',r=>nf.format(Math.round(r.ips))]];
  let rank=0;
  let rows=sorted.map(r=>{const sel=S.sel.has(r.name),rk=sel?++rank:'';
    const cells=MET.map(q=>{const v=r.bm[q.k],[lo,hi]=rng[q.k],t=hi>lo?(q.up?(v-lo)/(hi-lo):(hi-v)/(hi-lo)):.5,best=sel&&v===(q.up?hi:lo)&&hi>lo;
      return`<td class="${best?'best':''}" style="background:rgba(37,99,235,${sel?(.04+.2*t).toFixed(3):0})">${q.f(v)}</td>`}).join('');
    return`<tr class="${sel?'':'dim'}"><td><input type="checkbox" data-n="${esc(r.name)}" ${sel?'checked':''}></td><td>${rk}${rk===1?' &#11088;':''}</td><td class="l"><span class="dot" style="background:${r.color}"></span> <a href="#" data-go="${r.idx}">${esc(r.name)}</a>${r.status==='early_stopped'?'<span class="badge w">dừng sớm</span>':''}</td>${cells}${extra.map(x=>`<td>${x[2](r)}</td>`).join('')}</tr>`}).join('');
  const top=selRuns.slice().sort(cmpv)[0];
  let h=`<h2 style="margin-top:6px">So sánh các mô hình</h2><div class="hint">Số liệu lấy tại <b>epoch tốt nhất</b> của từng mô hình (theo accuracy trên validation). Ô viền xanh = tốt nhất cột; tích chọn ở cột đầu để đưa vào/loại ra khỏi các biểu đồ bên dưới.</div>`;
  h+=`<section class="card"><div class="ch"><div><h3>Bảng xếp hạng để chọn mô hình đưa vào báo cáo</h3><div class="sub">${top?`Theo ${m.n}: <b>${esc(top.name)}</b> dẫn đầu (${m.f(top.bm[m.k])})`:'Chưa chọn mô hình nào'}</div></div><div class="acts"><label class="mini">Xếp hạng theo <select id="rank">${MET.map(q=>`<option value="${q.k}"${q.k===S.rank?' selected':''}>${q.n}</option>`).join('')}</select></label><button class="b" id="csvCmp">Tải CSV</button></div></div>
  <div class="scroll"><table class="tbl"><thead><tr><th></th><th>Hạng</th><th class="l">Mô hình</th>${MET.map(q=>`<th>${q.n}${q.up?' &uarr;':' &darr;'}</th>`).join('')}${extra.map(x=>`<th>${x[1]}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div></section>`;
  const items=selRuns.map(r=>({name:r.name,color:r.color,v:r.bm[m.k]})).sort((a,b)=>m.up?b.v-a.v:a.v-b.v);
  const bar=Panel({title:`${m.n} tốt nhất của từng mô hình`,sub:'Giá trị tại epoch tốt nhất (validation)',file:'so_sanh_'+m.k,exW:900,
    acts:`<label class="mini"><input type="checkbox" id="bz" ${S.barZero?'checked':''}> Trục bắt đầu từ 0</label>`,
    render:(W,Tm,ex)=>items.length?barRender(items,m,S.barZero,W,Tm,ex):{svg:svgHead(W,80,Tm)+`<text x="${W/2}" y="40" text-anchor="middle" fill="${Tm.muted}">Chọn ít nhất một mô hình</text></svg>`,W,H:80},
    init:el=>{el.querySelector('#bz').onchange=e=>{S.barZero=e.target.checked;redrawAll()}}});
  const sc=Panel({title:'Accuracy so với Validation loss',sub:'Góc trên-bên-trái là tốt nhất',file:'so_sanh_acc_vs_loss',exW:900,render:(W,Tm,ex)=>selRuns.length?scatterRender(selRuns,W,Tm,ex):{svg:svgHead(W,80,Tm)+'</svg>',W,H:80}});
  CHARTS.push(bar,sc);
  const O=(title,key,o)=>{const c=LineChart(Object.assign({title,file:'so_sanh_'+slug(title),series:selRuns.map(r=>serOf(r,key,r.name,r.color))},o||{}));return add(c)};
  h+=`<div class="grid" style="margin-top:14px">${bar.html}${sc.html}</div><h2>Đường cong học (chồng các mô hình)</h2><div class="grid">`+
    O('Validation loss','val_loss',{canLog:true})+O('Validation accuracy','val_acc',{yf:v=>pct(v,3),yaf:v=>pct(v,1)})+
    O('Validation F1 (macro)','f1',{yf:v=>pct(v,3),yaf:v=>pct(v,1)})+O('Validation AUC (macro)','auc',{yf:v=>fmt(v,5)})+
    O('Train loss','train_loss',{canLog:true})+O('Khoảng cách overfit (Val loss - Train loss)',e=>e.val_loss-e.train_loss)+
    O('Learning rate','lr',{yf:v=>v.toExponential(3),yaf:v=>v===0?'0':v.toExponential(1)})+O('Thời gian mỗi epoch (giây)','epoch_time',{zero:true,yf:v=>fmt(v,2)+'s',yaf:v=>v.toFixed(0)+'s'})+'</div>';
  return{html:h,after(){
    document.querySelectorAll('#view input[data-n]').forEach(c=>c.onchange=()=>{c.checked?S.sel.add(c.dataset.n):S.sel.delete(c.dataset.n);renderView()});
    $('#rank').onchange=e=>{S.rank=e.target.value;renderView()};
    document.querySelectorAll('#view a[data-go]').forEach(a=>a.onclick=e=>{e.preventDefault();S.view=+a.dataset.go;renderTabs();renderView();scrollTo(0,0)});
    $('#csvCmp').onclick=()=>dl('so_sanh_mo_hinh.csv',new Blob([csvOf([['name'].concat(MET.map(q=>q.k),['best_epoch','epochs_run','n_params','total_time_s','img_per_sec'])].concat(sorted.map(r=>[r.name].concat(MET.map(q=>r.bm[q.k]),[r.bestEp,r.E.length,r.n_params,r.totalTime,r.ips]))))],{type:'text/csv'}))}}}

/* ---------- điều khiển chung ---------- */
function redrawAll(){CHARTS.forEach(c=>c.draw())}
function renderView(){CHARTS=[];const v=S.view==='cmp'?viewCmp():viewRun(RUNS[S.view]);$('#view').innerHTML=v.html;CHARTS.forEach(c=>c.init&&c.init());v.after&&v.after();redrawAll()}
function renderTabs(){const t=$('#tabs');t.innerHTML=(RUNS.length>1?`<button data-v="cmp" class="${S.view==='cmp'?'on':''}">So sánh tất cả</button>`:'')+RUNS.map(r=>`<button data-v="${r.idx}" class="${S.view===r.idx?'on':''}"><i style="background:${r.color}"></i>${esc(r.name)}</button>`).join('');
  t.querySelectorAll('button').forEach(b=>b.onclick=()=>{S.view=b.dataset.v==='cmp'?'cmp':+b.dataset.v;renderTabs();renderView();scrollTo(0,0)})}
$('#ttl').textContent=D.title;$('#meta').textContent=`${RUNS.length} mô hình · tạo lúc ${D.generated} · di chuột lên biểu đồ để xem chi tiết, kéo chuột để phóng to, nhấp đúp để reset`;
$('#smooth').oninput=e=>{S.smooth=+e.target.value;$('#smv').textContent=e.target.value;redrawAll()};
$('#xmode').onchange=e=>{S.xmode=e.target.value;CHARTS.forEach(c=>c.resetZoom&&c.resetZoom());redrawAll()};
$('#raw').onchange=e=>{S.raw=e.target.checked;redrawAll()};
$('#theme').onclick=()=>{const d=document.documentElement;d.dataset.theme=d.dataset.theme==='dark'?'light':'dark';redrawAll()};
if(matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches)document.documentElement.dataset.theme='dark';
let rt;addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(redrawAll,150)});
renderTabs();renderView();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./reports")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else folder / "report_all_models.html"
    build_report(load_runs(folder), out, title="Báo cáo huấn luyện - so sánh các mô hình")
