#!/usr/bin/env python3
"""EDA CIFAR-10 cho LAB2 CNN & Transfer Learning.

Phát triển từ EDA Fashion-MNIST của LAB1, giữ phong cách báo cáo HTML offline.

    python EDA.py
    python EDA.py --tsne-samples 0
    python EDA.py --knn-train 3000 --knn-val 1000
    python EDA.py --out reports/eda_report.html

Đầu ra mặc định:
    data/splits.npz         : train_indices / val_indices, chia stratified seed=42
    reports/eda_stats.json : số liệu được tính thực tế (không hardcode kết quả)
    reports/eda_report.html: HTML độc lập, ảnh/biểu đồ nhúng base64

Yêu cầu: numpy, matplotlib, pillow, torchvision (+ torch). scikit-learn chỉ
cần khi chạy t-SNE. Không huấn luyện CNN; không dùng test để chọn mô hình.
Đường dẫn tương đối --data-dir/--out/--stats được hiểu từ thư mục LAB2.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
K, H, W, C = 10, 32, 32, 3
PANEL, LINE, INK, MUTED = '#141f36', '#2a3858', '#e8ecf6', '#98a4c2'
CLASS_COLORS = ['#7aa2ff','#f2b84b','#ef6f6c','#43c6b0','#c58af9',
                '#9ccc65','#ff9a76','#4fc3f7','#f48fb1','#d7c49e']
DIV = LinearSegmentedColormap.from_list('eda_div', ['#5cc8ff','#1b2a4a','#ff7a6b'])
FEATS = [('brightness', 'Độ sáng (luma)'), ('contrast','Độ tương phản'),
         ('saturation','Độ bão hòa màu'), ('colorfulness','Độ phong phú màu'),
         ('edge','Mức biến thiên cạnh'), ('red','Mean kênh R'),
         ('green','Mean kênh G'), ('blue','Mean kênh B')]
SECTIONS = []
INSIGHTS = []
_T0 = time.perf_counter()

plt.rcParams.update({
    'figure.facecolor':PANEL, 'axes.facecolor':PANEL, 'savefig.facecolor':PANEL,
    'axes.edgecolor':LINE,'axes.labelcolor':MUTED,'axes.titlecolor':INK,
    'axes.titleweight':'bold','axes.titlesize':12,'axes.titlelocation':'left',
    'axes.titlepad':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':False,'grid.color':LINE,'grid.alpha':.55,'grid.linewidth':.6,
    'xtick.color':MUTED,'ytick.color':MUTED,'text.color':INK,'font.size':10,
    'legend.frameon':False,'legend.labelcolor':INK,
    'axes.prop_cycle':matplotlib.cycler(color=CLASS_COLORS),
    'figure.dpi':100,'savefig.bbox':'tight','savefig.pad_inches':.16,
})


def log(msg):
    print(f'[{time.perf_counter()-_T0:6.1f}s] {msg}', flush=True)


def fmt(n): return f'{int(n):,}'
def esc(s): return html.escape(str(s))
def jfloat(x): return float(x)


def add_section(sid, title, lead=''):
    SECTIONS.append({'id':sid,'title':title,'lead':lead,'body':[]})


def add(s): SECTIONS[-1]['body'].append(s)
def h3(s): return f'<h3>{esc(s)}</h3>'
def para(s): return f'<p>{s}</p>'
def note(s, kind='info'): return f'<div class="note {kind}">{s}</div>'


def stats(items):
    return '<div class="stats">'+''.join(
        f'<div class="stat"><b>{v}</b><span>{esc(label)}</span>'+
        (f'<small>{esc(sub)}</small>' if sub else '')+'</div>'
        for v,label,sub in ((it[0],it[1],it[2] if len(it)>2 else '') for it in items)
    )+'</div>'


def cname(k):
    return f'<span class="dot" style="background:{CLASS_COLORS[k]}"></span>{esc(CLASSES[k])}'


def table(headers, rows, sortable=True):
    header=''.join(f'<th>{esc(h)}</th>' for h in headers)
    cells=[]
    for row in rows:
        ts=[]
        for cell in row:
            if isinstance(cell,tuple):
                ts.append(f'<td class="num" data-v="{esc(cell[1])}">{cell[0]}</td>')
            else: ts.append(f'<td>{cell}</td>')
        cells.append('<tr>'+''.join(ts)+'</tr>')
    return '<div class="tbl-wrap"><table class="'+('sortable' if sortable else '')+'"><thead><tr>'+header+'</tr></thead><tbody>'+''.join(cells)+'</tbody></table></div>'


def fig_html(fig, caption='', dpi=130):
    buf=io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    img=base64.b64encode(buf.getvalue()).decode('ascii')
    cap='<figcaption>'+caption+'</figcaption>' if caption else ''
    return f'<figure class="fig"><img loading="lazy" src="data:image/png;base64,{img}" alt="{esc(re.sub("<[^>]+>","",caption))[:240]}" title="Bấm để phóng to">{cap}</figure>'


def png_b64(a):
    buf=io.BytesIO()
    Image.fromarray(np.asarray(np.clip(a,0,255), dtype=np.uint8)).save(buf,format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def px_img(a,alt='',size=60):
    if isinstance(a,np.ndarray): a=png_b64(a)
    return f'<img class="px" width="{size}" height="{size}" alt="{esc(alt)}" src="data:image/png;base64,{a}">'


def new_fig(nrows=1,ncols=1,w=None,h=None,**kwargs):
    return plt.subplots(nrows,ncols,figsize=(w or 5.5*ncols,h or 4.2*nrows),constrained_layout=True,**kwargs)


def hide_frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def handles():
    return [Line2D([0],[0],marker='o',ls='',color=CLASS_COLORS[i],label=CLASSES[i],markersize=7) for i in range(K)]


def legend(fig,ncol=5,y=-.02):
    fig.legend(handles=handles(),loc='upper center',ncol=ncol,bbox_to_anchor=(.5,y),columnspacing=1.0,handletextpad=.3)


def annotate_heat(ax,im,M,digits=2,size=8):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            r,g,b,_=im.cmap(im.norm(M[i,j]))
            col='#0b1220' if .299*r+.587*g+.114*b>.55 else '#ffffff'
            ax.text(j,i,f'{M[i,j]:.{digits}f}',ha='center',va='center',color=col,fontsize=size)


def hbox(ax, groups):
    low,high=np.inf,-np.inf
    for i,arr in enumerate(groups):
        q1,med,q3=np.percentile(arr,[25,50,75])
        iqr=q3-q1
        lo=np.min(arr[arr>=q1-1.5*iqr]); hi=np.max(arr[arr<=q3+1.5*iqr])
        low=min(low,lo,float(arr.mean()));high=max(high,hi,float(arr.mean()))
        ax.hlines(i,lo,hi,color=CLASS_COLORS[i],lw=1.5,alpha=.9)
        ax.barh(i,max(q3-q1,1e-7),left=q1,height=.65,color=CLASS_COLORS[i],alpha=.8)
        ax.plot([med,med],[i-.32,i+.32],color=PANEL,lw=2)
        ax.scatter(arr.mean(),i,color=INK,marker='D',s=14,zorder=4)
    pad=max((high-low)*.05,.003)
    ax.set_xlim(low-pad,high+pad);ax.set_yticks(range(K));ax.grid(axis='x');ax.set_axisbelow(True)


def resolve_path(p):
    p=Path(p)
    return p if p.is_absolute() else ROOT/p


def load_cifar10(data_dir:Path):
    try:
        from torchvision.datasets import CIFAR10
    except Exception as e:
        raise RuntimeError(f'Không import được torchvision.datasets.CIFAR10: {e}. Hãy cài PyTorch/torchvision tương thích.') from e
    train=CIFAR10(root=str(data_dir), train=True, download=True)
    test=CIFAR10(root=str(data_dir), train=False, download=True)
    xtr=np.asarray(train.data,dtype=np.uint8)
    ytr=np.asarray(train.targets,dtype=np.int64)
    xte=np.asarray(test.data,dtype=np.uint8)
    yte=np.asarray(test.targets,dtype=np.int64)
    if xtr.shape!=(50000,H,W,C) or xte.shape!=(10000,H,W,C):
        raise ValueError(f'CIFAR-10 không đúng shape: train {xtr.shape}, test {xte.shape}')
    if sorted(set(ytr))!=list(range(K)) or sorted(set(yte))!=list(range(K)):
        raise ValueError('Nhãn CIFAR-10 phải từ 0 tới 9.')
    if list(train.classes)!=CLASSES:
        raise ValueError(f'Class order khác dự kiến: {train.classes}')
    return xtr,ytr,xte,yte


def stratified_split(y,seed=42):
    """Train 45k/val 5k với CIFAR10; mỗi lớp có đúng 4500/500 ảnh."""
    rng=np.random.default_rng(seed)
    parts_tr,parts_val=[],[]
    for k in range(K):
        idx=np.flatnonzero(y==k)
        if len(idx)<2: raise ValueError('Mỗi lớp phải có ít nhất 2 ảnh để split.')
        perm=rng.permutation(idx)
        nval=500 if len(idx)==5000 else max(1,int(round(len(idx)*.1)))
        parts_val.append(perm[:nval]);parts_tr.append(perm[nval:])
    return rng.permutation(np.concatenate(parts_tr)), rng.permutation(np.concatenate(parts_val))


def load_or_create_split(y,path,seed=42,save=True):
    expected_tr,expected_val=stratified_split(y,seed)
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if not {'train_indices','val_indices'}.issubset(z.files):
                raise ValueError(f'{path} thiếu train_indices / val_indices; không tự ghi đè split cũ.')
            tr=z['train_indices'].astype(np.int64);val=z['val_indices'].astype(np.int64)
        # Không buộc so với seed: Train.py có thể đã tạo split stratified khác.
        if len(tr)+len(val)!=len(y) or len(np.unique(np.r_[tr,val]))!=len(y) or min(np.r_[tr,val])<0 or max(np.r_[tr,val])>=len(y):
            raise ValueError('Split có overlap, thiếu mẫu hoặc chỉ số không hợp lệ.')
        if len(y)==50000 and (len(tr)!=45000 or len(val)!=5000):
            raise ValueError('LAB2 yêu cầu split 45k/5k.')
        if len(y)==50000 and not np.all(np.bincount(y[val],minlength=K)==500):
            raise ValueError('Split validation không stratified 500 ảnh/lớp.')
        log(f'Dùng lại split có sẵn: {path}')
        return tr,val,'loaded_existing'
    if save:
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,train_indices=expected_tr,val_indices=expected_val,
                            seed=np.asarray(seed),method=np.asarray('numpy_stratified_per_class'))
        log(f'Đã lưu split: {path}')
    return expected_tr,expected_val,'created_seed_'+str(seed)


def pooled_rgb(x, out=8, batch=1024):
    """Ảnh 32x32 RGB -> đặc trưng 8x8 RGB, average pool 4x4, không resize dataset gốc."""
    assert H%out==0 and W%out==0
    s=H//out
    desc=np.empty((len(x),out*out*C),dtype=np.float32)
    for st in range(0,len(x),batch):
        a=x[st:st+batch].astype(np.float32)/255.0
        d=a.reshape(-1,out,s,out,s,C).mean(axis=(2,4))
        desc[st:st+len(a)]=d.reshape(-1,out*out*C)
    return desc


def extract_features(x,chunk=512):
    res={k:np.empty(len(x),np.float32) for k,_ in FEATS}
    for st in range(0,len(x),chunk):
        a=x[st:st+chunk].astype(np.float32)/255
        R,G,B=a[...,0],a[...,1],a[...,2]
        luma=.2126*R+.7152*G+.0722*B
        rg=R-G;yb=(R+G)/2-B
        # Saturation HSV theo max/min, tránh lỗi chia zero.
        mx=a.max(axis=-1); mn=a.min(axis=-1)
        sat=np.divide(mx-mn,mx,out=np.zeros_like(mx),where=mx>1e-8)
        vals={
            'brightness':luma.mean((1,2)),
            'contrast':luma.std((1,2)),
            'saturation':sat.mean((1,2)),
            'colorfulness':np.sqrt(rg.std((1,2))**2+yb.std((1,2))**2)
                        +.3*np.sqrt(rg.mean((1,2))**2+yb.mean((1,2))**2),
            'edge':(np.abs(np.diff(luma,axis=1)).mean((1,2))+
                    np.abs(np.diff(luma,axis=2)).mean((1,2)))/2,
            'red':R.mean((1,2)), 'green':G.mean((1,2)), 'blue':B.mean((1,2)),
        }
        for key,v in vals.items(): res[key][st:st+len(v)]=v
    return res


def channel_stats(x,idx,chunk=512):
    total=np.zeros(3,np.float64);ss=np.zeros(3,np.float64);n=0
    for st in range(0,len(idx),chunk):
        a=x[idx[st:st+chunk]].astype(np.float64)/255
        total+=a.sum((0,1,2));ss+=(a*a).sum((0,1,2));n+=a.shape[0]*H*W
    mu=total/n
    sd=np.sqrt(np.maximum(ss/n-mu*mu,0))
    return mu,sd


def image_hashes(x):
    # Chỉ bắt trùng tuyệt đối (không bắt gần trùng, ảnh resize hay crop).
    return [hashlib.md5(np.ascontiguousarray(im).tobytes()).digest() for im in x]


def ks_stat(a,b):
    a=np.sort(a);b=np.sort(b);v=np.r_[a,b]
    return float(np.max(np.abs(np.searchsorted(a,v,side='right')/len(a)-np.searchsorted(b,v,side='right')/len(b))))


def prepare(args, *, loader=load_cifar10):
    log('Nạp CIFAR-10 (torchvision)...')
    xtr,ytr,xte,yte=loader(args.data_dir)
    # Test loader synthetic được dùng riêng trong smoke test. Báo cáo thực tế chỉ nhận CIFAR chuẩn.
    tr,val,split_source=load_or_create_split(ytr,args.split_file,args.split_seed,save=True)
    c=SimpleNamespace(args=args,xtr=xtr,ytr=ytr,xte=xte,yte=yte,tr=tr,val=val,
                      split_source=split_source,rng=np.random.default_rng(args.seed),m={})
    c.idx=[np.flatnonzero(ytr==k) for k in range(K)]
    c.fitidx=[tr[ytr[tr]==k] for k in range(K)]
    c.validx=[val[ytr[val]==k] for k in range(K)]
    c.testidx=[np.flatnonzero(yte==k) for k in range(K)]
    log('Tính đặc trưng RGB và texture theo batch ...')
    c.F=extract_features(xtr)
    c.Fte=extract_features(xte)
    log('Tạo descriptor RGB 8×8 bằng average pooling ...')
    c.Z=pooled_rgb(xtr);c.Zte=pooled_rgb(xte)
    c.mu=c.Z[tr].mean(axis=0,dtype=np.float64).astype(np.float32)
    c.mus=np.stack([c.Z[i].mean(0) for i in c.fitidx])
    return c

# --------------------------------------------------------------------------- #
# Các mục EDA: cấu trúc kế thừa LAB1, nội dung đo lại hoàn toàn theo CIFAR-10
# --------------------------------------------------------------------------- #
def sec_overview(c):
    add_section('tong-quan','Tổng quan và toàn vẹn dữ liệu',
        'CIFAR-10 gồm 10 lớp ảnh màu tự nhiên. Bước đầu xác minh cấu trúc, nhãn và các ảnh trùng pixel trước khi huấn luyện CNN.')
    nbytes=(c.xtr.nbytes+c.xte.nbytes)/1e6
    add(stats([(fmt(len(c.ytr)),'ảnh train gốc'),(fmt(len(c.yte)),'ảnh official test'),
               ('32 × 32 × 3','ảnh RGB, uint8'),('10','lớp đối tượng'),
               (f'{c.xtr.min()}–{c.xtr.max()}','dải pixel 0–255'),(f'{nbytes:.1f} MB','dung lượng ảnh thô')]))
    htr=image_hashes(c.xtr);hte=image_hashes(c.xte)
    settr,sette=set(htr),set(hte)
    dup_tr=len(htr)-len(settr);dup_te=len(hte)-len(sette);overlap=len(settr & sette)
    label_by_hash={};conflict=0
    for digest,label in zip(htr,c.ytr):
        if digest in label_by_hash and label_by_hash[digest]!=label: conflict+=1
        label_by_hash.setdefault(digest,label)
    dark_tr=int(np.sum(c.F['brightness']<.04));bright_tr=int(np.sum(c.F['brightness']>.96))
    c.m['integrity']={'exact_duplicates_train':dup_tr,'exact_duplicates_test':dup_te,
        'exact_train_test_overlap':overlap,'exact_conflicting_labels_train':conflict,
        'near_black_train':dark_tr,'near_white_train':bright_tr}
    add(h3('Kiểm tra toàn vẹn'))
    add(table(['Hạng mục','Kết quả','Diễn giải'],[
        ['Shape train / test',f'{c.xtr.shape} / {c.xte.shape}','RGB, 32×32'],
        ['Nhãn hợp lệ',str(np.all((c.ytr>=0)&(c.ytr<K)) and np.all((c.yte>=0)&(c.yte<K))),'0–9'],
        ['Trùng hệt pixel trong train',fmt(dup_tr),'Băm MD5 toàn bộ 3.072 byte/ảnh'],
        ['Trùng hệt pixel trong test',fmt(dup_te),'Không phát hiện ảnh gần trùng'],
        ['Trùng hệt pixel giữa train / test',fmt(overlap),'Chỉ báo cần xem xét, không tự kết luận leakage'],
        ['Trùng pixel khác nhãn trong train',fmt(conflict),'Chỉ kiểm tra ảnh giống hệt'],
        ['Ảnh train gần đen / gần trắng',f'{fmt(dark_tr)} / {fmt(bright_tr)}','Theo ngưỡng luma trung bình 0,04 / 0,96']
    ],sortable=False))
    add(note('Ảnh CIFAR-10 có nền cảnh và nhiều vật thể; <b>không thể coi pixel tối là nền</b> hoặc suy ra bounding box đối tượng như Fashion-MNIST. Phép kiểm tra trùng chỉ phát hiện ảnh giống hệt về pixel.'))


def sec_classes(c):
    add_section('phan-phoi-lop','Phân phối nhãn và chia tập',
        'Chia 45k train / 5k validation phân tầng theo lớp; official test 10k được giữ độc lập, không dùng để chọn checkpoint hay tuning.')
    cnt0=np.bincount(c.ytr,minlength=K);ct=np.bincount(c.ytr[c.tr],minlength=K)
    cv=np.bincount(c.ytr[c.val],minlength=K);ce=np.bincount(c.yte,minlength=K)
    fig,axes=new_fig(1,2,w=15.5,h=5.1)
    for ax,counts,title in [(axes[0],cnt0,'Train gốc'),(axes[1],ce,'Official test')]:
        bars=ax.bar(np.arange(K),counts,color=CLASS_COLORS,width=.75)
        for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(counts)*.012,f'{int(b.get_height()):,}',ha='center',fontsize=8,color=INK)
        ax.set_title(title);ax.set_xticks(range(K),[str(i) for i in range(K)])
        ax.set_ylim(0,max(counts)*1.17);ax.grid(axis='y');ax.set_axisbelow(True)
    add(fig_html(fig,'Các lớp được đánh số từ 0 đến 9 theo thứ tự CIFAR-10 chuẩn. Màu chỉ dùng để phân biệt lớp.'))
    add(table(['Lớp','Train gốc','Train fit','Validation','Official test'],
        [[cname(k),(fmt(cnt0[k]),int(cnt0[k])),(fmt(ct[k]),int(ct[k])),(fmt(cv[k]),int(cv[k])),(fmt(ce[k]),int(ce[k]))] for k in range(K)]))
    add(stats([(fmt(len(c.tr)),'train fit'),(fmt(len(c.val)),'validation'),
               (str(c.args.split_seed),'seed khi tạo split'),
               ('Stratified','kiểu chia train/val')]))
    add(note(f'Chỉ số split lưu tại <code>{esc(c.args.split_file.name)}</code> ({esc(c.split_source)}). ' +
        'Train.py của LAB2 phải <b>đọc lại đúng train_indices và val_indices</b> trong file này; không tự chia lần nữa.', 'good'))
    c.m['split']={'train_fit':len(c.tr),'validation':len(c.val),
        'train_class_counts':ct.tolist(),'val_class_counts':cv.tolist(),
        'test_class_counts':ce.tolist(),'source':c.split_source,'seed_if_created':int(c.args.split_seed)}
    INSIGHTS.append(('good',f'<b>Chia train/validation rõ ràng</b>: {fmt(len(c.tr))} train fit và {fmt(len(c.val))} validation, stratified 10 lớp.'))


def sec_gallery(c,per_class=35):
    add_section('anh-mau','Duyệt ảnh RGB theo lớp',
        'Ảnh thật ở độ phân giải 32×32. Chọn lớp để xem các biến thể về bối cảnh, màu sắc, góc chụp và vật thể.')
    picks=[c.rng.choice(i,min(per_class,len(i)),replace=False) for i in c.fitidx]
    chips=['<button class="chip is-on" data-k="all">Tất cả lớp</button>']
    chips += [f'<button class="chip" data-k="{k}">{cname(k)}</button>' for k in range(K)]
    all_rows=''.join('<div class="g-row"><div class="g-label">'+cname(k)+'</div><div class="thumbs">'+
                     ''.join(px_img(c.xtr[int(i)],CLASSES[k],64) for i in picks[k][:9])+'</div></div>' for k in range(K))
    panels=['<div data-panel="all">'+all_rows+'</div>']
    panels += ['<div data-panel="'+str(k)+'" hidden><div class="thumbs big">'+
               ''.join(px_img(c.xtr[int(i)],CLASSES[k],72) for i in picks[k])+'</div></div>' for k in range(K)]
    add('<div class="gallery"><div class="chips" role="group" aria-label="Lọc lớp">'+''.join(chips)+'</div>'+''.join(panels)+'</div>')
    add(para('Lưu ý: ảnh ô tô / xe tải, mèo / chó, hoặc chim / máy bay có thể giống nhau về hình dạng thô. '
             'Đối với CNN, bối cảnh, cạnh và texture đều có thể ảnh hưởng đến dự đoán; ảnh minh họa không đủ để đo accuracy.'))


def sec_rgb(c):
    add_section('rgb','Thống kê RGB và chuẩn hóa',
        'Thống kê mean/std riêng từng kênh trên <b>train fit</b> (45k), để không đưa thông tin validation/test vào bước fit preprocessing của custom CNN.')
    mu,sd=channel_stats(c.xtr,c.tr)
    mu_te,sd_te=channel_stats(c.xte,np.arange(len(c.xte)))
    c.m['normalization']={'train_mean':mu.tolist(),'train_std':sd.tolist(),
                          'test_mean_observation_only':mu_te.tolist(),'test_std_observation_only':sd_te.tolist(),
                          'scope':'train_fit_only'}
    add(stats([(f'{mu[i]:.4f} / {sd[i]:.4f}',f'Mean / Std kênh {"RGB"[i]}') for i in range(C)]+[
        ('Train fit','nguồn thống kê chuẩn hóa')]))
    # Histogram xấp xỉ bằng mẫu pixel ngẫu nhiên có seed để tránh lưu/cộng mọi pixel.
    ntr=min(6000,len(c.tr));nte=min(1500,len(c.yte))
    seltr=c.rng.choice(c.tr,ntr,replace=False);selte=c.rng.choice(len(c.yte),nte,replace=False)
    fig,axes=new_fig(1,3,w=17.5,h=4.5)
    for ch,(ax,name,col) in enumerate(zip(axes,['Red','Green','Blue'],['#ff7875','#80cc9f','#78aeff'])):
        p=np.bincount(c.xtr[seltr,:,:,ch].ravel(),minlength=256)
        q=np.bincount(c.xte[selte,:,:,ch].ravel(),minlength=256)
        ax.plot(np.arange(256),p/p.sum(),color=col,lw=1.8,label='train fit')
        ax.plot(np.arange(256),q/q.sum(),color='#f2b84b',alpha=.85,lw=1.2,label='test (tham khảo)')
        ax.set_title(name);ax.set_xlabel('pixel 0–255');ax.set_ylabel('tần suất');ax.grid(axis='y');ax.legend()
    add(fig_html(fig,f'Histogram từng kênh, lấy mẫu ngẫu nhiên có seed: {fmt(ntr)} ảnh train fit và {fmt(nte)} ảnh test. Mean/std trong bảng tính trên toàn bộ train fit, không phải mẫu histogram.'))
    rows=[]
    for ch,name in enumerate(('Red','Green','Blue')):
        rows.append([name,(f'{mu[ch]:.5f}',float(mu[ch])),(f'{sd[ch]:.5f}',float(sd[ch])),
                     (f'{mu_te[ch]:.5f}',float(mu_te[ch])),(f'{sd_te[ch]:.5f}',float(sd_te[ch]))])
    add(table(['Kênh','Train mean','Train std','Test mean (tham khảo)','Test std'],rows))
    norm=', '.join(f'{x:.4f}' for x in mu);std=', '.join(f'{x:.4f}' for x in sd)
    add(note(f'Cho custom CNN dùng <code>ToTensor()</code> và <code>Normalize(({norm}), ({std}))</code> nếu muốn chuẩn hóa theo dữ liệu. '
             'Với ResNet18 / MobileNet pretrained, <b>ưu tiên đúng preprocessing gắn với weights API</b> (resize/interpolation/mean/std), '
             'không tự thay bằng thống kê CIFAR-10 nếu chưa thiết kế thí nghiệm riêng.', 'good'))
    INSIGHTS.append(('info',f'<b>Chuẩn hóa RGB</b>: mean train fit = ({norm}), std = ({std}); không sử dụng official test để tính normalization.'))


def sec_features(c):
    add_section('dac-trung-anh','Đặc trưng ảnh phục vụ CNN',
        'So sánh độ sáng, độ tương phản, độ bão hòa, độ phong phú màu, biến thiên cạnh và cường độ RGB theo từng lớp. Đây là thống kê thị giác, không phải feature map học bởi CNN.')
    fig,axes=new_fig(2,4,w=18.8,h=9.0,sharey=True)
    for ax,(key,title) in zip(axes.ravel(),FEATS):
        hbox(ax,[c.F[key][idx] for idx in c.fitidx]);ax.set_title(title,fontsize=10)
    for r in range(2): axes[r,0].set_yticklabels(CLASSES)
    axes[0,0].invert_yaxis()
    add(fig_html(fig,'Hộp: khoảng tứ phân vị; vạch: trung vị; hình thoi: trung bình. Biến thiên cạnh là tổng sai khác độ sáng ngang/dọc được chuẩn hóa.'))
    rows=[]
    for k in range(K):
        rows.append([cname(k)]+[(f'{np.mean(c.F[key][c.fitidx[k]]):.3f}',float(np.mean(c.F[key][c.fitidx[k]]))) for key,_ in FEATS[:5]])
    add(table(['Lớp','Brightness','Contrast','Saturation','Colorfulness','Edge'],rows))
    keys=[k for k,_ in FEATS]
    M=np.corrcoef(np.vstack([c.F[k][c.tr] for k in keys]))
    fig,ax=new_fig(w=8.4,h=7)
    im=ax.imshow(M,cmap=DIV,vmin=-1,vmax=1)
    annotate_heat(ax,im,M,2,8)
    ax.set_xticks(range(len(keys)),[n for _,n in FEATS],rotation=37,ha='right',fontsize=8)
    ax.set_yticks(range(len(keys)),[n for _,n in FEATS],fontsize=8)
    fig.colorbar(im,ax=ax,shrink=.85,label='Pearson r')
    add(fig_html(fig,'Tương quan Pearson trên train fit. Giá trị tương quan cao không chứng minh quan hệ nhân quả hay mức độ quan trọng của CNN.'))
    c.m['feature_mean_by_class']={key:[float(np.mean(c.F[key][idx])) for idx in c.fitidx] for key,_ in FEATS}
    c.m['feature_correlations_train']={keys[i]+'__'+keys[j]:float(M[i,j]) for i in range(len(keys)) for j in range(i+1,len(keys))}
    INSIGHTS.append(('info','<b>Màu và texture đều thay đổi theo lớp</b>; đây là động cơ hợp lý để CNN học đặc trưng không gian, nhưng phải kiểm chứng bằng thí nghiệm train/validation.'))


def sec_means(c):
    add_section('anh-trung-binh','Prototype RGB và mức biến thiên',
        'Ảnh trung bình RGB theo lớp cho thấy màu/bố cục tổng quát. Vùng ảnh sáng trong bản đồ độ lệch chuẩn có độ biến thiên pixel lớn hơn.')
    means=[];std_maps=[]
    for idx in c.fitidx:
        # tính theo lớp, giữ nguyên 32x32x3; không tạo thêm bản float toàn bộ dataset.
        a=c.xtr[idx].astype(np.float32)/255
        means.append(a.mean(axis=0));std_maps.append(a.std(axis=0).mean(axis=2))
    means=np.stack(means);std_maps=np.stack(std_maps)
    fig,axes=new_fig(2,K,w=20,h=5.4)
    for k in range(K):
        axes[0,k].imshow(means[k]);axes[1,k].imshow(std_maps[k],cmap='magma',vmin=0,vmax=.35)
        hide_frame(axes[0,k]);hide_frame(axes[1,k])
        axes[0,k].set_title(CLASSES[k],fontsize=9.5,color=CLASS_COLORS[k],loc='center')
    axes[0,0].set_ylabel('RGB trung bình',color=INK)
    axes[1,0].set_ylabel('Std theo pixel',color=INK)
    add(fig_html(fig,'Hàng trên: prototype RGB 32×32 của mỗi lớp; hàng dưới: trung bình std của ba kênh tại từng vị trí. Chỉ dùng train fit.'))
    fig,axes=new_fig(1,3,w=16.5,h=4.3)
    mean_all=np.mean(means,axis=0)
    # Không dùng trung bình không trọng số nếu số ảnh/lớp khác nhau: tính trực tiếp train fit.
    mean_all=c.xtr[c.tr].mean(axis=0,dtype=np.float64)/255
    axes[0].imshow(mean_all);hide_frame(axes[0]);axes[0].set_title('Ảnh trung bình train fit')
    for ch,(ax,name,col) in enumerate(zip(axes[1:],['Luma theo hàng','Luma theo cột'],['#7aa2ff','#43c6b0'])):
        for k in range(K):
            lum=np.dot(means[k],[.2126,.7152,.0722])
            val=lum.mean(axis=1 if ch==0 else 0)
            ax.plot(val,color=CLASS_COLORS[k],alpha=.9,lw=1.4)
        ax.set_title(name);ax.set_xlabel('vị trí pixel');ax.set_ylabel('luma trung bình');ax.grid()
    legend(fig,ncol=5,y=-.06)
    add(fig_html(fig,'Hồ sơ cường độ theo chiều dọc/ngang cho biết khác biệt bố cục; không phải mask phân đoạn đối tượng.'))
    c.mean_imgs=means
    c.m['prototype_scope']='train_fit_only'


def sec_similarity(c):
    add_section('tuong-dong-lop','Độ tương đồng các lớp trong không gian màu',
        'So sánh trung bình descriptor RGB 8×8 (average pooling), không phải embedding học được từ CNN. Kết quả chỉ có ý nghĩa mô tả.')
    Z=c.mus.astype(np.float64)
    grand=c.mu.astype(np.float64)
    D=Z-grand;nrm=np.linalg.norm(D,axis=1)
    cos=D@D.T/np.maximum(np.outer(nrm,nrm),1e-10)
    eu=np.linalg.norm(Z[:,None]-Z[None],axis=2)
    fig,axes=new_fig(1,2,w=16.8,h=7)
    im=axes[0].imshow(cos,cmap=DIV,vmin=-1,vmax=1)
    annotate_heat(axes[0],im,cos);axes[0].set_title('Cosine của prototype đã center');fig.colorbar(im,ax=axes[0],shrink=.75)
    im=axes[1].imshow(eu,cmap='viridis_r');annotate_heat(axes[1],im,eu);axes[1].set_title('Euclidean trên RGB 8×8');fig.colorbar(im,ax=axes[1],shrink=.75)
    for ax in axes:
        ax.set_xticks(range(K),CLASSES,rotation=40,ha='right');ax.set_yticks(range(K),CLASSES)
    add(fig_html(fig,'Khoảng cách dựa trên 192 chiều ảnh màu thu nhỏ; nhỏ hơn nghĩa là prototype gần nhau theo phép đo này, không dự báo chắc chắn cặp lớp CNN dễ nhầm.'))
    pairs=sorted((float(eu[i,j]),i,j) for i in range(K) for j in range(i+1,K))[:7]
    add(h3('Bảy cặp prototype gần nhất'))
    add(table(['Cặp lớp','Khoảng cách Euclid','Cosine'],
              [[cname(i)+' ↔ '+cname(j),(f'{d:.3f}',d),(f'{cos[i,j]:.3f}',float(cos[i,j]))] for d,i,j in pairs]))
    c.m['closest_prototypes']=[{'a':CLASSES[i],'b':CLASSES[j],'distance':d,'cosine':float(cos[i,j])} for d,i,j in pairs]
    INSIGHTS.append(('info','<b>Độ giống prototype</b> được đo trên ảnh RGB 8×8, khác đặc trưng learned trong CNN; không tự suy ra confusion matrix từ số đo này.'))


def compute_pca(c):
    """PCA trên descriptor 8x8x3=192 chiều của TRAIN FIT בלבד (không fit từ val/test)."""
    X=(c.Z[c.tr]-c.mu).astype(np.float32)
    cov=(X.T@X)/(len(X)-1)
    ev,V=np.linalg.eigh(cov.astype(np.float64))
    order=np.argsort(ev)[::-1]
    c.ev=np.maximum(ev[order],0);c.V=V[:,order].astype(np.float32)
    c.evr=c.ev/np.sum(c.ev);c.cum=np.cumsum(c.evr)
    c.pca_train=(c.Z[c.tr]-c.mu)@c.V[:,:50]
    del X


def sec_pca(c):
    add_section('pca','PCA trên đặc trưng RGB thu nhỏ',
        'Ảnh gốc có 3.072 giá trị. Để EDA chạy được trên máy phổ thông, PCA dùng ảnh được average-pool thành 8×8×3 = 192 chiều; đây KHÔNG phải PCA trên 3.072 pixel gốc.')
    ks={str(p):int(np.searchsorted(c.cum,p/100)+1) for p in (90,95,99)}
    c.m['pca']={'representation':'RGB average pool 32x32 -> 8x8x3 (192 dimensions)',
                 'fit_scope':'train_fit_only','components_for_variance_pct':ks,
                 'pc1_variance_pct':float(c.evr[0]*100),
                 'n_dimensions':c.Z.shape[1]}
    add(stats([(str(ks['90']),'PC giữ 90% phương sai'),(str(ks['95']),'PC giữ 95%'),
               (str(ks['99']),'PC giữ 99%'),(f'{c.evr[0]*100:.1f}%','phương sai PC1')]))
    fig,axes=new_fig(1,3,w=18,h=5,gridspec_kw={'width_ratios':[1,1,1.2]})
    axes[0].bar(np.arange(1,31),c.evr[:30]*100,color='#7aa2ff')
    axes[0].set_title('30 PC đầu tiên');axes[0].set_xlabel('thành phần');axes[0].set_ylabel('% phương sai');axes[0].grid(axis='y')
    axes[1].plot(np.arange(1,len(c.cum)+1),c.cum*100,color='#43c6b0',lw=2)
    for p,col in [(90,'#f2b84b'),(95,'#ef6f6c'),(99,'#c58af9')]:
        ax=axes[1];ax.axhline(p,color=col,ls='--',lw=1);ax.scatter([ks[str(p)]],[p],color=col,zorder=4)
    axes[1].set_ylim(0,102);axes[1].set_title('Phương sai tích lũy');axes[1].set_xlabel('số thành phần');axes[1].grid()
    pick=np.concatenate([c.rng.choice(i,min(140,len(i)),replace=False) for i in c.fitidx])
    proj=(c.Z[pick]-c.mu)@c.V[:,:2]
    axes[2].scatter(proj[:,0],proj[:,1],c=[CLASS_COLORS[k] for k in c.ytr[pick]],s=8,alpha=.55,lw=0)
    axes[2].set_title('PCA 2D (train fit, sample)');axes[2].set_xlabel('PC1');axes[2].set_ylabel('PC2');axes[2].grid()
    legend(fig,ncol=5,y=-.055)
    add(fig_html(fig,'PCA fit trên train fit; scatter là mẫu cân bằng lớp. Phân cụm 2D không đo được accuracy của CNN.'))
    add(h3('Tái tạo ảnh từ descriptor 8×8 và PCA'))
    select_cls=[0,2,3,5,8]
    ks2=[1,5,20,50,100]
    fig,axes=new_fig(len(select_cls),len(ks2)+2,w=14.2,h=1.6*len(select_cls))
    for r,k in enumerate(select_cls):
        i=int(c.validx[k][0]);z=c.Z[i]
        axes[r,0].imshow(c.xtr[i]);hide_frame(axes[r,0])
        if r==0:axes[r,0].set_title('32×32 gốc',fontsize=8.8,loc='center')
        axes[r,0].set_ylabel(CLASSES[k],fontsize=9,color=CLASS_COLORS[k])
        pooled=z.reshape(8,8,3)
        axes[r,1].imshow(pooled);hide_frame(axes[r,1])
        if r==0:axes[r,1].set_title('Pool 8×8',fontsize=8.8,loc='center')
        for j,kk in enumerate(ks2):
            rec=c.mu+(z-c.mu)@c.V[:,:kk]@c.V[:,:kk].T
            axes[r,j+2].imshow(np.clip(rec.reshape(8,8,3),0,1));hide_frame(axes[r,j+2])
            if r==0:axes[r,j+2].set_title(f'PC {kk}',fontsize=8.8,loc='center')
    add(fig_html(fig,'Các ảnh PCA chỉ tái tạo descriptor 8×8, không tái tạo chi tiết 32×32 gốc. Những thay đổi về độ sắc nét là do cả pooling lẫn PCA.'))
    INSIGHTS.append(('info',f'<b>Giảm chiều mô tả</b>: descriptor 192 chiều cần {ks["95"]} PC để giữ 95% phương sai; không diễn giải thành số kênh hoặc layer CNN cần dùng.'))


def sec_tsne(c):
    n=int(c.args.tsne_samples)
    if n<=0:
        add_section('tsne','Bản đồ t-SNE (đã tắt)')
        add(note('Bỏ qua t-SNE theo <code>--tsne-samples 0</code>. Không có số liệu t-SNE để kết luận.'))
        c.m['tsne']={'status':'disabled','samples':0}
        return
    add_section('tsne','Bản đồ t-SNE',
                't-SNE trên một mẫu ảnh train fit. Chỉ giúp quan sát cấu trúc lân cận của descriptor RGB 8×8, không đo hiệu năng CNN.')
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        add(note('Chưa cài scikit-learn, phần t-SNE bị bỏ qua.','warn'))
        c.m['tsne']={'status':'skipped_missing_sklearn','samples':0}
        return
    n=min(n,len(c.tr))
    if n<50:
        add(note('Số mẫu quá nhỏ để chạy t-SNE có ý nghĩa (tối thiểu 50).','warn'))
        c.m['tsne']={'status':'skipped_too_few_samples','samples':n}
        return
    # Giữ phân phối lớp trong tập minh họa; không sử dụng test.
    picks=[]
    per=n//K
    for idx in c.fitidx:
        picks.extend(c.rng.choice(idx,min(per,len(idx)),replace=False).tolist())
    rest=n-len(picks)
    if rest: picks.extend(c.rng.choice(np.setdiff1d(c.tr,np.asarray(picks)),rest,replace=False).tolist())
    sub=np.asarray(picks,dtype=np.int64)
    pca=(c.Z[sub]-c.mu)@c.V[:,:min(30,c.V.shape[1])]
    log(f't-SNE trên {len(sub)} ảnh train fit ...')
    embedding=TSNE(n_components=2,init='pca',perplexity=min(30,max(5,(len(sub)-1)//3)),
                   learning_rate='auto',random_state=int(c.args.seed)).fit_transform(pca)
    fig,ax=new_fig(w=10.2,h=8)
    ax.scatter(embedding[:,0],embedding[:,1],c=[CLASS_COLORS[k] for k in c.ytr[sub]],s=7,alpha=.65,lw=0)
    for k in range(K):
        pts=embedding[c.ytr[sub]==k]
        if len(pts):
            med=np.median(pts,axis=0)
            ax.text(*med,CLASSES[k],color='white',fontsize=8.5,ha='center',va='center',weight='bold',
                    bbox=dict(boxstyle='round,pad=.24',fc=CLASS_COLORS[k],alpha=.88,ec='none'))
    hide_frame(ax);ax.set_title('t-SNE 2 chiều (mẫu train fit)')
    add(fig_html(fig,f'Mẫu {fmt(len(sub))} ảnh, seed={c.args.seed}; PCA 30 chiều trước t-SNE, perplexity ≤30. Vị trí/khoảng cách các cụm trên 2D không tương đương khoảng cách phân loại của CNN.'))
    c.m['tsne']={'status':'generated','samples':len(sub),'seed':int(c.args.seed),'input_pca_components':min(30,c.V.shape[1])}


def confusion(y,p):
    cm=np.zeros((K,K),np.int64)
    np.add.at(cm,(y,p),1)
    return cm


def knn_predict(Ztr,ytr,Zval,k=5,chunk=256):
    sq=(Ztr*Ztr).sum(axis=1)
    out=np.empty(len(Zval),np.int64)
    for st in range(0,len(Zval),chunk):
        q=Zval[st:st+chunk]
        d=sq[None,:]-2*q@Ztr.T
        nn=np.argpartition(d,kth=k-1,axis=1)[:,:k]
        votes=ytr[nn]
        counts=np.zeros((len(q),K),np.int32)
        for j in range(k): counts[np.arange(len(q)),votes[:,j]]+=1
        out[st:st+len(q)]=counts.argmax(axis=1)
    return out


def sec_baseline(c):
    add_section('baseline','Baseline EDA không huấn luyện CNN',
        'Hai mốc tham khảo trên VALIDATION: nearest prototype và k-NN trên PCA của descriptor RGB 8×8. Không lấy kết quả này làm test accuracy và không thay thế thí nghiệm CNN ở LAB2.')
    z=c.Z[c.val]; d=np.square(z[:,None,:]-c.mus[None,:,:]).sum(axis=2)
    pred=d.argmin(axis=1);y=c.ytr[c.val]
    acc=float((pred==y).mean())
    cm_cent=confusion(y,pred)
    add(stats([(f'{acc*100:.2f}%','nearest prototype accuracy',f'validation {fmt(len(y))} ảnh')]))
    c.m['baseline']={'nearest_prototype_val_accuracy':acc,'nearest_prototype_val_n':len(y),
                     'knn_status':'disabled','metric_scope':'validation_only'}
    if c.args.knn_train>0 and c.args.knn_val>0:
        ntr=min(int(c.args.knn_train),len(c.tr));nv=min(int(c.args.knn_val),len(c.val))
        # k-NN với train subset lấy ngẫu nhiên, không fit trên validation.
        itr=c.rng.choice(c.tr,ntr,replace=False);iv=c.rng.choice(c.val,nv,replace=False)
        Ztr=(c.Z[itr]-c.mu)@c.V[:,:50]
        Zv=(c.Z[iv]-c.mu)@c.V[:,:50]
        log(f'Baseline k-NN: {ntr} train / {nv} validation ...')
        knnp=knn_predict(Ztr,c.ytr[itr],Zv)
        acc_knn=float((knnp==c.ytr[iv]).mean())
        cm=confusion(c.ytr[iv],knnp)
        add(stats([(f'{acc_knn*100:.2f}%','k-NN (k=5) + PCA50',f'{fmt(ntr)} train / {fmt(nv)} validation')]))
        c.m['baseline'].update(knn_status='generated',knn_val_accuracy=acc_knn,
                                knn_train_n=ntr,knn_val_n=nv,knn_k=5,knn_pca_components=50)
        fig,axes=new_fig(1,2,w=16.2,h=6.1)
        cmn=cm/np.maximum(cm.sum(axis=1,keepdims=True),1)*100
        im=axes[0].imshow(cmn,cmap='magma',vmin=0,vmax=100)
        annotate_heat(axes[0],im,cmn,0,8.5)
        axes[0].set_title('k-NN confusion matrix — validation (%)')
        axes[0].set_xticks(range(K),CLASSES,rotation=45,ha='right')
        axes[0].set_yticks(range(K),CLASSES)
        axes[0].set_xlabel('dự đoán');axes[0].set_ylabel('nhãn thật')
        fig.colorbar(im,ax=axes[0],shrink=.75)
        recall_cent=cm_cent.diagonal()/np.maximum(cm_cent.sum(1),1)
        recall_knn=cm.diagonal()/np.maximum(cm.sum(1),1)
        yy=np.arange(K)
        axes[1].barh(yy-.2,recall_cent*100,height=.37,label='nearest prototype',color='#f2b84b')
        axes[1].barh(yy+.2,recall_knn*100,height=.37,label='k-NN PCA',color='#7aa2ff')
        axes[1].set_yticks(yy,CLASSES);axes[1].invert_yaxis();axes[1].set_xlim(0,106)
        axes[1].set_xlabel('recall (%)');axes[1].set_title('Recall từng lớp (validation)')
        axes[1].legend(loc='lower right',fontsize=8);axes[1].grid(axis='x')
        add(fig_html(fig,'k-NN đánh giá trên mẫu validation, nên mỗi lớp có thể có support khác nhau; trục confusion chuẩn hóa theo hàng. Không dùng official test để chọn hyperparameter.'))
    else:
        add(note('Đã tắt k-NN; dùng <code>--knn-train 3000 --knn-val 1000</code> để chạy baseline chi phí vừa phải.'))
    add(para('Mốc EDA chỉ là tham chiếu đơn giản. Kết quả CNN cuối cùng phải lấy từ pipeline train/validation và checkpoint, không sao chép metric từ baseline này.'))
    INSIGHTS.append(('info',f'<b>Nearest prototype</b> trên {fmt(len(y))} ảnh validation đạt {acc*100:.2f}%; chỉ là mốc mô tả RGB, không phải kết quả CNN.'))


def sec_outliers(c,n=6):
    add_section('bat-thuong','Ảnh điển hình và ảnh khác biệt',
        'Trong mỗi lớp, xếp ảnh train fit theo khoảng cách đến prototype RGB 8×8. "Xa tâm" KHÔNG đồng nghĩa sai nhãn hoặc ảnh hỏng.')
    rows=[]
    for k in range(K):
        idx=c.fitidx[k]
        dist=np.linalg.norm(c.Z[idx]-c.mus[k],axis=1)
        order=np.argsort(dist)
        near=idx[order[:min(n,len(order))]];far=idx[order[-min(n,len(order)):][::-1]]
        rows.append('<div class="o-row"><div class="g-label">'+cname(k)+
                    f'<small>khoảng cách TB: {np.mean(dist):.2f}</small></div>'+ 
                    '<div class="thumbs">'+''.join(px_img(c.xtr[i],CLASSES[k],56) for i in near)+'</div>'+ 
                    '<div class="thumbs far">'+''.join(px_img(c.xtr[i],CLASSES[k],56) for i in far)+'</div></div>')
    add('<div class="o-head"><span></span><span>Gần prototype (điển hình)</span><span>Xa prototype (khác biệt)</span></div>'+''.join(rows))
    add(note('CIFAR-10 chứa cảnh nền đa dạng: ảnh xa tâm có thể khác về màu trời, góc máy hoặc bối cảnh chứ không nhất thiết là nhiễu. Không xóa tự động.', 'info'))


def sec_shift(c):
    add_section('train-test','So sánh train fit / validation / test',
        'So sánh mô tả phân phối ảnh trước train. Quan sát test ở đây chỉ để kiểm tra dữ liệu, <b>không</b> điều chỉnh kiến trúc hay chọn hyperparameter theo test.')
    fig,axes=new_fig(1,3,w=18,h=4.5)
    selections=[('brightness','Độ sáng'),('saturation','Độ bão hòa'),('edge','Biến thiên cạnh')]
    rows=[];ks_val=[];ks_test=[]
    for ax,(key,title) in zip(axes,selections):
        fit=c.F[key][c.tr];val=c.F[key][c.val];test=c.Fte[key]
        lo=float(min(fit.min(),val.min(),test.min()));hi=float(max(fit.max(),val.max(),test.max()))
        bins=np.linspace(lo,hi+1e-7,46)
        ax.hist(fit,bins=bins,density=True,color='#7aa2ff',alpha=.42,label='train fit')
        ax.hist(val,bins=bins,density=True,histtype='step',color='#43c6b0',lw=1.7,label='validation')
        ax.hist(test,bins=bins,density=True,histtype='step',color='#f2b84b',lw=1.5,label='test (tham khảo)')
        ax.set_title(title);ax.set_ylabel('mật độ');ax.grid(axis='y');ax.legend(fontsize=8)
    add(fig_html(fig,'So sánh phân phối ba tập trên cùng các phép đo màu/texture, không dùng test làm cơ sở chọn mô hình.'))
    for key,name in FEATS:
        fit=c.F[key][c.tr];val=c.F[key][c.val];test=c.Fte[key]
        kv=ks_stat(fit,val);kt=ks_stat(fit,test)
        ks_val.append(kv);ks_test.append(kt)
        rows.append([name,(f'{kv:.4f}',kv),(f'{kt:.4f}',kt),(f'{fit.mean():.3f}',float(fit.mean())),
                     (f'{val.mean():.3f}',float(val.mean())),(f'{test.mean():.3f}',float(test.mean()))])
    add(table(['Đặc trưng','KS fit/val','KS fit/test','Mean fit','Mean val','Mean test'],rows))
    add(para('KS nằm trong [0, 1], càng nhỏ thì hai phân phối thực nghiệm càng gần. Giá trị ngưỡng chỉ mang tính tham khảo; không phải chứng minh hai tập độc lập hoặc phân phối hoàn toàn giống nhau.'))
    c.m['distribution_comparison']={'max_ks_fit_val':max(ks_val),'max_ks_fit_test':max(ks_test),
        'ks_fit_val':{k:v for (k,_),v in zip(FEATS,ks_val)},
        'ks_fit_test_observation_only':{k:v for (k,_),v in zip(FEATS,ks_test)}}


def sec_aug(c):
    add_section('augmentation','Ảnh trước và sau augmentation',
        'Xem ảnh minh họa của các phép thường dùng cho CNN. Đây là preview bằng PIL, <b>không khẳng định</b> Train.py hiện đã triển khai các transform tương ứng.')
    cols=['Gốc','Lật ngang','Crop + padding 4','Color jitter','Xoay 12°','Gaussian blur','Erasing vùng nhỏ']
    classes=[0,2,3,5,8]
    fig,axes=new_fig(len(classes),len(cols),w=15,h=1.85*len(classes))
    rng=np.random.default_rng(c.args.seed)
    for r,k in enumerate(classes):
        src=c.xtr[int(c.fitidx[k][min(3,len(c.fitidx[k])-1)])]
        im=Image.fromarray(src)
        canvas=ImageOps.expand(im,border=4,fill=(0,0,0));x0=int(rng.integers(0,9));y0=int(rng.integers(0,9))
        crop=canvas.crop((x0,y0,x0+W,y0+H))
        jitter=ImageEnhance.Color(ImageEnhance.Brightness(im).enhance(1.12)).enhance(1.25)
        rot=im.rotate(12,resample=Image.Resampling.BILINEAR,fillcolor=(0,0,0))
        er=im.copy();draw=ImageDraw.Draw(er);draw.rectangle((8,9,16,17),fill=(110,110,110))
        out=[im,ImageOps.mirror(im),crop,jitter,rot,im.filter(ImageFilter.GaussianBlur(radius=.75)),er]
        for j,ob in enumerate(out):
            axes[r,j].imshow(ob);hide_frame(axes[r,j])
            if r==0:axes[r,j].set_title(cols[j],fontsize=8.5,loc='center')
            if j==0:axes[r,j].set_ylabel(CLASSES[k],fontsize=9,color=CLASS_COLORS[k])
    add(fig_html(fig,'Minh họa tác động trực quan, seed cố định; thay đổi thông số có thể làm mất thông tin nhãn, đặc biệt với vật thể rất nhỏ 32×32.'))
    add(table(['Phép biến đổi','Custom CNN 32×32','Pretrained backbone'],[
        ['RandomCrop(32, padding=4)','Cân nhắc dùng cho train','Cân nhắc theo resize và weights'],
        ['RandomHorizontalFlip','Thường phù hợp','Thường phù hợp'],
        ['ColorJitter','Mức nhẹ, có xác minh bằng validation','Mức nhẹ, có xác minh bằng validation'],
        ['RandomErasing','Cẩn trọng vì ảnh nhỏ','Dùng sau ToTensor và theo pipeline'],
        ['RandomVerticalFlip','Không khuyến nghị mặc định','Không khuyến nghị mặc định'],
        ['Resize / Normalize','Từ thống kê train fit nếu chọn','Theo preprocessing của pretrained weights API']
    ],sortable=False))
    add(note('Với pretrained ResNet18 / MobileNetV3-Small, resize đầu vào không tạo thêm chi tiết ảnh 32×32. Các phép Random* chỉ áp dụng <b>train</b>; validation/test phải deterministic. Kiểm tra transforms thật trong Train.py trước khi cập nhật data.md.', 'good'))


def sec_cnn_input(c):
    add_section('cnn-input','Chuẩn bị dữ liệu cho CNN và pretrained',
        'CIFAR-10 có 3 kênh màu; mô hình tự xây dựng và ImageNet-pretrained có thể yêu cầu kích thước và normalization khác nhau. Các kiểm tra này là định hướng triển khai, không phải kết quả đã train.')
    fig,axes=new_fig(1,4,w=13.5,h=3.3)
    ex=c.xtr[int(c.tr[0])]
    axes[0].imshow(ex);axes[0].set_title('Gốc RGB 32×32')
    for j,(label,ch,col) in enumerate([('R',0,'Reds'),('G',1,'Greens'),('B',2,'Blues')],start=1):
        axes[j].imshow(ex[:,:,ch],cmap=col,vmin=0,vmax=255);axes[j].set_title(f'Kênh {label}')
    for ax in axes:hide_frame(ax)
    add(fig_html(fig,'CNN xử lý tensor dạng [N, 3, H, W], không phải ảnh 1 kênh hay vector Flatten ở bước đầu.'))
    mu,sd=np.asarray(c.m['normalization']['train_mean']),np.asarray(c.m['normalization']['train_std'])
    add(table(['Pipeline','Dữ liệu đầu vào','Nguyên tắc'],[
        ['Custom CNN','[B, 3, 32, 32]','ToTensor, Normalize từ train fit; augmentation train-only'],
        ['ResNet18 pretrained','[B, 3, H, W]','Dùng preprocessing của weights (ví dụ crop/resize 224×224), ghi lại kích thước thật'],
        ['MobileNetV3 pretrained','[B, 3, H, W]','Tra weights.transforms() cho phiên bản đang dùng'],
        ['Validation / Test','Chỉ biến đổi tất định','Không flip, crop ngẫu nhiên, jitter hay erasing']
    ],sortable=False))
    add(note('Cấu hình <code>Normalize</code> theo thống kê train fit có mean = ('+
        ', '.join(f'{v:.4f}' for v in mu)+') và std = ('+
        ', '.join(f'{v:.4f}' for v in sd)+'). Nếu dùng pretrained, không tự mặc định mean/std này sẽ là tốt nhất.'))


def build_summary():
    section={'id':'tom-tat','title':'Phát hiện và lưu ý chính',
             'lead':'Các chỉ số dưới đây đều xuất phát từ dữ liệu vừa phân tích. Không chứa kết quả train CNN.',
             'body':['<ul class="insights">'+''.join(f'<li class="ins {k}">{v}</li>' for k,v in INSIGHTS)+'</ul>']}
    SECTIONS.insert(1,section)

# Reuse hệ thống layout/navigation/gallery/lightbox/tables của EDA LAB1.
# CSS và JS được NHÚNG TRỰC TIẾP vào file này (không import từ LAB1).
CSS = r"""
:root{--bg:#0b1220;--panel:#141f36;--panel2:#1a2744;--line:#2a3858;--stitch:#3b4b72;--ink:#e8ecf6;--mut:#98a4c2;
--accent:#7aa2ff;--good:#6fd3a4;--warn:#f2b84b;--bad:#ef6f6c;
--serif:"Iowan Old Style","Palatino Linotype",Palatino,Charter,Georgia,"Times New Roman",serif;
--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
*{box-sizing:border-box}
html{color-scheme:dark;scroll-behavior:smooth;scroll-padding-top:24px}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--accent)}
code{background:var(--panel2);padding:.1em .4em;border-radius:4px;font-size:.88em}
.shell{display:grid;grid-template-columns:250px minmax(0,1fr);max-width:1480px;margin:0 auto}
nav.side{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;padding:36px 22px;border-right:1px dashed var(--stitch)}
nav.side strong{display:block;font:600 1.15rem/1.3 var(--serif);margin-bottom:18px}
nav.side a{display:block;padding:7px 12px;margin:0 -12px;color:var(--mut);text-decoration:none;border-radius:6px;font-size:.93rem;line-height:1.35}
nav.side a:hover{color:var(--ink)}
nav.side a[aria-current]{color:var(--ink);background:var(--panel2);box-shadow:inset 3px 0 0 var(--accent)}
a:focus-visible,button:focus-visible,th:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
main{padding:0 clamp(20px,4vw,56px) 96px;min-width:0}
.hero{padding:64px 0 40px}
.hero h1{font:600 clamp(2.1rem,4.6vw,3.5rem)/1.08 var(--serif);margin:0 0 18px;letter-spacing:-.01em;max-width:18ch}
.hero .sub{color:var(--mut);font-size:1.08rem;max-width:62ch;margin:0 0 34px}
.swatches{display:grid;grid-template-columns:repeat(10,1fr);gap:10px}
.sw{margin:0;text-align:center;font-size:.76rem;color:var(--mut);line-height:1.25}
.sw .px{width:100%;height:auto;aspect-ratio:1;display:block;background:#050913;border-radius:5px 5px 0 0}
.sw i{display:block;height:4px;border-radius:0 0 3px 3px;margin-bottom:6px}
.meta{color:var(--mut);font-size:.85rem;margin:28px 0 0}
.meta span+span{margin-left:18px}
section{padding:56px 0 8px;border-top:1px dashed var(--stitch)}
h2{font:600 clamp(1.5rem,2.6vw,2rem)/1.2 var(--serif);margin:0 0 10px}
h3{font:600 1.2rem/1.3 var(--serif);margin:38px 0 12px}
.lead{color:var(--mut);max-width:72ch;margin:0 0 26px}
p{max-width:78ch}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:22px 28px;margin:26px 0}
.stat{border-left:2px solid var(--accent);padding-left:14px}
.stat b{display:block;font:600 1.9rem/1.15 var(--serif);font-variant-numeric:tabular-nums}
.stat span{display:block;color:var(--mut);font-size:.86rem;margin-top:2px}
.stat small{color:var(--mut);font-size:.78rem;opacity:.8}
.fig{margin:24px 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.fig img{display:block;width:100%;height:auto;cursor:zoom-in;border-radius:6px}
.fig figcaption{color:var(--mut);font-size:.88rem;padding:12px 6px 4px;max-width:90ch}
.note{margin:22px 0;padding:14px 18px;border-left:3px solid var(--accent);background:rgba(122,162,255,.08);border-radius:0 8px 8px 0;max-width:86ch}
.note.good{border-color:var(--good);background:rgba(111,211,164,.08)}
.note.warn{border-color:var(--warn);background:rgba(242,184,75,.09)}
.tbl-wrap{overflow-x:auto;margin:18px 0 26px;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
td:first-child{white-space:normal}
tr:last-child td{border-bottom:0}
thead th{background:var(--panel2);font-weight:600;position:sticky;top:0}
td.num{text-align:right;font-variant-numeric:tabular-nums}
table.sortable th{cursor:pointer;user-select:none}
th[data-dir=asc]::after{content:" ▲";font-size:.7em;color:var(--accent)}
th[data-dir=desc]::after{content:" ▼";font-size:.7em;color:var(--accent)}
.dot{display:inline-block;width:.7em;height:.7em;border-radius:50%;margin-right:.55em;vertical-align:baseline}
.tag{display:inline-block;padding:1px 10px;border-radius:99px;font-size:.8rem}
.tag.good{background:rgba(111,211,164,.15);color:var(--good)}
.tag.warn{background:rgba(242,184,75,.15);color:var(--warn)}
.insights{list-style:none;margin:8px 0 0;padding:0;display:grid;gap:12px;max-width:86ch}
.ins{padding:14px 18px 14px 20px;border-left:3px solid var(--accent);background:var(--panel);border-radius:0 8px 8px 0}
.ins.good{border-color:var(--good)}.ins.warn{border-color:var(--warn)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px}
.chip{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:99px;padding:6px 14px;font:inherit;font-size:.88rem;cursor:pointer}
.chip.is-on{background:var(--accent);color:#0b1220;border-color:var(--accent);font-weight:600}
.chip.is-on .dot{box-shadow:0 0 0 1.5px #0b1220}
.thumbs{display:flex;flex-wrap:wrap;gap:5px}
.px{image-rendering:pixelated;background:#050913;border-radius:3px;display:block}
.thumbs.big .px{width:72px;height:72px}
.g-row{display:grid;grid-template-columns:150px 1fr;gap:14px;align-items:center;padding:8px 0;border-bottom:1px dashed var(--line)}
.g-label{font-size:.9rem}.g-label small{display:block;color:var(--mut);font-size:.75rem;margin-top:2px}
.o-head,.o-row{display:grid;grid-template-columns:170px 1fr 1fr;gap:18px;align-items:center}
.o-head{color:var(--mut);font-size:.82rem;margin-bottom:6px}
.o-row{padding:8px 0;border-bottom:1px dashed var(--line)}
.thumbs.far{padding-left:16px;border-left:1px dashed var(--stitch)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(88px,1fr));gap:14px}
.tile{margin:0;text-align:center;font-size:.74rem;line-height:1.35}
.tile .px{width:64px;height:64px;margin:0 auto 4px}
#lb{border:0;padding:0;background:transparent;max-width:96vw;max-height:96vh}
#lb::backdrop{background:rgba(5,9,19,.88)}
#lb img{max-width:96vw;max-height:94vh;display:block;border-radius:8px;cursor:zoom-out}
footer{border-top:1px dashed var(--stitch);margin-top:64px;padding-top:22px;color:var(--mut);font-size:.85rem}
@media (max-width:1000px){
 .shell{grid-template-columns:1fr}
 nav.side{position:sticky;top:0;z-index:5;height:auto;display:flex;gap:4px;overflow-x:auto;padding:10px 16px;border-right:0;border-bottom:1px dashed var(--stitch);background:var(--bg)}
 nav.side strong{display:none}
 nav.side a{white-space:nowrap;margin:0}
 .swatches{grid-template-columns:repeat(5,1fr)}
 .o-head{display:none}.o-row,.g-row{grid-template-columns:1fr}
 .thumbs.far{padding-left:0;border-left:0}
}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
"""
JS = r"""
(()=>{
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const links=$$('nav.side a'),map=new Map(links.map(a=>[a.getAttribute('href').slice(1),a]));
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){links.forEach(l=>l.removeAttribute('aria-current'));const a=map.get(e.target.id);if(a){a.setAttribute('aria-current','true');a.scrollIntoView({block:'nearest'})}}}),{rootMargin:'-15% 0px -75% 0px'});
$$('main section').forEach(s=>io.observe(s));
const lb=$('#lb'),lbi=$('img',lb);
$$('.fig img').forEach(i=>i.addEventListener('click',()=>{lbi.src=i.src;lb.showModal()}));
lb.addEventListener('click',()=>lb.close());
$$('.gallery').forEach(g=>{const chips=$$('.chip',g),panels=$$('[data-panel]',g);
 chips.forEach(ch=>ch.addEventListener('click',()=>{chips.forEach(x=>x.classList.toggle('is-on',x===ch));panels.forEach(p=>p.hidden=p.dataset.panel!==ch.dataset.k)}))});
$$('table.sortable').forEach(t=>{const body=t.tBodies[0],ths=$$('th',t);
 ths.forEach((th,i)=>{th.tabIndex=0;th.setAttribute('role','button');
  const go=()=>{const asc=th.dataset.dir!=='asc';ths.forEach(x=>delete x.dataset.dir);th.dataset.dir=asc?'asc':'desc';
   const val=r=>{const c=r.cells[i],v=c.dataset.v??c.textContent,n=parseFloat(v);return isNaN(n)?v.toLowerCase():n};
   [...body.rows].sort((a,b)=>{const x=val(a),y=val(b);return(x>y?1:x<y?-1:0)*(asc?1:-1)}).forEach(r=>body.appendChild(r))};
  th.addEventListener('click',go);th.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();go()}})})});
})();
"""

EXTRA_CSS = r"""
/* Điều chỉnh cho CIFAR-10 RGB + chế độ sáng, vẫn giữ layout của LAB1. */
:root{--rgb-r:#ff7875;--rgb-g:#80cc9f;--rgb-b:#78aeff}
html[data-theme="light"]{color-scheme:light;--bg:#f4f6fa;--panel:#fff;--panel2:#eaf0f9;
 --line:#d6dfeb;--stitch:#c4d1e3;--ink:#1d2b43;--mut:#52647e;--accent:#255ac7;
 --good:#147750;--warn:#9b6715;--bad:#bb433e}
html[data-theme="light"] .px{background:#e9ecf1}
html[data-theme="light"] .chip.is-on{color:#fff}
html[data-theme="light"] .stat{border-color:var(--accent)}
.hero-actions{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:25px}
.theme-toggle{font:inherit;font-size:.86rem;padding:7px 14px;border:1px solid var(--line);
 background:var(--panel);color:var(--ink);border-radius:8px;cursor:pointer}
.theme-toggle:hover{border-color:var(--accent)}
.hero-label{display:inline-block;font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;
 color:var(--accent);font-weight:700;margin-bottom:10px}
.footnote{font-size:.78rem;color:var(--mut)}
.fig img{max-height:950px;object-fit:contain}
@media(max-width:650px){.hero{padding:38px 0 22px}.swatches{grid-template-columns:repeat(5,1fr)}
 .stats{grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.stat b{font-size:1.45rem}
 th,td{padding:7px 9px;font-size:.82rem}}
"""
EXTRA_JS = r"""
(()=>{
 const theme=document.getElementById('theme-toggle');
 if(theme){theme.addEventListener('click',()=>{
   const next=document.documentElement.dataset.theme==='light'?'dark':'light';
   document.documentElement.dataset.theme=next;
   theme.textContent=next==='light'?'☾ Chế độ tối':'☀ Chế độ sáng';
   theme.setAttribute('aria-label',theme.textContent);
 });}
})();
"""


def render_html(c,elapsed,created_at):
    nav=''.join(f'<a href="#{sec["id"]}">{esc(sec["title"])}</a>' for sec in SECTIONS)
    body=''.join('<section id="'+s['id']+'"><h2>'+esc(s['title'])+'</h2>'+(
        '<p class="lead">'+s['lead']+'</p>' if s['lead'] else '')+''.join(s['body'])+'</section>' for s in SECTIONS)
    sw=''.join('<figure class="sw">'+px_img(np.clip(c.mean_imgs[k]*255,0,255),CLASSES[k],84)+
               f'<i style="background:{CLASS_COLORS[k]}"></i>{esc(CLASSES[k])}</figure>' for k in range(K))
    meta=f'<span>Tạo lúc {created_at:%d/%m/%Y %H:%M}</span><span>Phân tích trong {elapsed:.0f} giây</span>'
    meta+=f'<span>numpy {np.__version__}, matplotlib {matplotlib.__version__}</span>'
    return f'''<!DOCTYPE html><html lang="vi" data-theme="dark"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EDA CIFAR-10 · LAB2 CNN & Transfer Learning</title>
<style>{CSS}\n{EXTRA_CSS}</style></head><body><div class="shell">
<nav class="side" aria-label="Mục lục"><strong>EDA / CIFAR-10</strong>{nav}</nav>
<main><header class="hero">
<span class="hero-label">LAB2 · CNN & Transfer Learning</span>
<h1>Khám phá dữ liệu CIFAR-10</h1>
<p class="sub">60.000 ảnh màu 32×32 thuộc 10 lớp. Từ cấu trúc tập dữ liệu, thống kê RGB,
 texture và ảnh mẫu đến PCA, augmentation và cách chuẩn bị đầu vào cho CNN/pretrained.
 Mỗi số liệu đều được tính từ dữ liệu, không phải kết quả huấn luyện.</p>
<div class="hero-actions"><button id="theme-toggle" class="theme-toggle" type="button" aria-label="Chế độ sáng">☀ Chế độ sáng</button>
<span class="footnote">Chọn ảnh/biểu đồ để xem phóng to · bảng có thể sắp xếp</span></div>
<div class="swatches">{sw}</div><p class="meta">{meta}</p></header>{body}
<footer>Báo cáo tạo bởi LAB2/EDA.py · thống kê ở reports/eda_stats.json · split tại data/splits.npz.
 Mở offline, không dùng CDN. Tập test không được sử dụng để chọn mô hình.</footer>
</main></div><dialog id="lb"><img alt="Biểu đồ phóng to"></dialog>
<script>{JS}\n{EXTRA_JS}</script></body></html>'''


def build_json(c,elapsed,created_at):
    return {'dataset':'CIFAR-10','purpose':'LAB2 CNN & Transfer Learning EDA',
            'generated_at':created_at.isoformat(timespec='seconds'),
            'elapsed_seconds':round(elapsed,2),
            'train_total':len(c.ytr),'official_test_total':len(c.yte),
            'shape':list(c.xtr.shape[1:]),'classes':CLASSES,
            'source':'torchvision.datasets.CIFAR10 (train=True / train=False)',
            'seed_visualization':int(c.args.seed),'split_file':str(c.args.split_file.relative_to(ROOT)) if c.args.split_file.is_relative_to(ROOT) else str(c.args.split_file),
            'report_file':str(c.args.out.relative_to(ROOT)) if c.args.out.is_relative_to(ROOT) else str(c.args.out),
            'metrics':c.m,
            'notes':['RGB mean/std fit only on train_indices','PCA on pooled 8x8x3 descriptors, not raw 3072 dimensions',
                     'k-NN / nearest prototype evaluated on validation only','official test descriptive statistics are NOT for model selection']}


def main(argv=None):
    ap=argparse.ArgumentParser(description='EDA CIFAR-10 RGB -> HTML tự chứa + JSON + split LAB2')
    ap.add_argument('--data-dir',type=Path,default=Path('data'))
    ap.add_argument('--out',type=Path,default=Path('reports/eda_report.html'))
    ap.add_argument('--stats',type=Path,default=Path('reports/eda_stats.json'))
    ap.add_argument('--split-file',type=Path,default=Path('data/splits.npz'))
    ap.add_argument('--split-seed',type=int,default=42)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--tsne-samples',type=int,default=1500,help='0 để tắt t-SNE')
    ap.add_argument('--knn-train',type=int,default=3000,help='0 để tắt k-NN')
    ap.add_argument('--knn-val',type=int,default=1200,help='0 để tắt k-NN')
    args=ap.parse_args(argv)
    for name in ('data_dir','out','stats','split_file'):
        setattr(args,name,resolve_path(getattr(args,name)))
    if args.stats==args.out:
        ap.error('--stats và --out phải là hai file khác nhau')
    if args.tsne_samples<0 or args.knn_train<0 or args.knn_val<0:
        ap.error('Số mẫu không thể âm')
    # Reset state để có thể gọi main() hơn một lần khi test.
    global _T0
    _T0=time.perf_counter();SECTIONS.clear();INSIGHTS.clear()
    c=prepare(args)
    for title,fn in [('Tổng quan',sec_overview),('Phân phối',sec_classes),('Gallery',sec_gallery),
                    ('RGB',sec_rgb),('Đặc trưng',sec_features),('Prototype',sec_means),
                    ('Tương đồng',sec_similarity)]:
        log(title);fn(c)
    log('PCA (192 chiều)');compute_pca(c);sec_pca(c)
    sec_tsne(c)
    for title,fn in [('Baseline trên validation',sec_baseline),('Ảnh khác biệt',sec_outliers),
                     ('Train/val/test',sec_shift),('Augmentation',sec_aug),('CNN inputs',sec_cnn_input)]:
        log(title);fn(c)
    build_summary()
    created_at=datetime.now()
    elapsed=time.perf_counter()-_T0
    doc=render_html(c,elapsed,created_at)
    output=build_json(c,elapsed,created_at)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.stats.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(doc,encoding='utf-8')
    args.stats.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    log(f'Đã lưu HTML: {args.out} ({args.out.stat().st_size/1e6:.2f} MB)')
    log(f'Đã lưu JSON: {args.stats}')
    return args.out,args.stats


if __name__=='__main__':
    try: main()
    except (ValueError,RuntimeError,MemoryError) as e:
        print(f'[ERROR] {e}',file=sys.stderr)
        raise SystemExit(1) from e
