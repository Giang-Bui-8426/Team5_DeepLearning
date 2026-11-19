#!/usr/bin/env python3
"""EDA UCI Appliances Energy Prediction cho Coursework (Topic 8).

Kế thừa cấu trúc báo cáo HTML offline của LAB2/EDA.py; nội dung đo lại hoàn toàn
cho chuỗi thời gian đa biến.

    python EDA.py
    python EDA.py --train-ratio 0.7 --val-ratio 0.15
    python EDA.py --assets-dir ""          # không xuất PNG cho docs

Đầu ra mặc định (đường dẫn tương đối tính từ thư mục Coursework):
    data/splits.json        : ranh giới train/val/test theo thời gian (không xáo)
    reports/eda_stats.json  : số liệu tính thực tế, không hardcode kết quả
    reports/eda_report.html : HTML độc lập, biểu đồ nhúng base64
    docs/assets/eda_*.png   : hình trích cho docs/data.md

Yêu cầu: numpy, pandas, matplotlib. Không huấn luyện mô hình. Thống kê dùng để
fit (tương quan, mean/std, profile theo giờ) chỉ tính trên train; validation dùng
cho baseline tham chiếu; test chỉ mô tả.
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
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
TARGET = 'Appliances'
STEPS_PER_HOUR, STEPS_PER_DAY = 6, 144
ROOMS = {1:'bếp', 2:'phòng khách', 3:'phòng giặt', 4:'văn phòng', 5:'phòng tắm',
         6:'ngoài nhà (phía bắc)', 7:'phòng ủi đồ', 8:'phòng ngủ con', 9:'phòng ngủ bố mẹ'}
WEATHER = ['T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint']
UNITS = {'Appliances':'Wh / 10 phút', 'lights':'Wh / 10 phút', 'T':'°C', 'RH':'%',
         'T_out':'°C', 'Press_mm_hg':'mm Hg', 'RH_out':'%', 'Windspeed':'m/s',
         'Visibility':'km', 'Tdewpoint':'°C', 'rv1':'không đơn vị', 'rv2':'không đơn vị'}
DOW = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN']
SPLITS = ['train', 'val', 'test']
SPLIT_NAMES = {'train':'Train', 'val':'Validation', 'test':'Test'}

PANEL, LINE, INK, MUTED = '#141f36', '#2a3858', '#e8ecf6', '#98a4c2'
ACCENT, WARN, BAD, GOOD = '#7aa2ff', '#f2b84b', '#ef6f6c', '#6fd3a4'
SPLIT_COLORS = {'train':ACCENT, 'val':WARN, 'test':BAD}
DIV = LinearSegmentedColormap.from_list('eda_div', ['#5cc8ff', '#1b2a4a', '#ff7a6b'])
SEQ = LinearSegmentedColormap.from_list('eda_seq', ['#141f36', '#3a5aa8', '#7aa2ff', '#f2b84b'])
SECTIONS, INSIGHTS = [], []
_T0 = time.perf_counter()

plt.rcParams.update({
    'figure.facecolor':PANEL, 'axes.facecolor':PANEL, 'savefig.facecolor':PANEL,
    'axes.edgecolor':LINE, 'axes.labelcolor':MUTED, 'axes.titlecolor':INK,
    'axes.titleweight':'bold', 'axes.titlesize':12, 'axes.titlelocation':'left',
    'axes.titlepad':10, 'axes.spines.top':False, 'axes.spines.right':False,
    'axes.grid':False, 'grid.color':LINE, 'grid.alpha':.55, 'grid.linewidth':.6,
    'xtick.color':MUTED, 'ytick.color':MUTED, 'text.color':INK, 'font.size':10,
    'legend.frameon':False, 'legend.labelcolor':INK,
    'figure.dpi':100, 'savefig.bbox':'tight', 'savefig.pad_inches':.16,
})


# --------------------------------------------------------------------------- #
# Tiện ích HTML/biểu đồ (giữ API của LAB2/EDA.py)
# --------------------------------------------------------------------------- #
def log(msg):
    print(f'[{time.perf_counter()-_T0:6.1f}s] {msg}', flush=True)


def fmt(n): return f'{int(n):,}'.replace(',', '.')
def f2(x, d=2): return f'{x:,.{d}f}'.replace(',', '_').replace('.', ',').replace('_', '.')
def esc(s): return html.escape(str(s))
def r(x, d=5): return float(round(float(x), d))


def add_section(sid, title, lead=''):
    SECTIONS.append({'id':sid, 'title':title, 'lead':lead, 'body':[]})


def add(s): SECTIONS[-1]['body'].append(s)
def h3(s): return f'<h3>{esc(s)}</h3>'
def para(s): return f'<p>{s}</p>'
def note(s, kind='info'): return f'<div class="note {kind}">{s}</div>'


def stats(items):
    return '<div class="stats">'+''.join(
        f'<div class="stat"><b>{v}</b><span>{esc(label)}</span>'+
        (f'<small>{esc(sub)}</small>' if sub else '')+'</div>'
        for v, label, sub in ((it[0], it[1], it[2] if len(it) > 2 else '') for it in items)
    )+'</div>'


def table(headers, rows, sortable=True):
    header = ''.join(f'<th>{esc(h)}</th>' for h in headers)
    cells = []
    for row in rows:
        ts = []
        for cell in row:
            if isinstance(cell, tuple):
                ts.append(f'<td class="num" data-v="{esc(cell[1])}">{cell[0]}</td>')
            else: ts.append(f'<td>{cell}</td>')
        cells.append('<tr>'+''.join(ts)+'</tr>')
    return ('<div class="tbl-wrap"><table class="'+('sortable' if sortable else '')+'"><thead><tr>'
            +header+'</tr></thead><tbody>'+''.join(cells)+'</tbody></table></div>')


def num(x, d=3): return (f2(x, d), x)


ASSETS: Path | None = None


def fig_html(fig, caption='', asset=None, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    if asset and ASSETS is not None:
        ASSETS.mkdir(parents=True, exist_ok=True)
        (ASSETS/f'{asset}.png').write_bytes(buf.getvalue())
    plt.close(fig)
    img = base64.b64encode(buf.getvalue()).decode('ascii')
    cap = '<figcaption>'+caption+'</figcaption>' if caption else ''
    return (f'<figure class="fig"><img loading="lazy" src="data:image/png;base64,{img}" '
            f'alt="{esc(re.sub("<[^>]+>", "", caption))[:240]}" title="Bấm để phóng to">{cap}</figure>')


def new_fig(nrows=1, ncols=1, w=None, h=None, **kwargs):
    return plt.subplots(nrows, ncols, figsize=(w or 5.5*ncols, h or 4.2*nrows), constrained_layout=True, **kwargs)


def annotate_heat(ax, im, M, digits=2, size=8):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            rr, g, b, _ = im.cmap(im.norm(M[i, j]))
            col = '#0b1220' if .299*rr+.587*g+.114*b > .55 else '#ffffff'
            ax.text(j, i, f'{M[i, j]:.{digits}f}', ha='center', va='center', color=col, fontsize=size)


def resolve_path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT/p


def rel(p):
    p = Path(p)
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


# --------------------------------------------------------------------------- #
# Phép tính
# --------------------------------------------------------------------------- #
def acf(x, nlags):
    x = np.asarray(x, float)-np.mean(x)
    f = np.fft.rfft(x, 2*len(x))
    a = np.fft.irfft(f*np.conj(f))[:nlags+1]
    return a/a[0]


def pacf_from_acf(rho, nlags):
    """Durbin-Levinson trên ACF mẫu."""
    out = np.zeros(nlags+1); out[0] = 1.0
    phi = np.zeros((nlags+1, nlags+1))
    phi[1, 1] = out[1] = rho[1]
    for k in range(2, nlags+1):
        num_ = rho[k]-np.dot(phi[k-1, 1:k], rho[k-1:0:-1])
        den = 1.0-np.dot(phi[k-1, 1:k], rho[1:k])
        phi[k, k] = out[k] = num_/den
        phi[k, 1:k] = phi[k-1, 1:k]-phi[k, k]*phi[k-1, k-1:0:-1]
    return out


def ks_stat(a, b):
    a, b = np.sort(a), np.sort(b)
    v = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, v, 'right')/len(a)-np.searchsorted(b, v, 'right')/len(b))))


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy())[0, 1])


def regress_metrics(y, p):
    e = p-y
    return {'rmse':r(np.sqrt(np.mean(e**2)), 4), 'mae':r(np.mean(np.abs(e)), 4),
            'r2':r(1-np.sum(e**2)/np.sum((y-y.mean())**2), 5), 'n':int(len(y))}


# --------------------------------------------------------------------------- #
# Nạp dữ liệu và split
# --------------------------------------------------------------------------- #
def load_csv(path: Path):
    if not path.exists():
        raise RuntimeError(f'Không tìm thấy {path}. Tải energydata_complete.csv từ UCI vào thư mục này.')
    raw = path.read_bytes()
    df = pd.read_csv(io.BytesIO(raw))
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    return df, hashlib.sha256(raw).hexdigest()


def chrono_split(n, train_ratio, val_ratio):
    n_tr = int(round(n*train_ratio)); n_va = int(round(n*val_ratio))
    if n_tr <= 0 or n_va <= 0 or n_tr+n_va >= n:
        raise ValueError('Tỉ lệ split không hợp lệ')
    return {'train':(0, n_tr), 'val':(n_tr, n_tr+n_va), 'test':(n_tr+n_va, n)}


def prepare(args):
    df, sha = load_csv(args.csv)
    feats = [c for c in df.columns if c not in ('date', TARGET)]
    bounds = chrono_split(len(df), args.train_ratio, args.val_ratio)
    split = np.empty(len(df), dtype=object)
    for k, (a, b) in bounds.items(): split[a:b] = k
    df['split'] = split
    c = SimpleNamespace(args=args, df=df, feats=feats, bounds=bounds, sha=sha, m={},
                        y=df[TARGET].to_numpy(float), t=df['date'])
    c.tr = df.iloc[slice(*bounds['train'])]
    c.m['source'] = {'file':rel(args.csv), 'sha256':sha}
    return c


def save_split(c):
    out = {'method':'chronological, no shuffle', 'ratios':{'train':c.args.train_ratio, 'val':c.args.val_ratio,
           'test':r(1-c.args.train_ratio-c.args.val_ratio, 4)}, 'csv_sha256':c.sha, 'splits':{}}
    for k, (a, b) in c.bounds.items():
        out['splits'][k] = {'start_row':a, 'end_row_exclusive':b, 'rows':b-a,
                            'start_time':str(c.t.iloc[a]), 'end_time':str(c.t.iloc[b-1])}
    c.args.split_file.parent.mkdir(parents=True, exist_ok=True)
    c.args.split_file.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    c.m['split'] = out


def shade_splits(ax, c):
    for k, (a, b) in c.bounds.items():
        ax.axvspan(c.t.iloc[a], c.t.iloc[b-1], color=SPLIT_COLORS[k], alpha=.08, lw=0)
        ax.text(c.t.iloc[a+(b-a)//2], 1.0, SPLIT_NAMES[k], transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', color=SPLIT_COLORS[k], fontsize=9)


# --------------------------------------------------------------------------- #
# Các mục EDA
# --------------------------------------------------------------------------- #
def sec_overview(c):
    df = c.df
    dt = df['date'].diff().dropna()
    step_counts = {str(k):int(v) for k, v in dt.value_counts().items()}
    gaps = int((dt != pd.Timedelta(minutes=10)).sum())
    miss = int(df[[TARGET]+c.feats].isna().sum().sum())
    dup_t = int(df['date'].duplicated().sum())
    dup_rows = int(df.drop(columns=['date', 'split']).duplicated().sum())
    nonfinite = int((~np.isfinite(df[[TARGET]+c.feats].to_numpy(float))).sum())
    days = (df['date'].iloc[-1]-df['date'].iloc[0])/pd.Timedelta(days=1)
    c.m['integrity'] = {'rows':len(df), 'columns':int(df.shape[1]-1), 'features':len(c.feats),
        'start':str(df['date'].iloc[0]), 'end':str(df['date'].iloc[-1]), 'span_days':r(days, 3),
        'step_counts':step_counts, 'irregular_steps':gaps, 'missing_values':miss,
        'duplicate_timestamps':dup_t, 'duplicate_rows_excluding_time':dup_rows, 'non_finite':nonfinite}

    groups = [
        ('Mục tiêu', TARGET, 'Điện năng thiết bị gia dụng trong 10 phút', UNITS[TARGET]),
        ('Năng lượng', 'lights', 'Điện năng đèn chiếu sáng trong 10 phút', UNITS['lights']),
        ('Cảm biến trong nhà', 'T1–T9', 'Nhiệt độ 9 vị trí: '+', '.join(f'T{k} {v}' for k, v in ROOMS.items()), '°C'),
        ('Cảm biến trong nhà', 'RH_1–RH_9', 'Độ ẩm tương đối cùng 9 vị trí', '%'),
        ('Trạm thời tiết', ', '.join(WEATHER), 'Trạm Chièvres, nội suy về bước 10 phút', '°C, mm Hg, %, m/s, km, °C'),
        ('Biến ngẫu nhiên', 'rv1, rv2', 'Tác giả dataset thêm vào để kiểm tra chọn đặc trưng', '—'),
    ]
    add_section('tong-quan', 'Tổng quan và toàn vẹn dữ liệu',
                'Một chuỗi thời gian đa biến, mỗi dòng là một mốc 10 phút. Không có nhãn lớp; mục tiêu là giá trị liên tục <code>Appliances</code>.')
    add(stats([(fmt(len(df)), 'mốc thời gian', f'{df["date"].iloc[0]:%d/%m/%Y} → {df["date"].iloc[-1]:%d/%m/%Y}'),
               (f2(days, 1), 'ngày liên tục'), (str(len(c.feats)), 'đặc trưng đầu vào', 'không tính date và mục tiêu'),
               (fmt(miss), 'giá trị thiếu'), (fmt(gaps), 'bước khác 10 phút'), (fmt(dup_t), 'timestamp trùng')]))
    add(table(['Nhóm', 'Cột', 'Ý nghĩa', 'Đơn vị'], [[esc(a), f'<code>{esc(b)}</code>', esc(d), esc(u)] for a, b, d, u in groups], sortable=False))
    if miss == 0 and gaps == 0 and dup_t == 0 and nonfinite == 0:
        add(note('Chuỗi đều 10 phút, không thiếu, không trùng timestamp, mọi giá trị hữu hạn. '
                 'Cửa sổ trượt có thể cắt trực tiếp theo chỉ số dòng mà không cần nội suy.', 'good'))
        INSIGHTS.append(('good', f'Dữ liệu sạch: {fmt(len(df))} mốc liên tục, bước 10 phút đều, 0 giá trị thiếu, 0 timestamp trùng.'))
    else:
        add(note('Phát hiện bất thường về toàn vẹn — xem <code>metrics.integrity</code>.', 'warn'))


def sec_target(c):
    y = c.y
    q = np.percentile(y, [1, 5, 25, 50, 75, 90, 95, 99])
    srt = np.sort(y)[::-1]
    top10_share = srt[:int(np.ceil(.1*len(y)))].sum()/y.sum()
    dy = np.diff(y)
    big_jump = int((np.abs(dy) >= 200).sum())
    sk = float(pd.Series(y).skew()); sk_log = float(pd.Series(np.log1p(y)).skew())
    kurt = float(pd.Series(y).kurt())
    vc = pd.Series(y).value_counts()
    mult10 = bool(np.all(np.mod(y, 10) == 0))
    c.m['target'] = {'unit':UNITS[TARGET], 'mean':r(y.mean(), 4), 'std':r(y.std(ddof=1), 4), 'min':r(y.min()), 'max':r(y.max()),
        'percentiles':{str(p):r(v) for p, v in zip([1, 5, 25, 50, 75, 90, 95, 99], q)},
        'skew':r(sk, 4), 'skew_log1p':r(sk_log, 4), 'excess_kurtosis':r(kurt, 4), 'unique_values':int(vc.size),
        'all_multiples_of_10':mult10, 'top_values':{str(int(k)):int(v) for k, v in vc.head(5).items()},
        'top10pct_points_energy_share':r(top10_share, 4), 'abs_step_change_ge_200':big_jump,
        'step_change_abs_p99':r(np.percentile(np.abs(dy), 99))}

    fig, ax = new_fig(1, 3, w=15, h=4.2)
    ax[0].hist(y, bins=np.arange(0, y.max()+20, 20), color=ACCENT, alpha=.9)
    for p, v in [('trung vị', q[3]), ('p95', q[6])]:
        ax[0].axvline(v, color=WARN, ls='--', lw=1); ax[0].text(v, ax[0].get_ylim()[1]*.92, f' {p} {v:.0f}', color=WARN, fontsize=9)
    ax[0].set(title='Appliances (Wh)', xlabel='Wh / 10 phút', ylabel='Số mốc')
    ax[1].hist(np.log1p(y), bins=60, color=GOOD, alpha=.9)
    ax[1].set(title='log(1 + Appliances)', xlabel='log1p(Wh)')
    cum = np.cumsum(srt)/srt.sum()
    ax[2].plot(np.arange(1, len(y)+1)/len(y)*100, cum*100, color=WARN)
    ax[2].axvline(10, color=MUTED, ls=':', lw=1); ax[2].axhline(top10_share*100, color=MUTED, ls=':', lw=1)
    ax[2].set(title='Tỉ trọng năng lượng tích lũy', xlabel='% mốc tiêu thụ cao nhất', ylabel='% tổng Wh')
    for a in ax: a.grid(axis='y'); a.set_axisbelow(True)

    add_section('muc-tieu', 'Biến mục tiêu Appliances',
                'Phân phối lệch phải mạnh: phần lớn thời gian tiêu thụ nền thấp, xen kẽ các đỉnh ngắn khi thiết bị bật.')
    add(stats([(f2(y.mean(), 1), 'trung bình (Wh)', f'std {f2(y.std(ddof=1), 1)}'), (f2(q[3], 0), 'trung vị (Wh)'),
               (f2(q[6], 0), 'p95 (Wh)', f'max {f2(y.max(), 0)}'), (f2(sk, 2), 'skewness', f'sau log1p: {f2(sk_log, 2)}'),
               (f2(top10_share*100, 1)+'%', 'năng lượng từ 10% mốc cao nhất'), (fmt(big_jump), 'bước nhảy |Δ| ≥ 200 Wh')]))
    add(fig_html(fig, 'Trái: phân phối gốc; giữa: sau <code>log1p</code>; phải: đường tích lũy năng lượng theo mốc giảm dần.', 'eda_target_distribution'))
    add(table(['Phân vị', 'p1', 'p5', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99'],
              [['Wh']+[num(v, 0) for v in q]], sortable=False))
    add(note(f'Mục tiêu chỉ nhận {vc.size} giá trị rời rạc, '+('tất cả là bội số của 10 Wh' if mult10 else 'không đều bước')+
             f'; hai giá trị phổ biến nhất là {int(vc.index[0])} Wh ({vc.iloc[0]/len(y)*100:.1f}%) và {int(vc.index[1])} Wh ({vc.iloc[1]/len(y)*100:.1f}%). '
             'Đây là độ phân giải của đồng hồ đo, không phải lỗi.'))
    add(note(f'Skewness {f2(sk, 2)} giảm còn {f2(sk_log, 2)} sau <code>log1p</code>. Huấn luyện trên log1p rồi đảo về Wh khi tính metric là '
             'phương án cần thí nghiệm; MSE trên thang gốc sẽ bị chi phối bởi số ít đỉnh.', 'warn'))
    INSIGHTS.append(('warn', f'Mục tiêu lệch phải mạnh (skew {f2(sk, 2)}; trung vị {q[3]:.0f} Wh, p95 {q[6]:.0f} Wh, max {y.max():.0f} Wh). '
                             f'10% mốc cao nhất chiếm {f2(top10_share*100, 1)}% tổng năng lượng → RMSE bị chi phối bởi đỉnh.'))


def sec_timeline(c):
    df = c.df.set_index('date')
    daily = df[[TARGET, 'T_out', 'T2']].resample('D').mean()
    fig, ax = new_fig(2, 1, w=15, h=7.2, sharex=True)
    ax[0].plot(c.t, c.y, color=ACCENT, lw=.35, alpha=.55, label='10 phút')
    ax[0].plot(daily.index+pd.Timedelta(hours=12), daily[TARGET], color=WARN, lw=1.8, label='trung bình ngày')
    ax[0].set(title='Appliances theo thời gian', ylabel='Wh / 10 phút'); ax[0].legend(loc='upper left', ncol=2)
    ax[1].plot(daily.index+pd.Timedelta(hours=12), daily['T_out'], color=BAD, lw=1.6, label='T_out (ngoài trời)')
    ax[1].plot(daily.index+pd.Timedelta(hours=12), daily['T2'], color=GOOD, lw=1.6, label='T2 (phòng khách)')
    ax[1].set(title='Nhiệt độ trung bình ngày', ylabel='°C'); ax[1].legend(loc='upper left', ncol=2)
    for a in ax: shade_splits(a, c); a.grid(axis='y'); a.set_axisbelow(True)

    week = slice(STEPS_PER_DAY*7, STEPS_PER_DAY*14)
    fig2, ax2 = new_fig(1, 1, w=15, h=3.6)
    ax2.plot(c.t.iloc[week], c.y[week], color=ACCENT, lw=1)
    ax2.set(title=f'Một tuần mẫu: {c.t.iloc[week.start]:%d/%m} – {c.t.iloc[week.stop-1]:%d/%m/%Y}', ylabel='Wh / 10 phút')
    ax2.grid(axis='y')
    add_section('thoi-gian', 'Diễn biến theo thời gian',
                'Dữ liệu kéo dài từ giữa mùa đông đến cuối mùa xuân; dải màu nền đánh dấu ranh giới train/validation/test.')
    add(fig_html(fig, 'Trên: mục tiêu 10 phút và trung bình ngày. Dưới: nhiệt độ ngoài trời và trong nhà tăng dần theo mùa.', 'eda_timeline'))
    add(fig_html(fig2, 'Phóng to một tuần: tiêu thụ nền ổn định, các đỉnh xuất hiện thành cụm ngắn trong ngày.', 'eda_week_sample'))


def sec_seasonality(c):
    tr = c.tr
    hour = tr['date'].dt.hour; dow = tr['date'].dt.dayofweek
    g = tr.groupby(hour)[TARGET]
    med, q25, q75, mean_h = g.median(), g.quantile(.25), g.quantile(.75), g.mean()
    by_dow = tr.groupby(dow)[TARGET].mean()
    heat = tr.pivot_table(index=dow, columns=hour, values=TARGET, aggfunc='mean').to_numpy()
    wk = tr[dow < 5][TARGET].mean(); we = tr[dow >= 5][TARGET].mean()
    night = tr[(hour >= 0) & (hour < 6)][TARGET].mean()
    peak_h = int(mean_h.idxmax()); low_h = int(mean_h.idxmin())
    c.m['seasonality_train'] = {'hourly_mean':{str(int(k)):r(v, 3) for k, v in mean_h.items()},
        'hourly_median':{str(int(k)):r(v, 3) for k, v in med.items()},
        'dow_mean':{DOW[int(k)]:r(v, 3) for k, v in by_dow.items()}, 'peak_hour':peak_h, 'lowest_hour':low_h,
        'peak_to_low_ratio':r(mean_h.max()/mean_h.min(), 4), 'weekday_mean':r(wk, 3), 'weekend_mean':r(we, 3),
        'night_0_6_mean':r(night, 3)}

    fig, ax = new_fig(1, 2, w=15, h=4.4, gridspec_kw={'width_ratios':[1.35, 1]})
    hrs = med.index.to_numpy()
    ax[0].fill_between(hrs, q25, q75, color=ACCENT, alpha=.22, lw=0, label='IQR')
    ax[0].plot(hrs, med, color=ACCENT, lw=2, marker='o', ms=3, label='trung vị')
    ax[0].plot(hrs, mean_h, color=WARN, lw=1.6, ls='--', label='trung bình')
    ax[0].set(title='Profile theo giờ trong ngày (train)', xlabel='Giờ', ylabel='Wh / 10 phút', xticks=range(0, 24, 2))
    ax[0].legend(loc='upper left', ncol=3); ax[0].grid(axis='y')
    im = ax[1].imshow(heat, aspect='auto', cmap=SEQ)
    ax[1].set(title='Trung bình theo thứ × giờ (train)', yticks=range(7), yticklabels=DOW, xticks=range(0, 24, 3), xlabel='Giờ')
    fig.colorbar(im, ax=ax[1], shrink=.85, label='Wh')
    add_section('chu-ky', 'Tính chu kỳ ngày và tuần',
                'Thống kê trên train. Tiêu thụ phụ thuộc mạnh vào giờ trong ngày, khác biệt giữa các thứ nhỏ hơn.')
    add(stats([(f'{peak_h}h', 'giờ trung bình cao nhất', f'{mean_h.max():.1f} Wh'), (f'{low_h}h', 'giờ thấp nhất', f'{mean_h.min():.1f} Wh'),
               (f2(mean_h.max()/mean_h.min(), 2)+'×', 'tỉ lệ cao/thấp theo giờ'), (f2(night, 1), 'trung bình 0h–6h (Wh)'),
               (f2(wk, 1), 'trung bình ngày thường'), (f2(we, 1), 'trung bình cuối tuần')]))
    add(fig_html(fig, 'Khoảng IQR rộng ở giờ cao điểm cho thấy cùng một giờ có thể nhỏ hoặc rất lớn: thông tin giờ cần nhưng chưa đủ để dự báo đỉnh.', 'eda_seasonality'))
    add(note('Mạng chỉ nhìn cửa sổ ngắn sẽ không biết “bây giờ là mấy giờ”. Mã hoá giờ trong ngày dạng sin/cos nên đưa vào đầu vào; '
             'chênh lệch theo thứ nhỏ nên không cần mã hoá thứ (đề xuất, chưa kiểm chứng bằng mô hình).'))
    INSIGHTS.append(('info', f'Chu kỳ ngày rõ: trung bình giờ {peak_h}h cao gấp {f2(mean_h.max()/mean_h.min(), 2)} lần giờ {low_h}h (train); '
                             f'ngày thường {f2(wk, 1)} Wh so với cuối tuần {f2(we, 1)} Wh; thứ cao nhất là {DOW[int(by_dow.idxmax())]} ({f2(by_dow.max(), 1)} Wh).'))


def sec_acf(c, max_lag=2*STEPS_PER_DAY, pacf_lags=36):
    ytr = c.tr[TARGET].to_numpy(float)
    a_raw = acf(ytr, max_lag); a_log = acf(np.log1p(ytr), max_lag)
    p_log = pacf_from_acf(a_log, pacf_lags)
    conf = 1.96/np.sqrt(len(ytr))
    keys = [1, 2, 3, 6, 12, 18, 36, 72, 144, 288]
    below = int(np.argmax(a_log[1:] < .2))+1
    sig_pacf = [int(k) for k in range(1, pacf_lags+1) if abs(p_log[k]) > conf]
    c.m['autocorrelation_train'] = {'acf_raw':{str(k):r(a_raw[k], 4) for k in keys}, 'acf_log1p':{str(k):r(a_log[k], 4) for k in keys},
        'pacf_log1p':{str(k):r(p_log[k], 4) for k in range(1, pacf_lags+1)}, 'confidence_95':r(conf, 5),
        'first_lag_acf_log1p_below_0.2':below, 'local_max_lag_near_1day':int(STEPS_PER_DAY-6+np.argmax(a_log[STEPS_PER_DAY-6:STEPS_PER_DAY+7])),
        'pacf_significant_lags':sig_pacf}
    fig, ax = new_fig(1, 2, w=15, h=4.3, gridspec_kw={'width_ratios':[1.6, 1]})
    lags = np.arange(max_lag+1)
    ax[0].plot(lags/STEPS_PER_HOUR, a_raw, color=ACCENT, lw=1.4, label='Wh gốc')
    ax[0].plot(lags/STEPS_PER_HOUR, a_log, color=GOOD, lw=1.4, label='log1p')
    ax[0].axhspan(-conf, conf, color=MUTED, alpha=.15, lw=0)
    for h in (24, 48): ax[0].axvline(h, color=WARN, ls=':', lw=1)
    ax[0].set(title='ACF của mục tiêu (train)', xlabel='Độ trễ (giờ)', ylabel='tự tương quan', xticks=range(0, 49, 6))
    ax[0].legend(); ax[0].grid(axis='y')
    k = np.arange(1, pacf_lags+1)
    ax[1].bar(k, p_log[1:], color=[GOOD if abs(v) > conf else MUTED for v in p_log[1:]], width=.7)
    ax[1].axhspan(-conf, conf, color=MUTED, alpha=.15, lw=0)
    ax[1].set(title='PACF log1p (train)', xlabel='Độ trễ (bước 10 phút)'); ax[1].grid(axis='y')
    add_section('tu-tuong-quan', 'Tự tương quan và độ dài cửa sổ',
                'ACF/PACF tính trên train để gợi ý độ dài lịch sử đưa vào mô hình.')
    add(stats([(f2(a_raw[1], 3), 'ACF lag 1 (10 phút)', f'log1p: {f2(a_log[1], 3)}'), (f2(a_log[6], 3), 'ACF log1p lag 6 (1 giờ)'),
               (f'{below} bước', 'ACF log1p lần đầu < 0,2', f'≈ {below/STEPS_PER_HOUR:.1f} giờ'),
               (f2(a_log[144], 3), 'ACF log1p lag 144 (24 giờ)'), (f2(a_log[72], 3), 'ACF log1p lag 72 (12 giờ)')]))
    add(fig_html(fig, 'Vùng xám là khoảng tin cậy 95% của nhiễu trắng. Đỉnh lặp lại tại 24 h và 48 h xác nhận chu kỳ ngày.', 'eda_acf'))
    add(table(['Độ trễ', 'Thời gian', 'ACF gốc', 'ACF log1p'],
              [[str(k), f'{k*10//60} h {k*10%60:02d}′', num(a_raw[k]), num(a_log[k])] for k in keys], sortable=False))
    add(note(f'Thông tin gần nhất mạnh nhất (ACF lag 1 = {f2(a_raw[1], 3)}) rồi giảm nhanh, nhưng tương quan quay lại tại 24 h '
             f'(log1p {f2(a_log[144], 3)}). Gợi ý thử cửa sổ từ vài giờ đến 24 h (36–144 bước); chọn cụ thể bằng validation ở tài liệu training.'))
    INSIGHTS.append(('info', f'Tự tương quan mạnh ở bước kế tiếp (lag 1: {f2(a_raw[1], 3)}) và có chu kỳ 24 h (lag 144, log1p: {f2(a_log[144], 3)}) '
                             '→ cửa sổ đầu vào nên thử trong khoảng 36–144 bước.'))


def sec_correlation(c):
    tr = c.tr
    yt = tr[TARGET].to_numpy(float); ynext = np.r_[yt[1:], np.nan]
    rows = []
    for f in c.feats:
        x = tr[f].to_numpy(float)
        if np.std(x) == 0: continue
        pear = float(np.corrcoef(x, yt)[0, 1]); spr = spearman(x, yt)
        lead = float(np.corrcoef(x[:-1], ynext[:-1])[0, 1])
        rows.append((f, pear, spr, lead))
    rows.sort(key=lambda z: -abs(z[2]))
    c.m['correlation_train'] = {f:{'pearson':r(p, 4), 'spearman':r(s, 4), 'pearson_with_next_step':r(l, 4)} for f, p, s, l in rows}

    X = tr[c.feats].to_numpy(float)
    C = np.corrcoef(X, rowvar=False)
    pairs = sorted(((abs(C[i, j]), c.feats[i], c.feats[j], C[i, j]) for i in range(len(c.feats)) for j in range(i+1, len(c.feats))), reverse=True)
    high = [(a, b, v) for _, a, b, v in pairs if abs(v) >= .9]
    c.m['feature_collinearity_train'] = {'pairs_abs_ge_0.9':len(high), 'top10':[{'a':a, 'b':b, 'pearson':r(v, 4)} for _, a, b, v in pairs[:10]]}
    yx = tr[TARGET]
    y_rel = {'mean':r(yx.mean(), 3)}
    for k in (0, 1):
        sub = yx[(tr['lights'] > 0) == bool(k)]
        y_rel['lights_on' if k else 'lights_off'] = {'rows':int(len(sub)), 'mean':r(sub.mean(), 3)}
    c.m['lights_train'] = y_rel | {'zero_share':r((tr['lights'] == 0).mean(), 4)}

    fig, ax = new_fig(1, 2, w=15, h=7.4, gridspec_kw={'width_ratios':[1, 1.35]})
    top = rows[:]
    yy = np.arange(len(top))[::-1]
    ax[0].barh(yy+.2, [z[2] for z in top], height=.38, color=ACCENT, label='Spearman')
    ax[0].barh(yy-.2, [z[1] for z in top], height=.38, color=WARN, alpha=.8, label='Pearson')
    ax[0].set_yticks(yy); ax[0].set_yticklabels([z[0] for z in top], fontsize=8)
    ax[0].axvline(0, color=MUTED, lw=.8); ax[0].legend(loc='lower right')
    ax[0].set(title='Tương quan với Appliances (train)', xlabel='hệ số'); ax[0].grid(axis='x')
    order = [z[0] for z in rows]
    idx = [c.feats.index(f) for f in order]
    im = ax[1].imshow(C[np.ix_(idx, idx)], cmap=DIV, vmin=-1, vmax=1)
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels(order, rotation=90, fontsize=7)
    ax[1].set_yticks(range(len(order))); ax[1].set_yticklabels(order, fontsize=7)
    ax[1].set_title('Tương quan giữa các đặc trưng (train)'); fig.colorbar(im, ax=ax[1], shrink=.8)

    best = rows[0]
    add_section('tuong-quan', 'Tương quan đặc trưng với mục tiêu',
                'Pearson, Spearman (theo hạng, ít nhạy với đỉnh) và Pearson với giá trị bước kế tiếp — tất cả trên train.')
    add(stats([(f2(best[2], 3), f'Spearman lớn nhất ({best[0]})'), (f2(max(abs(z[1]) for z in rows), 3), '|Pearson| lớn nhất'),
               (str(len(high)), 'cặp đặc trưng |r| ≥ 0,9'), (f'{(tr["lights"] == 0).mean()*100:.1f}%', 'mốc lights = 0')]))
    add(fig_html(fig, 'Không đặc trưng nào tương quan tuyến tính mạnh với mục tiêu; các cảm biến cùng loại (nhiệt độ với nhau, độ ẩm với nhau) tương quan cao.', 'eda_correlation'))
    add(table(['Đặc trưng', 'Pearson', 'Spearman', 'Pearson với y(t+1)'],
              [[f'<code>{esc(f)}</code>', num(p), num(s), num(l)] for f, p, s, l in rows]))
    add(table(['Cặp có tương quan cao nhất', 'Pearson'], [[f'<code>{a}</code> – <code>{b}</code>', num(v)] for _, a, b, v in pairs[:8]], sortable=False))
    add(note(f'Tương quan đơn biến với mục tiêu đều yếu (|Spearman| ≤ {f2(abs(best[2]), 3)}). Tín hiệu dự báo chủ yếu nằm ở lịch sử của chính '
             'mục tiêu và giờ trong ngày; quan hệ với cảm biến (nếu có) là phi tuyến hoặc có trễ — đây là giả thuyết để mô hình kiểm chứng.', 'warn'))
    real = [(a, b, v) for a, b, v in high if {a, b} != {'rv1', 'rv2'}]
    INSIGHTS.append(('warn', f'Không đặc trưng nào tương quan mạnh với mục tiêu (|Spearman| lớn nhất {f2(abs(best[2]), 3)} – {best[0]}). '
                             'Tín hiệu chính nằm ở lịch sử của chính mục tiêu và giờ trong ngày'+
                             (f'; cặp trùng thông tin rõ nhất: {real[0][0]}–{real[0][1]} (r = {f2(real[0][2], 3)}).' if real else '.')))


def sec_random(c):
    same = bool(np.array_equal(c.df['rv1'].to_numpy(), c.df['rv2'].to_numpy()))
    cm = c.m['correlation_train']
    c.m['random_variables'] = {'rv1_equals_rv2':same, 'rv1':cm.get('rv1'), 'rv2':cm.get('rv2')}
    add_section('bien-ngau-nhien', 'Biến ngẫu nhiên rv1, rv2', 'Kiểm tra hai biến nhiễu do tác giả dataset chủ động thêm vào.')
    add(stats([('Có' if same else 'Không', 'rv1 trùng hệt rv2'), (f2(cm['rv1']['pearson'], 4), 'Pearson rv1 – mục tiêu'),
               (f2(cm['rv1']['spearman'], 4), 'Spearman rv1 – mục tiêu')]))
    add(note(('rv1 và rv2 giống nhau từng giá trị' if same else 'rv1, rv2 khác nhau')+' và gần như không tương quan với mục tiêu. '
             'Đề xuất loại cả hai khỏi đầu vào; nếu giữ để làm đối chứng thì attention/feature importance gán cho chúng phải ≈ 0.', 'good'))
    INSIGHTS.append(('good', f'rv1, rv2 là nhiễu {"giống hệt nhau" if same else ""} (|r| với mục tiêu ≈ {f2(abs(cm["rv1"]["pearson"]), 3)}) → loại khỏi đầu vào, còn {len(c.feats)-2} đặc trưng.'))


def sec_split(c):
    df = c.df
    rows, shift = [], {}
    for k in SPLITS:
        s = df[df['split'] == k]
        rows.append([f'<b style="color:{SPLIT_COLORS[k]}">{SPLIT_NAMES[k]}</b>', (fmt(len(s)), len(s)),
                     f'{s["date"].iloc[0]:%d/%m/%Y %H:%M}', f'{s["date"].iloc[-1]:%d/%m/%Y %H:%M}',
                     (f2(len(s)/STEPS_PER_DAY, 1), len(s)), num(s[TARGET].mean(), 1), num(s[TARGET].median(), 0),
                     num(s['T_out'].mean(), 2), num(s['T2'].mean(), 2)])
    tr = c.tr
    out_range = {}
    for f in [TARGET]+c.feats:
        lo, hi = tr[f].min(), tr[f].max()
        shift[f] = {}
        for k in ('val', 'test'):
            v = df.loc[df['split'] == k, f].to_numpy(float)
            shift[f][k] = {'ks':r(ks_stat(tr[f].to_numpy(float), v), 5), 'outside_train_range_share':r(np.mean((v < lo) | (v > hi)), 5)}
        out_range[f] = shift[f]['test']['outside_train_range_share']
    norm = {f:{'mean':r(tr[f].mean(), 6), 'std':r(tr[f].std(ddof=0), 6)} for f in [TARGET]+c.feats}
    c.m['split_summary'] = {k:{'rows':int((df['split'] == k).sum()), 'target_mean':r(df.loc[df['split'] == k, TARGET].mean(), 4),
                               'target_median':r(df.loc[df['split'] == k, TARGET].median(), 4)} for k in SPLITS}
    c.m['distribution_shift'] = shift
    c.m['normalization_train'] = norm | {'log1p_target':{'mean':r(np.log1p(tr[TARGET]).mean(), 6), 'std':r(np.log1p(tr[TARGET]).std(ddof=0), 6)}}

    ks_sorted = sorted(((shift[f]['test']['ks'], f) for f in shift), reverse=True)
    top = ks_sorted[:12]
    fig, ax = new_fig(1, 2, w=15, h=4.8)
    yy = np.arange(len(top))[::-1]
    ax[0].barh(yy+.2, [shift[f]['val']['ks'] for _, f in top], height=.38, color=WARN, label='train vs val')
    ax[0].barh(yy-.2, [shift[f]['test']['ks'] for _, f in top], height=.38, color=BAD, label='train vs test')
    ax[0].set_yticks(yy); ax[0].set_yticklabels([f for _, f in top], fontsize=8)
    ax[0].set(title='KS lớn nhất so với train', xlabel='thống kê KS'); ax[0].legend(loc='lower right'); ax[0].grid(axis='x')
    bins = np.arange(0, 400, 10)
    for k in SPLITS:
        ax[1].hist(df.loc[df['split'] == k, TARGET].clip(upper=bins[-1]), bins=bins, density=True, histtype='step', lw=1.8, color=SPLIT_COLORS[k], label=SPLIT_NAMES[k])
    ax[1].set(title='Appliances theo tập (cắt ở 400 Wh)', xlabel='Wh / 10 phút', ylabel='mật độ'); ax[1].legend()

    worst = ks_sorted[0]; tks = shift[TARGET]
    n_out = sum(1 for f in c.feats if out_range[f] > .05)
    add_section('split', 'Split theo thời gian và độ lệch phân phối',
                f'Chia theo thứ tự thời gian, không xáo: {c.args.train_ratio:.0%} / {c.args.val_ratio:.0%} / {1-c.args.train_ratio-c.args.val_ratio:.0%}. '
                'Validation và test là tương lai so với train, giống điều kiện triển khai.')
    add(table(['Tập', 'Số mốc', 'Bắt đầu', 'Kết thúc', 'Số ngày', 'Mean Wh', 'Trung vị Wh', 'Mean T_out', 'Mean T2'], rows, sortable=False))
    add(fig_html(fig, 'Đặc trưng nhiệt độ lệch mạnh vì train là mùa đông, validation/test là mùa xuân; phân phối mục tiêu lệch ít hơn.', 'eda_split_shift'))
    add(table(['Đặc trưng', 'KS val', 'KS test', '% test ngoài [min,max] train'],
              [[f'<code>{f}</code>', num(shift[f]['val']['ks']), num(shift[f]['test']['ks']), num(shift[f]['test']['outside_train_range_share']*100, 1)]
               for _, f in ks_sorted]))
    add(note(f'Lệch lớn nhất: <code>{worst[1]}</code> (KS test {f2(worst[0], 3)}); {n_out} đặc trưng có hơn 5% giá trị test nằm ngoài khoảng train. '
             f'Mục tiêu lệch ít hơn (KS val {f2(tks["val"]["ks"], 3)}, test {f2(tks["test"]["ks"], 3)}). Mô hình dựa nhiều vào nhiệt độ tuyệt đối có nguy cơ '
             'ngoại suy kém; cần đối chiếu sai số validation khi bỏ/giữ nhóm nhiệt độ.', 'warn'))
    add(note('Chuẩn hoá (mean/std) chỉ fit trên train, lưu tại <code>metrics.normalization_train</code>. Một mẫu thuộc tập chứa <b>mốc mục tiêu</b> của nó; '
             'cửa sổ đầu vào chỉ gồm các mốc trước mục tiêu nên được phép lấy lịch sử từ tập liền trước (quá khứ, không rò rỉ). '
             'Không mẫu train nào có mục tiêu nằm trong validation/test.'))
    INSIGHTS.append(('warn', f'Lệch mùa: train {c.t.iloc[0]:%d/%m}–{c.t.iloc[c.bounds["train"][1]-1]:%d/%m}, test {c.t.iloc[c.bounds["test"][0]]:%d/%m}–{c.t.iloc[-1]:%d/%m}; '
                             f'KS lớn nhất {f2(worst[0], 3)} ({worst[1]}), {n_out} đặc trưng có >5% giá trị test ngoài khoảng train.'))


def sec_baseline(c):
    y = c.y; a, b = c.bounds['val']
    idx = np.arange(a, b)
    yv = y[idx]
    tr = c.tr
    hour_mean = tr.groupby(tr['date'].dt.hour)[TARGET].mean()
    preds = {
        'Persistence: ŷ(t+1) = y(t)':y[idx-1],
        'Seasonal naive: ŷ(t+1) = y(t+1−144)':y[idx-STEPS_PER_DAY],
        'Trung bình theo giờ (fit trên train)':hour_mean.reindex(c.t.iloc[idx].dt.hour).to_numpy(),
        'Trung bình train (hằng số)':np.full(len(idx), tr[TARGET].mean()),
        'Trung vị train (hằng số)':np.full(len(idx), tr[TARGET].median()),
    }
    res = {k:regress_metrics(yv, p) for k, p in preds.items()}
    c.m['baselines_validation'] = {'horizon':'1 step (10 min)', 'unit':'Wh', 'results':res}
    best = min(res, key=lambda k: res[k]['rmse'])
    fig, ax = new_fig(1, 2, w=15, h=4.2, gridspec_kw={'width_ratios':[1, 1.6]})
    names = list(res)[::-1]
    ax[0].barh(range(len(names)), [res[n]['rmse'] for n in names], color=ACCENT, height=.55, label='RMSE')
    ax[0].barh(range(len(names)), [res[n]['mae'] for n in names], color=WARN, height=.3, label='MAE')
    ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels([n.split(':')[0] for n in names], fontsize=8)
    ax[0].set(title='Baseline trên validation', xlabel='Wh'); ax[0].legend(loc='lower right'); ax[0].grid(axis='x')
    show = slice(0, 3*STEPS_PER_DAY)
    ax[1].plot(c.t.iloc[idx][show], yv[show], color=INK, lw=1, label='thực tế')
    ax[1].plot(c.t.iloc[idx][show], preds['Persistence: ŷ(t+1) = y(t)'][show], color=ACCENT, lw=1, alpha=.85, label='persistence')
    ax[1].plot(c.t.iloc[idx][show], preds['Trung bình theo giờ (fit trên train)'][show], color=WARN, lw=1.4, label='TB theo giờ')
    ax[1].set(title='3 ngày đầu validation', ylabel='Wh / 10 phút'); ax[1].legend(ncol=3, loc='upper left'); ax[1].grid(axis='y')
    add_section('baseline', 'Baseline tham chiếu trên validation',
                'Dự báo 1 bước (10 phút) không dùng học máy. Đây là mốc sàn để đặt ngưỡng cho LSTM và Transformer, không phải kết quả mô hình.')
    add(table(['Baseline', 'RMSE (Wh)', 'MAE (Wh)', 'R²'],
              [[esc(k), num(v['rmse'], 2), num(v['mae'], 2), num(v['r2'], 4)] for k, v in res.items()]))
    add(fig_html(fig, 'Persistence theo sát chuỗi nhưng luôn trễ một bước ở các đỉnh; trung bình theo giờ chỉ nắm được mức nền.', 'eda_baseline'))
    p = res['Persistence: ŷ(t+1) = y(t)']
    add(note(f'Baseline tốt nhất theo RMSE: <b>{esc(best)}</b>. Persistence đạt RMSE {f2(p["rmse"], 2)} Wh, MAE {f2(p["mae"], 2)} Wh, R² {f2(p["r2"], 3)}. '
             'Mô hình học sâu phải vượt persistence mới có giá trị; MAE và RMSE nên báo cáo cùng nhau vì đỉnh làm RMSE tăng mạnh.'))
    INSIGHTS.append(('info', f'Baseline persistence trên validation: RMSE {f2(p["rmse"], 2)} Wh, MAE {f2(p["mae"], 2)} Wh, R² {f2(p["r2"], 3)} — mốc sàn cho LSTM/Transformer.'))


def sec_input(c):
    keep = [TARGET]+[f for f in c.feats if f not in ('rv1', 'rv2')]
    n_in = len(keep)+2
    c.m['model_input_suggestion'] = {'input_columns':keep, 'features_dropped':['rv1', 'rv2'], 'n_numeric_columns':len(keep),
        'time_encodings':['hour_sin', 'hour_cos'], 'n_input_channels':n_in, 'target_transform_candidate':'log1p',
        'window_candidates_steps':[36, 72, 144], 'horizon_steps':1, 'status':'đề xuất từ EDA, chưa kiểm chứng bằng mô hình'}
    add_section('dau-vao', 'Gợi ý đầu vào cho mô hình chuỗi thời gian',
                'Tổng hợp từ các mục trên. Đây là đề xuất để tài liệu model/training quyết định, chưa phải cấu hình đã kiểm chứng.')
    add(table(['Thành phần', 'Đề xuất', 'Căn cứ EDA'], [
        ['Cột số', f'{len(keep)} cột: lịch sử <code>Appliances</code> + 25 đặc trưng; bỏ <code>rv1</code>, <code>rv2</code>', 'ACF lag 1 cao; rv1/rv2 là nhiễu'],
        ['Mã hoá thời gian', 'sin/cos giờ trong ngày', 'Chu kỳ ngày rõ; chu kỳ tuần yếu nên không thêm thứ'],
        ['Mục tiêu', 'Thử <code>log1p</code>, metric luôn tính lại trên Wh', 'Skewness cao, đỉnh chi phối MSE'],
        ['Cửa sổ', '36 / 72 / 144 bước (6 / 12 / 24 giờ)', 'ACF giảm nhanh nhưng có chu kỳ 24 h'],
        ['Chuẩn hoá', 'Mean/std fit trên train', 'Tránh rò rỉ; lệch mùa giữa các tập'],
        ['Tensor', f'<code>x: [B, W, {n_in}]</code>, <code>y: [B, 1]</code>', 'Dự báo 1 bước'],
    ], sortable=False))


def build_summary():
    SECTIONS.insert(1, {'id':'tom-tat', 'title':'Phát hiện và lưu ý chính',
                        'lead':'Các chỉ số dưới đây đều xuất phát từ dữ liệu vừa phân tích. Không chứa kết quả huấn luyện.',
                        'body':['<ul class="insights">'+''.join(f'<li class="ins {k}">{v}</li>' for k, v in INSIGHTS)+'</ul>']})


# CSS/JS nhúng trực tiếp, kế thừa layout của LAB2/EDA.py (không import).
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
main{padding:0 clamp(16px,4vw,56px) 96px;min-width:0}
.hero{padding:64px 0 40px}
.hero h1{font:600 clamp(2.1rem,4.6vw,3.5rem)/1.08 var(--serif);margin:0 0 18px;letter-spacing:-.01em;max-width:20ch}
.hero .sub{color:var(--mut);font-size:1.08rem;max-width:62ch;margin:0 0 34px}
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
.fig img{display:block;width:100%;height:auto;cursor:zoom-in;border-radius:6px;max-height:950px;object-fit:contain}
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
.insights{list-style:none;margin:8px 0 0;padding:0;display:grid;gap:12px;max-width:86ch}
.ins{padding:14px 18px 14px 20px;border-left:3px solid var(--accent);background:var(--panel);border-radius:0 8px 8px 0}
.ins.good{border-color:var(--good)}.ins.warn{border-color:var(--warn)}
#lb{border:0;padding:0;background:transparent;max-width:96vw;max-height:96vh}
#lb::backdrop{background:rgba(5,9,19,.88)}
#lb img{max-width:96vw;max-height:94vh;display:block;border-radius:8px;cursor:zoom-out}
footer{border-top:1px dashed var(--stitch);margin-top:64px;padding-top:22px;color:var(--mut);font-size:.85rem}
html[data-theme="light"]{color-scheme:light;--bg:#f4f6fa;--panel:#fff;--panel2:#eaf0f9;
 --line:#d6dfeb;--stitch:#c4d1e3;--ink:#1d2b43;--mut:#52647e;--accent:#255ac7;--good:#147750;--warn:#9b6715;--bad:#bb433e}
.hero-actions{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:25px}
.theme-toggle{font:inherit;font-size:.86rem;padding:7px 14px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;cursor:pointer}
.theme-toggle:hover{border-color:var(--accent)}
.hero-label{display:inline-block;font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:700;margin-bottom:10px}
.footnote{font-size:.78rem;color:var(--mut)}
@media (max-width:1000px){
 .shell{grid-template-columns:1fr}
 nav.side{position:sticky;top:0;z-index:5;height:auto;display:flex;gap:4px;overflow-x:auto;padding:10px 16px;border-right:0;border-bottom:1px dashed var(--stitch);background:var(--bg)}
 nav.side strong{display:none}
 nav.side a{white-space:nowrap;margin:0}
}
@media(max-width:650px){.hero{padding:38px 0 22px}.stats{grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.stat b{font-size:1.45rem}
 th,td{padding:7px 9px;font-size:.82rem}}
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
$$('table.sortable').forEach(t=>{const body=t.tBodies[0],ths=$$('th',t);
 ths.forEach((th,i)=>{th.tabIndex=0;th.setAttribute('role','button');
  const go=()=>{const asc=th.dataset.dir!=='asc';ths.forEach(x=>delete x.dataset.dir);th.dataset.dir=asc?'asc':'desc';
   const val=r=>{const c=r.cells[i],v=c.dataset.v??c.textContent,n=parseFloat(v);return isNaN(n)?v.toLowerCase():n};
   [...body.rows].sort((a,b)=>{const x=val(a),y=val(b);return(x>y?1:x<y?-1:0)*(asc?1:-1)}).forEach(r=>body.appendChild(r))};
  th.addEventListener('click',go);th.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();go()}})})});
const theme=document.getElementById('theme-toggle');
if(theme){theme.addEventListener('click',()=>{
 const next=document.documentElement.dataset.theme==='light'?'dark':'light';
 document.documentElement.dataset.theme=next;
 theme.textContent=next==='light'?'☾ Chế độ tối':'☀ Chế độ sáng';});}
})();
"""


def render_html(c, elapsed, created_at):
    nav = ''.join(f'<a href="#{s["id"]}">{esc(s["title"])}</a>' for s in SECTIONS)
    body = ''.join('<section id="'+s['id']+'"><h2>'+esc(s['title'])+'</h2>'+(
        '<p class="lead">'+s['lead']+'</p>' if s['lead'] else '')+''.join(s['body'])+'</section>' for s in SECTIONS)
    meta = (f'<span>Tạo lúc {created_at:%d/%m/%Y %H:%M}</span><span>Phân tích trong {elapsed:.0f} giây</span>'
            f'<span>pandas {pd.__version__}, numpy {np.__version__}, matplotlib {matplotlib.__version__}</span>')
    i = c.m['integrity']
    return f'''<!DOCTYPE html><html lang="vi" data-theme="dark"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EDA Appliances Energy</title>
<style>{CSS}</style></head><body><div class="shell">
<nav class="side" aria-label="Mục lục"><strong>EDA / Appliances Energy</strong>{nav}</nav>
<main><header class="hero">
<span class="hero-label">Coursework · Topic 8 · Multivariate time-series regression</span>
<h1>Khám phá dữ liệu UCI Appliances Energy</h1>
<p class="sub">{fmt(i["rows"])} mốc 10 phút từ {i["start"][:10]} đến {i["end"][:10]}: điện năng thiết bị gia dụng,
 nhiệt độ/độ ẩm 9 vị trí trong nhà và 6 biến thời tiết. Mỗi số liệu đều tính từ dữ liệu, không phải kết quả huấn luyện.</p>
<div class="hero-actions"><button id="theme-toggle" class="theme-toggle" type="button">☀ Chế độ sáng</button>
<span class="footnote">Bấm biểu đồ để phóng to · bảng có thể sắp xếp</span></div>
<p class="meta">{meta}</p></header>{body}
<footer>Báo cáo tạo bởi Coursework/EDA.py · thống kê ở reports/eda_stats.json · split tại data/splits.json.
 Mở offline, không dùng CDN. Tập test chỉ dùng để mô tả, không dùng chọn mô hình.</footer>
</main></div><dialog id="lb"><img alt="Biểu đồ phóng to"></dialog>
<script>{JS}</script></body></html>'''


def build_json(c, elapsed, created_at):
    return {'dataset':'UCI Appliances Energy Prediction', 'purpose':'Coursework Topic 8 EDA',
            'generated_at':created_at.isoformat(timespec='seconds'), 'elapsed_seconds':round(elapsed, 2),
            'target':TARGET, 'features':c.feats, 'split_file':rel(c.args.split_file), 'report_file':rel(c.args.out),
            'metrics':c.m,
            'notes':['correlation, seasonality, ACF, normalization fit only on train rows',
                     'baselines evaluated on validation only; test used only for descriptive shift statistics',
                     'model_input_suggestion is a proposal derived from EDA, not a validated configuration']}


def main(argv=None):
    global ASSETS, _T0
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='EDA UCI Appliances Energy -> HTML tự chứa + JSON + split')
    ap.add_argument('--csv', type=Path, default=Path('data/raw/energydata_complete.csv'))
    ap.add_argument('--out', type=Path, default=Path('reports/eda_report.html'))
    ap.add_argument('--stats', type=Path, default=Path('reports/eda_stats.json'))
    ap.add_argument('--split-file', type=Path, default=Path('data/splits.json'))
    ap.add_argument('--assets-dir', type=str, default='docs/assets', help='chuỗi rỗng để không xuất PNG')
    ap.add_argument('--train-ratio', type=float, default=.70)
    ap.add_argument('--val-ratio', type=float, default=.15)
    args = ap.parse_args(argv)
    for name in ('csv', 'out', 'stats', 'split_file'):
        setattr(args, name, resolve_path(getattr(args, name)))
    if args.stats == args.out:
        ap.error('--stats và --out phải là hai file khác nhau')
    ASSETS = resolve_path(args.assets_dir) if args.assets_dir else None
    _T0 = time.perf_counter(); SECTIONS.clear(); INSIGHTS.clear()
    c = prepare(args)
    save_split(c)
    for title, fn in [('Tổng quan', sec_overview), ('Mục tiêu', sec_target), ('Thời gian', sec_timeline),
                      ('Chu kỳ', sec_seasonality), ('ACF/PACF', sec_acf), ('Tương quan', sec_correlation),
                      ('rv1/rv2', sec_random), ('Split & shift', sec_split), ('Baseline', sec_baseline), ('Đầu vào', sec_input)]:
        log(title); fn(c)
    build_summary()
    created_at = datetime.now(); elapsed = time.perf_counter()-_T0
    args.out.parent.mkdir(parents=True, exist_ok=True); args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(c, elapsed, created_at), encoding='utf-8')
    args.stats.write_text(json.dumps(build_json(c, elapsed, created_at), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    log(f'Đã lưu HTML: {args.out} ({args.out.stat().st_size/1e6:.2f} MB)')
    log(f'Đã lưu JSON: {args.stats}')
    log(f'Đã lưu split: {args.split_file}')
    return args.out, args.stats


if __name__ == '__main__':
    try: main()
    except (ValueError, RuntimeError, MemoryError) as e:
        print(f'[ERROR] {e}', file=sys.stderr)
        raise SystemExit(1) from e
