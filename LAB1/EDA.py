#!/usr/bin/env python3
"""
EDA.py - Exploratory Data Analysis cho Fashion-MNIST (dữ liệu ảnh).

Sinh ra MỘT file HTML tự chứa (ảnh nhúng base64, không cần internet để xem):

    python EDA.py                          # -> ./reports/eda_report.html
    python EDA.py --out eda.html           # đổi nơi lưu
    python EDA.py --tsne-samples 0         # bỏ t-SNE cho nhanh
    python EDA.py --knn-test 3000          # kNN baseline chỉ chạy 3000 ảnh test

Chỉ cần numpy + matplotlib (+ Pillow đi kèm matplotlib).
sklearn là tuỳ chọn (cho t-SNE); torch là tuỳ chọn (để tái tạo ĐÚNG phép chia
train/val 50k/10k trong Train.py).
Dữ liệu đọc từ ./data/FashionMNIST/raw (cùng chỗ torchvision tải về); nếu chưa
có thì tự tải từ GitHub mirror của Zalando.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import html
import io
import re
import struct
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# --------------------------------------------------------------------------- #
# Hằng số & giao diện
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
K, H, W = 10, 28, 28
MIRROR = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {"xtr": "train-images-idx3-ubyte", "ytr": "train-labels-idx1-ubyte",
         "xte": "t10k-images-idx3-ubyte", "yte": "t10k-labels-idx1-ubyte"}

# Bảng màu lấy từ thuốc nhuộm vải (indigo, nghệ, quế, ...). PANEL phải trùng với CSS.
PANEL, LINE, INK, MUTED, BGDARK = "#141f36", "#2a3858", "#e8ecf6", "#98a4c2", "#0b1220"
CLASS_COLORS = ["#7aa2ff", "#f2b84b", "#ef6f6c", "#43c6b0", "#c58af9",
                "#9ccc65", "#ff9a76", "#4fc3f7", "#f48fb1", "#d7c49e"]

DIV = LinearSegmentedColormap.from_list("dye_div", ["#5cc8ff", "#1b2a4a", "#ff7a6b"])  # xanh - tối - đỏ

plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
    "axes.titlepad": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": LINE, "grid.alpha": 0.55, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 10, "legend.frameon": False, "legend.labelcolor": INK,
    "axes.prop_cycle": matplotlib.cycler(color=CLASS_COLORS),
    "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

FEATS = [("brightness", "Độ sáng trung bình"),
         ("contrast", "Độ tương phản (std pixel)"),
         ("coverage", "Độ phủ vật thể (tỉ lệ pixel > 0.1)"),
         ("edge", "Mật độ cạnh (gradient TB)"),
         ("bbox_h", "Chiều cao vật thể (px)"),
         ("bbox_w", "Chiều rộng vật thể (px)"),
         ("aspect", "Tỉ lệ rộng / cao"),
         ("symmetry", "Đối xứng trái-phải (1 = đối xứng hoàn hảo)")]

SECTIONS: list[dict] = []
INSIGHTS: list[tuple[str, str]] = []
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Nạp dữ liệu
# --------------------------------------------------------------------------- #
def read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_fashion_mnist(data_dir: Path):
    raw = data_dir / "FashionMNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, stem in FILES.items():
        path = next((p for p in (raw / stem, raw / f"{stem}.gz") if p.exists()), None)
        if path is None:
            path = raw / f"{stem}.gz"
            log(f"Đang tải {stem}.gz từ GitHub mirror ...")
            try:
                urllib.request.urlretrieve(MIRROR + f"{stem}.gz", path)
            except Exception as exc:  # noqa: BLE001
                path.unlink(missing_ok=True)
                sys.exit(f"Không tải được dữ liệu ({exc}).\n"
                         "Hãy chạy Train.py một lần (torchvision sẽ tải vào ./data) "
                         "hoặc trỏ --data-dir tới thư mục chứa FashionMNIST/raw.")
        out[key] = read_idx(path)
    return out["xtr"], out["ytr"].astype(np.int64), out["xte"], out["yte"].astype(np.int64)


def val_split(n_total=60000, n_fit=50000, seed=1):
    """Tái tạo phép chia train/val của Train.py (torch.randperm, Generator seed=1)."""
    try:
        import torch
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=g).numpy()
        exact = True
    except Exception:  # noqa: BLE001
        perm = np.random.default_rng(seed).permutation(n_total)
        exact = False
    return perm[:n_fit], perm[n_fit:], exact


# --------------------------------------------------------------------------- #
# Tiện ích HTML
# --------------------------------------------------------------------------- #
def esc(s) -> str:
    return html.escape(str(s))


def fmt(n) -> str:
    return f"{int(n):,}"


def add_section(sid: str, title: str, lead: str = "") -> dict:
    sec = {"id": sid, "title": title, "lead": lead, "body": []}
    SECTIONS.append(sec)
    return sec


def add(snippet: str) -> None:
    SECTIONS[-1]["body"].append(snippet)


def h3(text: str) -> str:
    return f"<h3>{text}</h3>"


def para(text: str) -> str:
    return f"<p>{text}</p>"


def note(text: str, kind: str = "info") -> str:
    return f'<div class="note {kind}">{text}</div>'


def stats(items) -> str:
    cells = []
    for it in items:
        value, label = it[0], it[1]
        sub = f"<small>{it[2]}</small>" if len(it) > 2 else ""
        cells.append(f'<div class="stat"><b>{value}</b><span>{label}</span>{sub}</div>')
    return f'<div class="stats">{"".join(cells)}</div>'


def cname(k: int) -> str:
    return f'<span class="dot" style="background:{CLASS_COLORS[k]}"></span>{esc(CLASSES[k])}'


def table(headers, rows, sortable=True) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = []
        for cell in r:
            if isinstance(cell, tuple):
                tds.append(f'<td class="num" data-v="{cell[1]}">{cell[0]}</td>')
            else:
                tds.append(f"<td>{cell}</td>")
        trs.append(f'<tr>{"".join(tds)}</tr>')
    cls = "sortable" if sortable else ""
    return (f'<div class="tbl-wrap"><table class="{cls}"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')


def fig_b64(fig, dpi=140) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_html(fig, caption="", dpi=140) -> str:
    alt = esc(re.sub(r"<[^>]+>", "", caption))[:240]
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (f'<figure class="fig"><img src="data:image/png;base64,{fig_b64(fig, dpi)}" '
            f'alt="{alt}" title="Bấm để phóng to">{cap}</figure>')


def png_b64(arr_u8: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr_u8, dtype=np.uint8)).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def px_img(b64: str, alt="", size=56) -> str:
    return (f'<img class="px" width="{size}" height="{size}" alt="{esc(alt)}" '
            f'src="data:image/png;base64,{b64}">')


# --------------------------------------------------------------------------- #
# Tiện ích vẽ
# --------------------------------------------------------------------------- #
def new_fig(nrows=1, ncols=1, w=None, h=None, **kw):
    return plt.subplots(nrows, ncols, figsize=(w or 5.6 * ncols, h or 4.2 * nrows),
                        constrained_layout=True, **kw)


def class_handles():
    return [Line2D([0], [0], marker="o", ls="", color=CLASS_COLORS[k], label=CLASSES[k], markersize=7)
            for k in range(K)]


def bottom_legend(fig, ncol=5, y=-0.02):
    fig.legend(handles=class_handles(), loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y),
               columnspacing=1.6, handletextpad=0.3)


def hide_frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def hbox(ax, groups, whis=1.5):
    """Boxplot ngang tự vẽ (không phụ thuộc phiên bản matplotlib), mỗi lớp một màu.
    Không đảo trục y ở đây (trục có thể được chia sẻ); nơi gọi tự invert_yaxis() một lần."""
    lo_all, hi_all = np.inf, -np.inf
    for i, g in enumerate(groups):
        col = CLASS_COLORS[i]
        q1, med, q3 = np.percentile(g, [25, 50, 75])
        iqr = q3 - q1
        lo, hi = g[g >= q1 - whis * iqr].min(), g[g <= q3 + whis * iqr].max()
        lo_all, hi_all = min(lo_all, lo, g.mean()), max(hi_all, hi, g.mean())
        ax.hlines(i, lo, hi, color=col, lw=1.4, alpha=0.85)
        ax.barh(i, max(q3 - q1, 1e-9), left=q1, height=0.62, color=col, alpha=0.8, edgecolor="none")
        ax.plot([med, med], [i - 0.31, i + 0.31], color=PANEL, lw=2.2)
        ax.scatter(g.mean(), i, marker="D", s=14, color=INK, zorder=4)
    pad = (hi_all - lo_all) * 0.05 or 0.5
    ax.set_xlim(lo_all - pad, hi_all + pad)
    ax.set_yticks(range(len(groups)))
    ax.grid(axis="x")
    ax.set_axisbelow(True)


def annotate_heat(ax, im, M, fmtstr="{:.2f}", size=8):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            r, g, b, _ = im.cmap(im.norm(M[i, j]))
            col = "#0b1220" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "#ffffff"
            ax.text(j, i, fmtstr.format(M[i, j]), ha="center", va="center", color=col, fontsize=size)


def bar_labels(ax, bars, fmtstr="{:,.0f}", size=8, dx=0.0, horizontal=False):
    for b in bars:
        if horizontal:
            ax.text(b.get_width() + dx, b.get_y() + b.get_height() / 2, fmtstr.format(b.get_width()),
                    va="center", fontsize=size, color=INK)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dx, fmtstr.format(b.get_height()),
                    ha="center", va="bottom", fontsize=size, color=INK)


# --------------------------------------------------------------------------- #
# Chuẩn bị dữ liệu & đặc trưng
# --------------------------------------------------------------------------- #
def image_features(X: np.ndarray) -> dict[str, np.ndarray]:
    """Các đặc trưng thống kê cho mỗi ảnh. X: (N, 784) float32 trong [0, 1]."""
    im = X.reshape(-1, H, W)
    f = {"brightness": im.mean(axis=(1, 2)), "contrast": im.std(axis=(1, 2))}
    mask = im > 0.1
    f["coverage"] = mask.mean(axis=(1, 2))
    f["edge"] = np.abs(np.diff(im, axis=2)).mean(axis=(1, 2)) + np.abs(np.diff(im, axis=1)).mean(axis=(1, 2))
    rows, cols = mask.any(axis=2), mask.any(axis=1)
    empty = ~rows.any(axis=1)
    top, bot = rows.argmax(1), H - 1 - rows[:, ::-1].argmax(1)
    left, right = cols.argmax(1), W - 1 - cols[:, ::-1].argmax(1)
    h = np.where(empty, 0, bot - top + 1)
    w = np.where(empty, 0, right - left + 1)
    f["bbox_h"], f["bbox_w"] = h.astype(np.float32), w.astype(np.float32)
    f["aspect"] = np.where(h > 0, w / np.maximum(h, 1), 0).astype(np.float32)
    tot = im.sum(axis=(1, 2)) + 1e-6
    f["cy"] = (im.sum(axis=2) * np.arange(H)).sum(1) / tot
    f["cx"] = (im.sum(axis=1) * np.arange(W)).sum(1) / tot
    f["symmetry"] = 1 - np.abs(im - im[:, :, ::-1]).sum(axis=(1, 2)) / (2 * tot)
    return {k: v.astype(np.float32) for k, v in f.items()}


def prepare(args) -> SimpleNamespace:
    xtr, ytr, xte, yte = load_fashion_mnist(args.data_dir)
    c = SimpleNamespace(xtr_u8=xtr, ytr=ytr, xte_u8=xte, yte=yte, m={}, args=args)
    c.Xtr = xtr.reshape(len(xtr), -1).astype(np.float32) / 255.0
    c.Xte = xte.reshape(len(xte), -1).astype(np.float32) / 255.0
    c.rng = np.random.default_rng(args.seed)
    c.idx = [np.flatnonzero(ytr == k) for k in range(K)]
    c.idx_te = [np.flatnonzero(yte == k) for k in range(K)]
    c.mu = c.Xtr.mean(axis=0, dtype=np.float64).astype(np.float32)
    c.mus = np.stack([c.Xtr[i].mean(axis=0) for i in c.idx])
    log("Tính đặc trưng cấp ảnh ...")
    c.F = image_features(c.Xtr)
    c.Fte = image_features(c.Xte)
    return c


def class_mean(c, key):
    return np.array([c.F[key][i].mean() for i in c.idx])


# --------------------------------------------------------------------------- #
# 1. Tổng quan + kiểm tra toàn vẹn
# --------------------------------------------------------------------------- #
def sec_overview(c):
    add_section("tong-quan", "Tổng quan bộ dữ liệu",
                "Fashion-MNIST là ảnh xám 28×28 của 10 nhóm sản phẩm thời trang do Zalando công bố, "
                "dùng thay MNIST vì khó hơn. Phần này kiểm tra cấu trúc và độ sạch của dữ liệu trước khi đi sâu.")
    mem = (c.xtr_u8.nbytes + c.xte_u8.nbytes) / 1e6
    add(stats([(fmt(len(c.ytr)), "ảnh train"), (fmt(len(c.yte)), "ảnh test"),
               ("28 × 28 × 1", "kích thước ảnh (kênh xám)"), ("10", "số lớp"),
               (f"{c.xtr_u8.min()}–{c.xtr_u8.max()}", "dải pixel (uint8)"),
               (f"{mem:.1f} MB", "dung lượng thô")]))

    def md5s(X):
        return [hashlib.md5(x.tobytes()).digest() for x in X.reshape(len(X), -1)]

    htr, hte = md5s(c.xtr_u8), md5s(c.xte_u8)
    dup_tr = len(htr) - len(set(htr))
    dup_te = len(hte) - len(set(hte))
    overlap = len(set(htr) & set(hte))
    seen, conflict = {}, 0
    for hh, y in zip(htr, c.ytr):
        if hh in seen and seen[hh] != y:
            conflict += 1
        seen.setdefault(hh, y)
    empty_tr = int((c.xtr_u8.reshape(len(c.ytr), -1).max(axis=1) == 0).sum())
    empty_te = int((c.xte_u8.reshape(len(c.yte), -1).max(axis=1) == 0).sum())
    never = int((c.Xtr.max(axis=0) == 0).sum())
    c.m.update(dup_tr=dup_tr, dup_te=dup_te, overlap=overlap, conflict=conflict, never=never)

    ok, bad = '<span class="tag good">ổn</span>', '<span class="tag warn">cần xem</span>'
    rows = [
        ["Giá trị thiếu (NaN)", "0 (dữ liệu uint8, không thể NaN)", ok],
        ["Ảnh trùng lặp trong tập train", f"{fmt(dup_tr)} ảnh", ok if dup_tr == 0 else bad],
        ["Ảnh trùng lặp trong tập test", f"{fmt(dup_te)} ảnh", ok if dup_te == 0 else bad],
        ["Ảnh xuất hiện ở cả train và test (rò rỉ dữ liệu)", f"{fmt(overlap)} ảnh", ok if overlap == 0 else bad],
        ["Cùng một ảnh nhưng khác nhãn (train)", f"{fmt(conflict)} trường hợp", ok if conflict == 0 else bad],
        ["Ảnh hoàn toàn đen", f"train {fmt(empty_tr)}, test {fmt(empty_te)}", ok if empty_tr + empty_te == 0 else bad],
        ["Vị trí pixel luôn bằng 0 trên toàn tập train", f"{never} / 784 pixel", ok],
    ]
    add(h3("Kiểm tra toàn vẹn"))
    add(table(["Hạng mục", "Kết quả", "Đánh giá"], rows, sortable=False))
    add(para("Các ảnh trùng lặp được xác định bằng mã băm MD5 của 784 byte pixel, nên chỉ bắt được ảnh "
             "giống hệt từng pixel, không bắt được ảnh gần giống."))


# --------------------------------------------------------------------------- #
# 2. Phân phối lớp
# --------------------------------------------------------------------------- #
def sec_classes(c):
    add_section("phan-phoi-lop", "Phân phối các lớp",
                "Mất cân bằng lớp ảnh hưởng trực tiếp đến cách chọn metric (accuracy hay F1) và cách chia train/val. "
                "Ngoài tập train/test gốc, mình kiểm tra luôn phép chia 50k/10k trong Train.py.")
    cnt_tr = np.bincount(c.ytr, minlength=K)
    cnt_te = np.bincount(c.yte, minlength=K)
    fit_idx, val_idx, exact = val_split()
    cnt_fit = np.bincount(c.ytr[fit_idx], minlength=K)
    cnt_val = np.bincount(c.ytr[val_idx], minlength=K)
    ratio = cnt_tr.max() / cnt_tr.min()
    c.m.update(ratio=ratio, val_exact=exact)

    fig, axes = new_fig(1, 3, w=18, h=4.6, gridspec_kw={"width_ratios": [1, 1, 1.15]})
    x = np.arange(K)
    for ax, cnt, title in ((axes[0], cnt_tr, "Train gốc (60.000 ảnh)"), (axes[1], cnt_te, "Test (10.000 ảnh)")):
        bars = ax.bar(x, cnt, color=CLASS_COLORS, width=0.72)
        bar_labels(ax, bars, dx=cnt.max() * 0.01)
        ax.set_ylim(0, cnt.max() * 1.15)
        ax.set_xticks(x, [str(k) for k in x])
        ax.set_xlabel("chỉ số lớp")
        ax.set_title(title)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    ax = axes[2]
    wdt = 0.26
    for j, (cnt, lab, col) in enumerate(((cnt_fit, f"Train fit ({fmt(len(fit_idx))})", "#7aa2ff"),
                                         (cnt_val, f"Validation ({fmt(len(val_idx))})", "#f2b84b"),
                                         (cnt_te, "Test (10.000)", "#43c6b0"))):
        ax.bar(x + (j - 1) * wdt, cnt / cnt.sum() * 100 - 10, width=wdt, label=lab, color=col)
    ax.axhline(0, color=INK, lw=1, alpha=0.7)
    lim = max(np.abs(cnt_val / cnt_val.sum() * 100 - 10).max(), np.abs(cnt_fit / cnt_fit.sum() * 100 - 10).max()) * 1.6
    ax.set_ylim(-lim, lim)
    ax.set_xticks(x, [str(k) for k in x])
    ax.set_xlabel("chỉ số lớp")
    ax.set_ylabel("lệch so với 10% (điểm phần trăm)")
    ax.set_title("Độ lệch tỉ lệ lớp so với mức đều (10%)")
    ax.legend(ncol=3, fontsize=8, loc="upper center")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    cap = ("Chỉ số lớp: " + ", ".join(f"{k} = {CLASSES[k]}" for k in range(K)) + ". ")
    cap += ("Phép chia 50k/10k được tái tạo chính xác bằng torch.randperm (seed = 1)."
            if exact else "Không có torch trong môi trường này nên phép chia 50k/10k chỉ là xấp xỉ (cùng seed nhưng khác bộ sinh số).")
    add(fig_html(fig, cap))

    rows = [[cname(k), (fmt(cnt_tr[k]), cnt_tr[k]), (f"{cnt_tr[k] / cnt_tr.sum() * 100:.2f}%", cnt_tr[k]),
             (fmt(cnt_te[k]), cnt_te[k]), (fmt(cnt_fit[k]), cnt_fit[k]), (fmt(cnt_val[k]), cnt_val[k]),
             (f"{cnt_val[k] / cnt_val.sum() * 100:.2f}%", cnt_val[k])] for k in range(K)]
    add(table(["Lớp", "Train", "% train", "Test", "Train fit", "Validation", "% validation"], rows))
    spread = (cnt_val / cnt_val.sum() * 100).max() - (cnt_val / cnt_val.sum() * 100).min()
    c.m["val_spread"] = spread
    add(note(f"Tỉ lệ lớp lớn nhất / nhỏ nhất ở tập train là <b>{ratio:.2f}</b>, tức dữ liệu cân bằng. "
             f"Trong tập validation 10k, lớp nhiều nhất và ít nhất chỉ lệch nhau <b>{spread:.2f} điểm phần trăm</b> "
             "(do chia ngẫu nhiên, không phân tầng). Với dữ liệu cân bằng thế này, accuracy là metric đủ tin cậy; "
             "macro-F1 vẫn hữu ích để thấy lớp nào kéo điểm xuống.", "good" if ratio < 1.1 else "warn"))
    if ratio < 1.1:
        INSIGHTS.append(("good", f"<b>Dữ liệu cân bằng</b>: mỗi lớp {fmt(cnt_tr[0])} ảnh train và {fmt(cnt_te[0])} ảnh test "
                                 f"(tỉ lệ max/min = {ratio:.2f}), không cần class weight hay resampling."))
    else:
        INSIGHTS.append(("warn", f"<b>Mất cân bằng lớp</b>: tỉ lệ max/min = {ratio:.2f}; nên cân nhắc class weight hoặc oversampling."))


# --------------------------------------------------------------------------- #
# 3. Duyệt ảnh mẫu (tương tác)
# --------------------------------------------------------------------------- #
def sec_gallery(c, per_class=60, per_row=14):
    add_section("anh-mau", "Duyệt ảnh mẫu",
                "Nhìn tận mắt trước khi nhìn số liệu. Chọn một lớp để xem ngẫu nhiên "
                f"{per_class} ảnh train của lớp đó; mỗi lần chạy lại script dùng cùng seed nên bộ ảnh giữ nguyên.")
    cache: dict[int, str] = {}

    def tile(i):
        if i not in cache:
            cache[i] = png_b64(c.xtr_u8[i])
        return px_img(cache[i], CLASSES[c.ytr[i]], 56)

    picks = [c.rng.choice(c.idx[k], per_class, replace=False) for k in range(K)]
    chips = ['<button class="chip is-on" data-k="all">Tất cả lớp</button>']
    chips += [f'<button class="chip" data-k="{k}">{cname(k)}</button>' for k in range(K)]
    all_rows = "".join(f'<div class="g-row"><div class="g-label">{cname(k)}</div>'
                       f'<div class="thumbs">{"".join(tile(i) for i in picks[k][:per_row])}</div></div>' for k in range(K))
    panels = [f'<div data-panel="all">{all_rows}</div>']
    panels += [f'<div data-panel="{k}" hidden><div class="thumbs big">{"".join(tile(i) for i in picks[k])}</div></div>'
               for k in range(K)]
    add(f'<div class="gallery"><div class="chips" role="group" aria-label="Chọn lớp">{"".join(chips)}</div>'
        f'{"".join(panels)}</div>')
    add(para("Hãy để ý các lớp trông rất giống nhau ở độ phân giải 28×28: T-shirt/top, Shirt, Pullover và Coat "
             "đều là thân áo có tay; Sandal, Sneaker và Ankle boot đều là giày. Các phần sau sẽ đo mức độ nhập nhằng này bằng số."))


# --------------------------------------------------------------------------- #
# 4. Thống kê pixel & chuẩn hoá
# --------------------------------------------------------------------------- #
def sec_pixels(c):
    add_section("thong-ke-pixel", "Thống kê pixel và chuẩn hoá",
                "Phân phối cường độ pixel quyết định cách chuẩn hoá đầu vào. Đây cũng là chỗ đối chiếu với "
               
    fit_idx, _, _ = val_split()
    X_fit = c.Xtr[fit_idx]
    m_tr, s_tr = float(X_fit.mean(dtype=np.float64)), float(X_fit.std(dtype=np.float64))
    m_te, s_te = float(c.Xte.mean(dtype=np.float64)), float(c.Xte.std(dtype=np.float64))
    zero_tr = float((c.xtr_u8 == 0).mean())
    c.m.update(mean=m_tr, std=s_tr, zero_share=zero_tr)

    p_tr = np.bincount(c.xtr_u8.ravel(), minlength=256) / c.xtr_u8.size
    p_te = np.bincount(c.xte_u8.ravel(), minlength=256) / c.xte_u8.size
    zero_cls = np.array([(c.xtr_u8[i] == 0).mean() * 100 for i in c.idx])

    add(stats([(f"{m_tr:.4f}", "mean pixel (train, thang 0-1)"), (f"{s_tr:.4f}", "std pixel (train)"),
               (f"{m_te:.4f}", "mean pixel (test)"), (f"{s_te:.4f}", "std pixel (test)"),
               (f"{zero_tr * 100:.1f}%", "pixel bằng 0 (nền đen)")]))

    fig, axes = new_fig(1, 3, w=18, h=4.4)
    ax = axes[0]
    ax.fill_between(np.arange(256), p_tr, step="mid", color="#7aa2ff", alpha=0.45, label="train")
    ax.step(np.arange(256), p_te, where="mid", color="#f2b84b", lw=1.2, label="test")
    ax.set_yscale("log")
    ax.set_xlabel("cường độ pixel (0-255)")
    ax.set_ylabel("tần suất (log)")
    ax.set_title("Phân phối cường độ pixel")
    ax.text(0.97, 0.9, f"pixel = 0: {zero_tr * 100:.1f}%", transform=ax.transAxes, ha="right", color=INK)
    ax.legend(loc="center right")
    ax.grid(axis="y")
    ax = axes[1]
    fg_tr, fg_te = p_tr[1:] / p_tr[1:].sum(), p_te[1:] / p_te[1:].sum()
    ax.fill_between(np.arange(1, 256), fg_tr, step="mid", color="#43c6b0", alpha=0.4, label="train")
    ax.step(np.arange(1, 256), fg_te, where="mid", color="#f2b84b", lw=1.2, label="test")
    ax.set_xlabel("cường độ pixel (chỉ pixel > 0)")
    ax.set_ylabel("tần suất (đã chuẩn hoá)")
    ax.set_title("Phân phối trên phần vật thể")
    ax.legend()
    ax.grid(axis="y")
    ax = axes[2]
    bars = ax.barh(range(K), zero_cls, color=CLASS_COLORS, height=0.7)
    bar_labels(ax, bars, "{:.1f}%", dx=0.6, horizontal=True)
    ax.set_yticks(range(K), CLASSES)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% pixel nền (= 0)")
    ax.set_title("Mức độ thưa của từng lớp")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    add(fig_html(fig, "Gần một nửa số pixel là nền đen tuyệt đối, và phần vật thể phân bố rộng trên toàn dải sáng. "
                      "Train và test gần như trùng khít."))

    rows = [
        ["Chỉ ToTensor() (thang 0-1)", f"{m_tr:.4f}", f"{s_tr:.4f}", "Chưa chuẩn hoá"],
        ["Normalize((0.5,), (0.5,)) &mdash; đang dùng",
         f"{(m_tr - 0.5) / 0.5:+.4f}", f"{s_tr / 0.5:.4f}", "Đưa về [-1, 1]; lệch âm, std &lt; 1"],
        [f"Normalize(({m_tr:.4f},), ({s_tr:.4f},)) &mdash; theo thống kê train", "0.0000", "1.0000",
         "Zero-mean, unit-variance"],
    ]
    add(h3("Chuẩn hoá đầu vào: đang dùng và theo dữ liệu"))
    add(table(["Cấu hình", "Mean sau chuẩn hoá", "Std sau chuẩn hoá", "Nhận xét"], rows, sortable=False))
    add(note(f"Normalize((0.5,), (0.5,)) hoạt động tốt trong thực tế, nhưng sau bước này dữ liệu có mean "
             f"≈ <b>{(m_tr - 0.5) / 0.5:+.2f}</b> và std ≈ <b>{s_tr / 0.5:.2f}</b>, tức chưa zero-mean/unit-variance. "
             f"Nếu muốn thử, dùng mean = {m_tr:.4f}, std = {s_tr:.4f} (tính trên train). "
             "Khi đổi, nhớ sửa luôn bước <code>image * 0.5 + 0.5</code> trong <code>show_predictions</code> của Predict.py cho khớp.", "info"))
    INSIGHTS.append(("info", f"<b>Chuẩn hoá</b>: thống kê train là mean = {m_tr:.4f}, std = {s_tr:.4f}. Cấu hình hiện tại (0.5, 0.5) "
                             f"cho mean ≈ {(m_tr - 0.5) / 0.5:+.2f}, std ≈ {s_tr / 0.5:.2f}; có thể thử chuẩn hoá theo thống kê thật."))
    INSIGHTS.append(("info", f"<b>Nền đen chiếm {zero_tr * 100:.0f}% pixel</b>, vật thể thường nằm giữa ảnh; " +
                             ("không có vị trí pixel nào luôn bằng 0." if c.m["never"] == 0 else f"có {c.m['never']} vị trí pixel luôn bằng 0.")))


# --------------------------------------------------------------------------- #
# 5. Đặc trưng cấp ảnh theo lớp
# --------------------------------------------------------------------------- #
def sec_features(c):
    add_section("dac-trung-lop", "Đặc trưng ảnh theo từng lớp",
                "Từ mỗi ảnh mình rút ra các số đo đơn giản (độ sáng, độ phủ, cạnh, kích thước vật thể, đối xứng) "
                "để xem các lớp khác nhau ở đâu, và lớp nào chồng lấn nhau.")
    fig, axes = new_fig(2, 4, w=19, h=9.2, sharey=True)
    for ax, (key, title) in zip(axes.ravel(), FEATS):
        hbox(ax, [c.F[key][i] for i in c.idx])
        ax.set_title(title, fontsize=10.5)
    for r in range(2):
        axes[r, 0].set_yticklabels(CLASSES)
    axes[0, 0].invert_yaxis()  # trục y chia sẻ: đảo một lần để T-shirt/top ở trên cùng
    add(fig_html(fig, "Mỗi hộp là khoảng tứ phân vị (25%-75%), vạch tối là trung vị, hình thoi sáng là trung bình, "
                      "râu kéo tới 1,5 lần IQR. Hộp của hai lớp càng chồng nhau thì đặc trưng đó càng ít giúp phân biệt."))

    add(h3("Hình dạng và vị trí vật thể"))
    fig, axes = new_fig(1, 2, w=13, h=5.4)
    sub = np.concatenate([c.rng.choice(i, 700, replace=False) for i in c.idx])
    ysub = c.ytr[sub]
    jit = lambda n: c.rng.uniform(-0.35, 0.35, n)  # noqa: E731
    axes[0].scatter(c.F["bbox_w"][sub] + jit(len(sub)), c.F["bbox_h"][sub] + jit(len(sub)),
                    c=[CLASS_COLORS[k] for k in ysub], s=6, alpha=0.45, linewidths=0)
    axes[0].set_xlabel("chiều rộng vật thể (px)")
    axes[0].set_ylabel("chiều cao vật thể (px)")
    axes[0].set_title("Khung bao vật thể")
    axes[0].grid()
    axes[1].scatter(c.F["cx"][sub], c.F["cy"][sub], c=[CLASS_COLORS[k] for k in ysub], s=6, alpha=0.45, linewidths=0)
    axes[1].set_xlabel("tọa độ x của trọng tâm sáng")
    axes[1].set_ylabel("tọa độ y của trọng tâm sáng")
    axes[1].invert_yaxis()
    axes[1].set_title("Trọng tâm cường độ sáng")
    axes[1].grid()
    bottom_legend(fig)
    full = float(((c.F["bbox_h"] >= 27) | (c.F["bbox_w"] >= 27)).mean())
    c.m["full_frame"] = full
    add(fig_html(fig, "Mỗi điểm là một ảnh train (700 ảnh mỗi lớp, có nhiễu nhẹ để khỏi chồng điểm). "
                      f"Các điểm xếp thành hình chữ L: <b>{full * 100:.1f}%</b> ảnh có vật thể chạm gần kín khung (≥ 27 px) ở ít nhất một chiều, "
                      "vì ảnh gốc được phóng/thu cho vừa khung 28 px. Vì vậy kích thước tuyệt đối gần như không giúp phân biệt lớp; "
                      "tỉ lệ rộng/cao mới là thông tin: quần, váy cao và hẹp, giày bè ngang, áo và túi gần vuông."))

    add(h3("Tương quan giữa các đặc trưng"))
    keys = [k for k, _ in FEATS] + ["cx", "cy"]
    labels = ["độ sáng", "tương phản", "độ phủ", "cạnh", "cao", "rộng", "rộng/cao", "đối xứng", "tâm x", "tâm y"]
    M = np.corrcoef(np.vstack([c.F[k] for k in keys]))
    fig, ax = new_fig(1, 1, w=8.5, h=7)
    im = ax.imshow(M, cmap=DIV, vmin=-1, vmax=1)
    annotate_heat(ax, im, M, size=8)
    ax.set_xticks(range(len(keys)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(keys)), labels)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="hệ số tương quan Pearson")
    add(fig_html(fig, f"Độ sáng tương quan mạnh với độ phủ ({M[0, 2]:.2f}) và độ tương phản ({M[0, 1]:.2f}) vì cùng tăng theo diện tích vật thể; "
                      f"mật độ cạnh độc lập hơn ({M[3, 0]:.2f} với độ sáng, {M[3, 2]:.2f} với độ phủ). "
                      f"Chiều cao và tỉ lệ rộng/cao gần như đối nghịch ({M[4, 6]:.2f}), nên không cần giữ cả hai khi dựng đặc trưng thủ công."))

    mean_f = {k: class_mean(c, k) for k, _ in FEATS}
    zero_cls = np.array([(c.xtr_u8[i] == 0).mean() * 100 for i in c.idx])
    spread = np.array([np.linalg.norm(c.Xtr[i] - c.mus[k], axis=1).mean() for k, i in enumerate(c.idx)])
    c.m["spread"] = spread
    rows = []
    for k in range(K):
        rows.append([cname(k), (f"{mean_f['brightness'][k]:.3f}", mean_f["brightness"][k]),
                     (f"{mean_f['coverage'][k]:.3f}", mean_f["coverage"][k]),
                     (f"{mean_f['edge'][k]:.3f}", mean_f["edge"][k]),
                     (f"{mean_f['bbox_h'][k]:.1f} × {mean_f['bbox_w'][k]:.1f}", mean_f["bbox_h"][k]),
                     (f"{mean_f['symmetry'][k]:.3f}", mean_f["symmetry"][k]),
                     (f"{zero_cls[k]:.1f}%", zero_cls[k]), (f"{spread[k]:.2f}", spread[k])])
    add(h3("Bảng tóm tắt theo lớp (bấm tiêu đề cột để sắp xếp)"))
    add(table(["Lớp", "Độ sáng TB", "Độ phủ", "Mật độ cạnh", "Cao × rộng (px)", "Đối xứng L-R", "% nền", "Phân tán quanh tâm lớp"], rows))

    sym = mean_f["symmetry"]
    c.m.update(sym_min=float(sym.min()), sym_max=float(sym.max()), sym_lo=int(sym.argmin()), sym_hi=int(sym.argmax()))
    INSIGHTS.append(("info", f"<b>Vật thể luôn được phóng cho vừa khung</b>: {c.m['full_frame'] * 100:.0f}% ảnh chạm gần kín 28 px ở ít nhất một chiều, "
                             "nên MLP phải dựa vào hình dạng và kết cấu chứ không phải kích thước tuyệt đối."))
    INSIGHTS.append(("info", f"<b>Độ phủ vật thể</b> khác nhau rõ giữa các lớp (từ {mean_f['coverage'].min():.2f} ở "
                             f"{esc(CLASSES[int(mean_f['coverage'].argmin())])} đến {mean_f['coverage'].max():.2f} ở "
                             f"{esc(CLASSES[int(mean_f['coverage'].argmax())])}); mô hình có thể tận dụng cả thông tin hình dạng thô lẫn chi tiết."))


# --------------------------------------------------------------------------- #
# 6. Ảnh trung bình, độ biến thiên, pixel quan trọng
# --------------------------------------------------------------------------- #
def sec_means(c):
    add_section("anh-trung-binh", "Ảnh trung bình và vùng pixel quan trọng",
                "Trung bình và độ lệch chuẩn theo từng pixel cho thấy dáng \"điển hình\" của mỗi lớp, "
                "vùng nào ổn định và vùng nào biến động.")
    stds = np.stack([c.Xtr[i].std(axis=0) for i in c.idx])
    freq = np.stack([(c.Xtr[i] > 0.1).mean(axis=0) for i in c.idx])
    rows = [("Ảnh trung bình", c.mus, "gray"), ("Độ lệch chuẩn (std)", stds, "magma"),
            ("P(pixel có vật thể)", freq, "cividis")]
    fig, axes = new_fig(3, K, w=20, h=6.9)
    for r, (name, data, cmap) in enumerate(rows):
        for k in range(K):
            ax = axes[r, k]
            ax.imshow(data[k].reshape(H, W), cmap=cmap)
            hide_frame(ax)
            if r == 0:
                ax.set_title(CLASSES[k], fontsize=9.5, color=CLASS_COLORS[k], loc="center")
            if k == 0:
                ax.set_ylabel(name, fontsize=10, color=INK)
    add(fig_html(fig, "Hàng 1: ảnh trung bình của lớp. Hàng 2: độ lệch chuẩn, sáng nghĩa là pixel đó thay đổi nhiều giữa các ảnh cùng lớp "
                      "(thường là viền và chi tiết). Hàng 3: xác suất pixel thuộc vật thể."))

    mu = c.mu
    n_k = np.array([len(i) for i in c.idx])
    between = (n_k[:, None] * (c.mus - mu) ** 2).sum(0) / (K - 1)
    within = sum(((c.Xtr[i] - c.mus[k]) ** 2).sum(0) for k, i in enumerate(c.idx)) / (len(c.ytr) - K)
    Fmap = between / (within + 1e-8)
    c.m["F_top"] = np.argsort(Fmap)[::-1][:5]

    fig, axes = new_fig(1, 3, w=17, h=5)
    for ax, data, title, cmap in ((axes[0], mu.reshape(H, W), "Ảnh trung bình toàn bộ tập train", "gray"),
                                  (axes[1], c.Xtr.std(axis=0).reshape(H, W), "Độ lệch chuẩn từng pixel", "magma"),
                                  (axes[2], np.log10(Fmap + 1e-3).reshape(H, W), "Khả năng phân biệt lớp (log10 F-ratio)", "viridis")):
        im = ax.imshow(data, cmap=cmap)
        hide_frame(ax)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8)
    add(fig_html(fig, "Bản đồ F-ratio (phương sai giữa các lớp chia phương sai trong lớp) cho thấy pixel nào giúp tách lớp tốt nhất; "
                      "vùng sáng nhất tập trung ở phần vai/cổ phía trên và một dải bên phải nửa dưới (nơi các loại giày khác nhau), còn các góc và viền ngoài đóng góp ít nhất."))

    pr_row = [c.mus[k].reshape(H, W).mean(axis=1) for k in range(K)]
    pr_col = [c.mus[k].reshape(H, W).mean(axis=0) for k in range(K)]
    fig, axes = new_fig(1, 2, w=13, h=4.4, sharey=True)
    for k in range(K):
        axes[0].plot(range(H), pr_row[k], color=CLASS_COLORS[k], lw=1.8)
        axes[1].plot(range(W), pr_col[k], color=CLASS_COLORS[k], lw=1.8)
    axes[0].set_xlabel("chỉ số hàng (0 = trên cùng)")
    axes[0].set_ylabel("cường độ trung bình")
    axes[0].set_title("Hồ sơ theo chiều dọc")
    axes[1].set_xlabel("chỉ số cột (0 = trái)")
    axes[1].set_title("Hồ sơ theo chiều ngang")
    for ax in axes:
        ax.grid()
    bottom_legend(fig)
    add(fig_html(fig, "Cường độ trung bình theo từng hàng và cột. Áo, quần, váy trải khá đều theo chiều dọc; giày và túi tập trung ở khoảng hàng 10-22."))


# --------------------------------------------------------------------------- #
# 7. Độ tương đồng giữa các lớp
# --------------------------------------------------------------------------- #
def sec_similarity(c):
    add_section("tuong-dong-lop", "Độ tương đồng giữa các lớp",
                "So sánh các ảnh trung bình của lớp để biết những cặp lớp nào gần nhau trong không gian pixel. "
                "Đây là dự báo đầu tiên về các cặp nhãn mà mô hình dễ nhầm.")
    D = c.mus - c.mu
    nrm = np.linalg.norm(D, axis=1)
    cos = (D @ D.T) / np.outer(nrm, nrm)
    eu = np.linalg.norm(c.mus[:, None] - c.mus[None], axis=2)
    fig, axes = new_fig(1, 2, w=17, h=7.4)
    im = axes[0].imshow(cos, cmap=DIV, vmin=-1, vmax=1)
    annotate_heat(axes[0], im, cos)
    axes[0].set_title("Cosine giữa các ảnh trung bình (đã trừ trung bình toàn cục)")
    fig.colorbar(im, ax=axes[0], shrink=0.75)
    im = axes[1].imshow(eu, cmap="viridis_r")
    annotate_heat(axes[1], im, eu, "{:.1f}")
    axes[1].set_title("Khoảng cách Euclid giữa các ảnh trung bình")
    fig.colorbar(im, ax=axes[1], shrink=0.75)
    for ax in axes:
        ax.set_xticks(range(K), CLASSES, rotation=40, ha="right")
        ax.set_yticks(range(K), CLASSES)
        ax.spines[:].set_visible(False)
    add(fig_html(fig, "Bên trái: cosine gần +1 nghĩa là hai lớp lệch khỏi \"ảnh trung bình\" theo cùng hướng. "
                      "Bên phải: màu vàng sáng là hai lớp rất gần nhau."))
    pairs = sorted(((eu[i, j], i, j) for i in range(K) for j in range(i + 1, K)))[:7]
    c.m["close_pairs"] = pairs
    rows = [[f"{cname(i)} &harr; {cname(j)}", (f"{d:.2f}", d), (f"{cos[i, j]:.2f}", cos[i, j])] for d, i, j in pairs]
    add(h3("Bảy cặp lớp gần nhau nhất"))
    add(table(["Cặp lớp", "Khoảng cách Euclid", "Cosine"], rows))
    txt = ", ".join(f"{CLASSES[i]} &harr; {CLASSES[j]}" for _, i, j in pairs[:3])
    INSIGHTS.append(("warn", f"<b>Các cặp lớp gần nhau nhất</b> trong không gian pixel: {txt}. Đây là những cặp mô hình có nguy cơ nhầm nhiều nhất."))


# --------------------------------------------------------------------------- #
# 8. PCA
# --------------------------------------------------------------------------- #
def compute_pca(c):
    Xc = c.Xtr - c.mu
    cov = (Xc.T @ Xc) / (len(Xc) - 1)
    ev, V = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(ev)[::-1]
    ev, V = ev[order], V[:, order]
    c.ev, c.V = ev, V
    c.evr = ev / ev.sum()
    c.cum = np.cumsum(c.evr)
    c.Z = (Xc @ V[:, :100]).astype(np.float32)
    c.Zte = ((c.Xte - c.mu) @ V[:, :100]).astype(np.float32)
    del Xc


def sec_pca(c):
    add_section("pca", "Giảm chiều bằng PCA",
                "784 chiều có thực sự cần thiết không? PCA cho biết lượng thông tin nằm trong vài thành phần chính "
                "đầu tiên và liệu các lớp có tách được trong 2 chiều hay không.")
    k90, k95, k99 = (int(np.searchsorted(c.cum, t) + 1) for t in (0.90, 0.95, 0.99))
    c.m.update(k90=k90, k95=k95, k99=k99)
    add(stats([(str(k90), "thành phần giữ 90% phương sai"), (str(k95), "giữ 95%"), (str(k99), "giữ 99%"),
               (f"{c.evr[0] * 100:.1f}%", "PC1 một mình")]))

    fig, axes = new_fig(1, 3, w=19, h=5, gridspec_kw={"width_ratios": [1, 1, 1.25]})
    n_show = 30
    axes[0].bar(range(1, n_show + 1), c.evr[:n_show] * 100, color="#7aa2ff")
    axes[0].set_xlabel("thành phần chính")
    axes[0].set_ylabel("% phương sai")
    axes[0].set_title("Phương sai giải thích (30 thành phần đầu)")
    axes[0].grid(axis="y")
    axes[0].set_axisbelow(True)
    axes[1].plot(range(1, 785), c.cum * 100, color="#43c6b0", lw=2)
    for t, kk, col in ((90, k90, "#f2b84b"), (95, k95, "#ef6f6c"), (99, k99, "#c58af9")):
        axes[1].axhline(t, color=col, ls="--", lw=1)
        axes[1].plot([kk], [t], "o", color=col)
        if kk < 300:
            axes[1].text(kk + 12, t - 4.5, f"{t}% @ {kk}", color=col, fontsize=9)
        else:
            axes[1].text(kk - 12, t - 4.5, f"{t}% @ {kk}", color=col, fontsize=9, ha="right")
    axes[1].set_xlim(0, max(400, int(k99 * 1.1)))
    axes[1].set_ylim(0, 102)
    axes[1].set_xlabel("số thành phần giữ lại")
    axes[1].set_ylabel("% phương sai tích lũy")
    axes[1].set_title("Phương sai tích lũy")
    axes[1].grid()
    sub = np.concatenate([c.rng.choice(i, 600, replace=False) for i in c.idx])
    axes[2].scatter(c.Z[sub, 0], c.Z[sub, 1], c=[CLASS_COLORS[k] for k in c.ytr[sub]], s=7, alpha=0.55, linewidths=0)
    axes[2].set_xlabel(f"PC1 ({c.evr[0] * 100:.1f}%)")
    axes[2].set_ylabel(f"PC2 ({c.evr[1] * 100:.1f}%)")
    axes[2].set_title("Chiếu 2 chiều (600 ảnh mỗi lớp)")
    axes[2].grid()
    bottom_legend(fig, ncol=10, y=-0.02)
    add(fig_html(fig, f"Chỉ cần {k95} thành phần (khoảng {k95 / 784 * 100:.0f}% số chiều gốc) là giữ được 95% phương sai. "
                      "Trong 2 chiều đầu, giày tách thành một dải riêng và quần/váy nằm ở đáy, còn nhóm áo và túi chồng lên nhau ở bên phải."))

    fig, axes = new_fig(2, 5, w=15, h=6.6)
    for j, ax in enumerate(axes.ravel()):
        v = c.V[:, j].reshape(H, W)
        lim = np.abs(v).max()
        ax.imshow(v, cmap=DIV, vmin=-lim, vmax=lim)
        hide_frame(ax)
        ax.set_title(f"PC{j + 1} ({c.evr[j] * 100:.1f}%)", fontsize=10, loc="center")
    add(h3("Eigen-image: mười hướng biến thiên chính"))
    add(fig_html(fig, "Màu đỏ và xanh là hai cực của mỗi hướng (dấu của eigenvector là tùy ý). PC1 chủ yếu đo mức vật thể phủ vùng thân giữa ảnh, nên tách giày khỏi áo."))

    ks = [1, 5, 10, 25, 50, 100]
    show = [0, 1, 3, 5, 8, 9]
    fig, axes = new_fig(len(show), len(ks) + 1, w=10.5, h=1.7 * len(show))
    for r, cls in enumerate(show):
        i = c.idx_te[cls][0]
        x = c.Xte[i]
        axes[r, 0].imshow(x.reshape(H, W), cmap="gray", vmin=0, vmax=1)
        hide_frame(axes[r, 0])
        if r == 0:
            axes[r, 0].set_title("gốc", fontsize=9, loc="center")
        axes[r, 0].set_ylabel(CLASSES[cls], fontsize=8.5, color=CLASS_COLORS[cls])
        for j, kk in enumerate(ks):
            rec = c.mu + (x - c.mu) @ c.V[:, :kk] @ c.V[:, :kk].T
            axes[r, j + 1].imshow(np.clip(rec, 0, 1).reshape(H, W), cmap="gray", vmin=0, vmax=1)
            hide_frame(axes[r, j + 1])
            if r == 0:
                axes[r, j + 1].set_title(f"k = {kk}", fontsize=9, loc="center")
    add(h3("Khôi phục ảnh từ k thành phần chính"))
    add(fig_html(fig, "Với k = 25 đã nhận ra loại sản phẩm; k = 50 trở đi chỉ bổ sung chi tiết như nếp vải và hoa văn.", dpi=150))
    INSIGHTS.append(("info", f"<b>Số chiều thực chất thấp</b>: {k90} / {k95} / {k99} thành phần giữ 90 / 95 / 99% phương sai (trên 784 chiều). "
                             "Đầu vào dư thừa nhiều, nên dropout hoặc weight decay thường có ích cho MLP."))


# --------------------------------------------------------------------------- #
# 9. t-SNE (tuỳ chọn)
# --------------------------------------------------------------------------- #
def sec_tsne(c):
    n = c.args.tsne_samples
    if n <= 0:
        return
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        log("Bỏ qua t-SNE (chưa cài scikit-learn).")
        return
    log(f"Chạy t-SNE trên {n} ảnh ...")
    add_section("tsne", "Bản đồ t-SNE",
                f"t-SNE giữ lại cấu trúc lân cận cục bộ nên thường tách cụm tốt hơn PCA. Dùng {fmt(n)} ảnh train ngẫu nhiên, "
                "rút gọn bằng PCA 50 chiều trước khi chạy.")
    sub = c.rng.choice(len(c.ytr), n, replace=False)
    emb = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
               random_state=c.args.seed).fit_transform(c.Z[sub, :50])
    fig, ax = new_fig(1, 1, w=10, h=8)
    ax.scatter(emb[:, 0], emb[:, 1], c=[CLASS_COLORS[k] for k in c.ytr[sub]], s=6, alpha=0.7, linewidths=0)
    for k in range(K):
        pts = emb[c.ytr[sub] == k]
        cx, cy = np.median(pts, axis=0)
        ax.text(cx, cy, CLASSES[k], color="white", fontsize=9, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=CLASS_COLORS[k], ec="none", alpha=0.85))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_title("t-SNE 2 chiều")
    add(fig_html(fig, "Giày, túi và quần tạo các vùng riêng (ba loại giày vẫn lẫn vào nhau ở rìa). Pullover, Coat và Shirt trộn gần như hoàn toàn, "
                      "còn T-shirt/top và Dress nằm sát nhau ở phía dưới."))


# --------------------------------------------------------------------------- #
# 10. Baseline đơn giản
# --------------------------------------------------------------------------- #
def knn_predict(Ztr, ytr, Zte, k=5, chunk=500):
    sq = (Ztr ** 2).sum(axis=1)
    pred = np.empty(len(Zte), np.int64)
    for s in range(0, len(Zte), chunk):
        q = Zte[s:s + chunk]
        d = sq[None, :] - 2 * q @ Ztr.T  # bỏ ||q||^2: không ảnh hưởng thứ hạng láng giềng
        nn = np.argpartition(d, k, axis=1)[:, :k]
        votes = ytr[nn]
        counts = np.zeros((len(q), K), np.int32)
        for j in range(k):
            counts[np.arange(len(q)), votes[:, j]] += 1
        pred[s:s + chunk] = counts.argmax(axis=1)
    return pred


def confusion(y, p):
    cm = np.zeros((K, K), np.int64)
    np.add.at(cm, (y, p), 1)
    return cm


def sec_baseline(c):
    add_section("baseline", "Baseline đơn giản: mốc để so với MLP",
                "Hai bộ phân loại không cần huấn luyện: gán nhãn theo ảnh trung bình gần nhất, và k-NN (k = 5) trên 50 thành phần PCA. "
                "Kết quả vừa là mốc tham chiếu cho MLP, vừa chỉ ra lớp nào vốn khó.")
    # Nearest centroid
    d = (c.Xte ** 2).sum(1)[:, None] - 2 * c.Xte @ c.mus.T + (c.mus ** 2).sum(1)[None]
    p_cent = d.argmin(axis=1)
    acc_cent = float((p_cent == c.yte).mean())
    # kNN
    n_knn = min(c.args.knn_test, len(c.yte))
    sel = np.arange(n_knn) if n_knn == len(c.yte) else c.rng.choice(len(c.yte), n_knn, replace=False)
    log(f"Chạy k-NN trên {n_knn} ảnh test ...")
    p_knn = knn_predict(c.Z[:, :50], c.ytr, c.Zte[sel, :50], k=5)
    y_knn = c.yte[sel]
    acc_knn = float((p_knn == y_knn).mean())
    cm_knn = confusion(y_knn, p_knn)
    rec_knn = cm_knn.diagonal() / cm_knn.sum(1)
    rec_cent = confusion(c.yte, p_cent).diagonal() / np.bincount(c.yte, minlength=K)
    c.m.update(acc_cent=acc_cent, acc_knn=acc_knn, rec_knn=rec_knn, n_knn=n_knn)

    add(stats([(f"{acc_cent * 100:.2f}%", "accuracy: ảnh trung bình gần nhất", f"trên {fmt(len(c.yte))} ảnh test"),
               (f"{acc_knn * 100:.2f}%", "accuracy: k-NN (k=5) + PCA 50", f"trên {fmt(n_knn)} ảnh test")]))

    fig, axes = new_fig(1, 2, w=17, h=7, gridspec_kw={"width_ratios": [1.15, 1]})
    cmn = cm_knn / cm_knn.sum(1, keepdims=True) * 100
    im = axes[0].imshow(cmn, cmap="magma", vmin=0, vmax=100)
    annotate_heat(axes[0], im, cmn, "{:.0f}", size=8.5)
    axes[0].set_xticks(range(K), CLASSES, rotation=40, ha="right")
    axes[0].set_yticks(range(K), CLASSES)
    axes[0].set_xlabel("nhãn dự đoán")
    axes[0].set_ylabel("nhãn thật")
    axes[0].set_title("Ma trận nhầm lẫn của k-NN (% theo hàng)")
    axes[0].spines[:].set_visible(False)
    fig.colorbar(im, ax=axes[0], shrink=0.7)
    ax = axes[1]
    y = np.arange(K)
    bars_c = ax.barh(y - 0.19, rec_cent * 100, height=0.36, color="#f2b84b", label="ảnh trung bình gần nhất")
    bar_labels(ax, bars_c, "{:.0f}", dx=0.8, horizontal=True, size=8)
    bars = ax.barh(y + 0.19, rec_knn * 100, height=0.36, color="#7aa2ff", label="k-NN + PCA")
    bar_labels(ax, bars, "{:.0f}", dx=0.8, horizontal=True, size=8)
    ax.set_yticks(y, CLASSES)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("recall theo lớp (%)")
    ax.set_title("Recall từng lớp")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8.5)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    add(fig_html(fig, "Hầu hết lỗi dồn vào nhóm áo: Shirt bị nhầm với T-shirt/top, Pullover và Coat; Sneaker/Sandal/Ankle boot nhầm lẫn lẫn nhau ở mức thấp hơn."))

    off = [(cm_knn[i, j], i, j) for i in range(K) for j in range(K) if i != j]
    off.sort(reverse=True)
    c.m["top_conf"] = off[:6]
    rows = [[cname(i), cname(j), (fmt(n), n), (f"{n / cm_knn[i].sum() * 100:.1f}%", n)] for n, i, j in off[:8]]
    add(h3("Tám kiểu nhầm lẫn phổ biến nhất (k-NN)"))
    add(table(["Nhãn thật", "Bị đoán thành", "Số ảnh", "% của lớp thật"], rows))

    wrong = np.flatnonzero(p_knn != y_knn)
    pick = c.rng.choice(wrong, min(28, len(wrong)), replace=False)
    tiles = []
    for w_ in pick:
        i = int(sel[w_])
        tiles.append(f'<figure class="tile">{px_img(png_b64(c.xte_u8[i]), "", 64)}'
                     f'<figcaption><span style="color:{CLASS_COLORS[y_knn[w_]]}">{esc(CLASSES[y_knn[w_]])}</span><br>'
                     f'&rarr; <span style="color:{CLASS_COLORS[p_knn[w_]]}">{esc(CLASSES[p_knn[w_]])}</span></figcaption></figure>')
    add(h3("Một số ảnh test bị k-NN đoán sai"))
    add(f'<div class="tiles">{"".join(tiles)}</div>')
    add(para("Dòng trên là nhãn thật, dòng dưới là nhãn bị đoán. Nhiều ảnh trong số này, kể cả người nhìn cũng phải phân vân."))

    worst = np.argsort(rec_knn)[:3]
    INSIGHTS.append(("warn", f"<b>Lớp khó nhất</b> theo k-NN: " + ", ".join(f"{esc(CLASSES[k])} ({rec_knn[k] * 100:.0f}%)" for k in worst) +
                             ". Hãy xem các lớp này đầu tiên khi đọc confusion matrix của MLP."))
    INSIGHTS.append(("info", f"<b>Mốc tham chiếu</b>: ảnh trung bình gần nhất đạt {acc_cent * 100:.1f}%, k-NN + PCA đạt {acc_knn * 100:.1f}%. "
                             "MLP của bạn nên vượt rõ rệt các mốc này; nếu không, nhiều khả năng có lỗi ở pipeline."))


# --------------------------------------------------------------------------- #
# 11. Ảnh điển hình / bất thường
# --------------------------------------------------------------------------- #
def sec_outliers(c, n=8):
    add_section("bat-thuong", "Ảnh điển hình và ảnh bất thường",
                "Với mỗi lớp, mình xếp các ảnh train theo khoảng cách tới ảnh trung bình của lớp. Ảnh gần tâm nhất là \"mẫu chuẩn\"; "
                "ảnh xa tâm nhất là ứng viên cho nhiễu nhãn, sản phẩm lạ hoặc góc chụp bất thường.")
    rows = []
    for k in range(K):
        d = np.linalg.norm(c.Xtr[c.idx[k]] - c.mus[k], axis=1)
        order = np.argsort(d)
        typ, atyp = c.idx[k][order[:n]], c.idx[k][order[-n:][::-1]]
        t_html = "".join(px_img(png_b64(c.xtr_u8[i]), "", 52) for i in typ)
        a_html = "".join(px_img(png_b64(c.xtr_u8[i]), "", 52) for i in atyp)
        rows.append(f'<div class="o-row"><div class="g-label">{cname(k)}<small>d TB = {d.mean():.2f}</small></div>'
                    f'<div class="thumbs">{t_html}</div><div class="thumbs far">{a_html}</div></div>')
    add('<div class="o-head"><span></span><span>Gần tâm lớp nhất (điển hình)</span><span>Xa tâm lớp nhất (bất thường)</span></div>')
    add(f'<div class="outliers">{"".join(rows)}</div>')
    add(note("Ảnh bất thường không nhất thiết là sai: nhiều ảnh chỉ là biến thể hiếm (áo có hoa văn đậm, túi có quai dài). "
             "Nếu muốn dọn dữ liệu, hãy xem tay danh sách này thay vì xoá tự động.", "info"))


# --------------------------------------------------------------------------- #
# 12. Train vs Test
# --------------------------------------------------------------------------- #
def ks_stat(a, b):
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, allv, side="right") / len(a) - np.searchsorted(b, allv, side="right") / len(b)).max())


def sec_shift(c):
    add_section("train-test", "So sánh train và test",
                "Nếu tập test có phân phối khác train, accuracy trên test sẽ không phản ánh đúng chất lượng mô hình. "
                "Mình so sánh các đặc trưng ảnh bằng thống kê Kolmogorov-Smirnov (KS), 0 là trùng khít, 1 là tách hẳn.")
    fig, axes = new_fig(1, 3, w=18, h=4.4)
    for ax, key, title in ((axes[0], "brightness", "Độ sáng trung bình mỗi ảnh"), (axes[1], "coverage", "Độ phủ vật thể mỗi ảnh")):
        lo, hi = 0, max(c.F[key].max(), c.Fte[key].max())
        bins = np.linspace(lo, hi, 50)
        ax.hist(c.F[key], bins=bins, density=True, color="#7aa2ff", alpha=0.45, label="train")
        ax.hist(c.Fte[key], bins=bins, density=True, histtype="step", color="#f2b84b", lw=1.6, label="test")
        ax.set_title(title)
        ax.set_ylabel("mật độ")
        ax.legend()
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    diff = c.Xte.mean(axis=0) - c.mu
    lim = max(np.abs(diff).max(), 1e-6)
    im = axes[2].imshow(diff.reshape(H, W), cmap=DIV, vmin=-lim, vmax=lim)
    hide_frame(axes[2])
    axes[2].set_title("Chênh lệch ảnh trung bình (test − train)")
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    add(fig_html(fig, f"Chênh lệch lớn nhất ở bất kỳ pixel nào chỉ là {np.abs(diff).max():.3f} trên thang 0-1."))

    rows, ks_all = [], {}
    for key, title in FEATS:
        v = ks_stat(c.F[key], c.Fte[key])
        ks_all[key] = v
        rows.append([title, (f"{v:.4f}", v), (f"{c.F[key].mean():.3f}", c.F[key].mean()), (f"{c.Fte[key].mean():.3f}", c.Fte[key].mean())])
    c.m["ks_max"] = max(ks_all.values())
    add(table(["Đặc trưng", "KS (train vs test)", "Trung bình train", "Trung bình test"], rows))
    kind = "good" if c.m["ks_max"] < 0.05 else "warn"
    add(note(f"KS lớn nhất là <b>{c.m['ks_max']:.3f}</b>: " + ("hai tập gần như cùng phân phối, kết quả trên test đáng tin." if kind == "good"
                                                                   else "có độ lệch đáng chú ý giữa train và test, cần xem lại."), kind))
    INSIGHTS.append((kind, f"<b>Train và test cùng phân phối</b> (KS lớn nhất = {c.m['ks_max']:.3f}); validation tách từ train nên cũng cùng phân phối."))


# --------------------------------------------------------------------------- #
# 13. Tăng cường dữ liệu
# --------------------------------------------------------------------------- #
def shift_img(img, dy, dx):
    out = np.zeros_like(img)
    out[max(dy, 0):H - max(-dy, 0), max(dx, 0):W - max(-dx, 0)] = img[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return out


def sec_aug(c):
    add_section("tang-cuong", "Xem trước các phép tăng cường dữ liệu",
                "Phép biến đổi chỉ hợp lệ khi nó giữ nguyên nhãn và vẫn giống dữ liệu thật. Dữ liệu đo được ở trên giúp chọn phép nào nên dùng.")
    rng = np.random.default_rng(c.args.seed)
    show = [0, 1, 3, 5, 8, 9]
    cols = ["Gốc", "Lật ngang ✓", "Lật dọc ✗", "Dịch (+3, −2) ✓", "Nhiễu Gauss σ=0.1", "Random erasing ✓"]
    fig, axes = new_fig(len(show), len(cols), w=11, h=1.75 * len(show))
    for r, cls in enumerate(show):
        x = c.Xte[c.idx_te[cls][3]].reshape(H, W)
        er = x.copy()
        y0, x0 = rng.integers(2, 18, 2)
        er[y0:y0 + 8, x0:x0 + 8] = 0
        outs = [x, x[:, ::-1], x[::-1, :], shift_img(x, 3, -2), np.clip(x + rng.normal(0, 0.1, x.shape), 0, 1), er]
        for j, o in enumerate(outs):
            axes[r, j].imshow(o, cmap="gray", vmin=0, vmax=1)
            hide_frame(axes[r, j])
            if r == 0:
                axes[r, j].set_title(cols[j], fontsize=8.8, loc="center")
            if j == 0:
                axes[r, j].set_ylabel(CLASSES[cls], fontsize=8.5, color=CLASS_COLORS[cls])
    add(fig_html(fig, "✓ = nên cân nhắc, ✗ = không nên: ảnh test không bao giờ bị lộn ngược, nên lật dọc chỉ làm mô hình học một phân phối không tồn tại."))
    sym = class_mean(c, "symmetry")
    add(para(f"Độ đối xứng trái-phải trung bình theo lớp nằm trong khoảng <b>{sym.min():.2f} đến {sym.max():.2f}</b> "
             f"(thấp nhất ở {esc(CLASSES[int(sym.argmin())])}, cao nhất ở {esc(CLASSES[int(sym.argmax())])}). "
             "Với áo, quần, váy, đối xứng cao nên lật ngang gần như không đổi ảnh. Với giày và túi, ảnh lật ngang vẫn là một sản phẩm hợp lệ "
             "(chỉ đổi chiều mũi giày), nên <code>RandomHorizontalFlip</code> trong Train.py là lựa chọn an toàn."))
    add(note("Train.py đã import <code>RandomCrop</code> và <code>RandomErasing</code> nhưng <code>train_transform</code> hiện chỉ có "
             "<code>RandomHorizontalFlip</code>. Hai phép này đều hợp lệ với dữ liệu trên. Lưu ý <code>RandomErasing</code> "
             "phải đặt <b>sau</b> <code>ToTensor()</code>, và vì MLP không có tính bất biến dịch chuyển như CNN nên một phép dịch nhỏ "
             "(<code>RandomCrop(28, padding=2)</code>) thường giúp giảm overfit.", "info"))
    INSIGHTS.append(("info", "<b>Tăng cường dữ liệu</b>: lật ngang là hợp lệ (đã dùng). Có thể thử thêm dịch nhỏ (RandomCrop padding 2) và RandomErasing; "
                             "không nên lật dọc."))


# --------------------------------------------------------------------------- #
# Tóm tắt & dựng HTML
# --------------------------------------------------------------------------- #
def build_summary(c):
    sec = {"id": "tom-tat", "title": "Phát hiện chính", "lead": "Những điều rút ra được từ toàn bộ báo cáo, xếp theo thứ tự các phần.", "body": []}
    items = "".join(f'<li class="ins {k}">{t}</li>' for k, t in INSIGHTS)
    sec["body"].append(f'<ul class="insights">{items}</ul>')
    SECTIONS.insert(1, sec)


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


def render_html(c, elapsed: float) -> str:
    nav = "".join(f'<a href="#{s["id"]}">{esc(s["title"])}</a>' for s in SECTIONS)
    body = "".join(
        f'<section id="{s["id"]}"><h2>{esc(s["title"])}</h2>'
        + (f'<p class="lead">{s["lead"]}</p>' if s["lead"] else "") + "".join(s["body"]) + "</section>"
        for s in SECTIONS)
    sw = "".join(
        f'<figure class="sw">{px_img(png_b64((c.mus[k] / c.mus[k].max() * 255).reshape(H, W)), CLASSES[k], 84)}'
        f'<i style="background:{CLASS_COLORS[k]}"></i>{esc(CLASSES[k])}</figure>' for k in range(K))
    meta = (f'<span>Tạo lúc {datetime.now():%d/%m/%Y %H:%M}</span><span>Chạy trong {elapsed:.0f} giây</span>'
            f'<span>numpy {np.__version__}, matplotlib {matplotlib.__version__}</span>')
    return f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EDA Fashion-MNIST</title>
<style>{CSS}</style></head>
<body><div class="shell">
<nav class="side" aria-label="Mục lục"><strong>EDA Fashion-MNIST</strong>{nav}</nav>
<main>
<header class="hero">
<h1>Khám phá dữ liệu Fashion-MNIST</h1>
<p class="sub">70.000 ảnh xám 28×28 thuộc 10 nhóm sản phẩm. Báo cáo đi từ cấu trúc dữ liệu và thống kê pixel đến mức độ khó phân biệt giữa các lớp,
để biết trước mô hình MLP sẽ gặp khó ở đâu. Mười ô dưới đây là ảnh trung bình của từng lớp.</p>
<div class="swatches">{sw}</div>
<p class="meta">{meta}</p>
</header>
{body}
<footer>Báo cáo sinh bởi EDA.py. Bấm vào bất kỳ biểu đồ nào để phóng to, bấm tiêu đề cột trong bảng để sắp xếp.</footer>
</main></div>
<dialog id="lb"><img alt="Biểu đồ phóng to"></dialog>
<script>{JS}</script></body></html>"""


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="EDA Fashion-MNIST -> báo cáo HTML")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "eda_report.html")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tsne-samples", type=int, default=5000, help="0 = bỏ t-SNE")
    ap.add_argument("--knn-test", type=int, default=10000, help="số ảnh test cho k-NN baseline")
    args = ap.parse_args()

    log("Nạp dữ liệu ...")
    c = prepare(args)
    steps = [("Tổng quan", sec_overview), ("Phân phối lớp", sec_classes), ("Ảnh mẫu", sec_gallery),
             ("Thống kê pixel", sec_pixels), ("Đặc trưng theo lớp", sec_features), ("Ảnh trung bình", sec_means),
             ("Tương đồng lớp", sec_similarity)]
    for name, fn in steps:
        log(name)
        fn(c)
    log("PCA")
    compute_pca(c)
    sec_pca(c)
    sec_tsne(c)
    for name, fn in (("Baseline", sec_baseline), ("Ảnh bất thường", sec_outliers), ("Train vs Test", sec_shift),
                     ("Tăng cường dữ liệu", sec_aug)):
        log(name)
        fn(c)
    build_summary(c)
    html_text = render_html(c, time.time() - _T0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text, encoding="utf-8")
    log(f"Xong: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
