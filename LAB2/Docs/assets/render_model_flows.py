"""Render the architecture figures referenced by LAB2/Docs/model.md."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as PlotPath


ROOT = Path(__file__).resolve().parent
INK = "#16304B"
PALETTE = {
    "frozen": ("#EDF2F7", "#ABBACB"),
    "tune": ("#FFF0CD", "#D8A245"),
    "head": ("#DEF5E9", "#52A582"),
    "input": ("#E6F0FF", "#7FA7DA"),
    "op": ("#FFFFFF", "#B4C7DD"),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "svg.hashsalt": "lab2-model-flows"})


class Diagram:
    def __init__(self, name, title, subtitle, height=12):
        self.name = name
        self.fig = plt.figure(figsize=(12, height), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 1), ylim=(0, 1))
        self.ax.axis("off")
        self.boxes, self.edges, self.text_bounds = {}, [], []
        self.label(.045, .965, title, 23, "left", "bold")
        self.label(.045, .929, subtitle, 11.5, "left")

    def label(self, x, y, value, size=11, align="center", weight="normal", color=INK):
        return self.ax.text(x, y, value, ha=align, va="center", fontsize=size,
                            fontweight=weight, color=color, linespacing=1.4)

    def node(self, key, x, y, title, body="", kind="frozen", w=.445, h=.056,
             title_size=12.2, body_size=10.7):
        fill, stroke = PALETTE[kind]
        patch = FancyBboxPatch((x-w/2, y-h/2), w, h,
                               boxstyle="round,pad=0.004,rounding_size=0.009",
                               linewidth=1.1, facecolor=fill, edgecolor=stroke)
        self.ax.add_patch(patch)
        if body:
            texts = [self.label(x, y+h*.19, title, title_size, weight="bold"),
                     self.label(x, y-h*.22, body, body_size)]
        else:
            texts = [self.label(x, y, title, title_size, weight="bold")]
        self.text_bounds.extend((text, patch) for text in texts)
        self.boxes[key] = (x, y, w, h, title, body, kind)

    def arrow(self, start, end, side=None, color="#71839A", dashed=False):
        a, b = self.boxes[start], self.boxes[end]
        if side is None:
            points = [(a[0], a[1]-a[3]/2-.004),
                      (b[0], b[1]+b[3]/2+.005)]
        else:
            points = [(a[0]+a[2]/2+.003, a[1]), (side, a[1]),
                      (side, b[1]), (b[0]+b[2]/2+.006, b[1])]
        self.path(points, color, dashed)
        self.edges.append((start, end, dashed))

    def path(self, points, color="#71839A", dashed=False):
        line = PlotPath(points, [PlotPath.MOVETO]+[PlotPath.LINETO]*(len(points)-1))
        self.ax.add_patch(FancyArrowPatch(path=line, arrowstyle="-|>",
                          mutation_scale=12, linewidth=1.25, color=color,
                          linestyle="--" if dashed else "-", zorder=1))

    def chain(self, prefix, x, rows, top=.85, bottom=.145, width=.445, height=.052):
        count = len(rows)
        for i, (title, body, kind) in enumerate(rows):
            y = top-(top-bottom)*i/max(1, count-1)
            key = f"{prefix}{i}"
            self.node(key, x, y, title, body, kind, width, height)
            if i:
                self.arrow(f"{prefix}{i-1}", key)

    def legend(self):
        for x, kind, text in ((.065, "frozen", "Pretrained đóng băng"),
                              (.375, "tune", "Mở khóa khi fine-tune"),
                              (.715, "head", "Head mới luôn trainable")):
            self.ax.add_patch(FancyBboxPatch((x, .886), .018, .012,
                              boxstyle="round,pad=0.001", facecolor=PALETTE[kind][0],
                              edgecolor=PALETTE[kind][1], linewidth=1))
            self.label(x+.029, .892, text, 10.2, "left")

    def save(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for text, patch in self.text_bounds:
            outer = patch.get_window_extent(renderer)
            for _ in range(25):
                inner = text.get_window_extent(renderer)
                if (inner.x0 >= outer.x0+3 and inner.x1 <= outer.x1-3
                        and inner.y0 >= outer.y0+2 and inner.y1 <= outer.y1-2):
                    break
                text.set_fontsize(text.get_fontsize()-.2)
            else:
                raise ValueError(f"Text does not fit in {self.name}: {text.get_text()}")
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        canvas = self.fig.bbox
        for text in self.ax.texts:
            box = text.get_window_extent(renderer)
            if not (box.x0 >= 0 and box.y0 >= 0 and box.x1 <= canvas.width
                    and box.y1 <= canvas.height):
                raise ValueError(f"Text outside figure: {self.name}: {text.get_text()}")
        for extension in ("png", "svg"):
            target = ROOT/f"{self.name}.{extension}"
            metadata = {"Date": None} if extension == "svg" else {}
            self.fig.savefig(target, dpi=180, facecolor="white", metadata=metadata)
            if extension == "svg":
                content = target.read_text(encoding="utf-8")
                target.write_text("\n".join(line.rstrip() for line in content.splitlines())
                                  + "\n", encoding="utf-8")
        dot = [f'digraph {self.name} {{', '  rankdir=TB;',
               '  node [shape=box, style="rounded,filled", fontname="DejaVu Sans"];']
        for key, (_, _, _, _, title, body, kind) in self.boxes.items():
            label = json.dumps(title+"\n"+body, ensure_ascii=False)
            dot.append(f'  {key} [label={label}, fillcolor="{PALETTE[kind][0]}"];')
        for start, end, dashed in self.edges:
            style = ' [style=dashed]' if dashed else ''
            dot.append(f'  {start} -> {end}{style};')
        dot.append('}')
        (ROOT/f"{self.name}.dot").write_text("\n".join(dot)+"\n", encoding="utf-8")
        plt.close(self.fig)
        print(self.name, "PNG / SVG / DOT saved")


def overview():
    d = Diagram("model_overview", "Luồng tổng quan · 3 kiến trúc pretrained",
                "CIFAR-10 → preprocessing ImageNet → backbone → head mới → 10 logits", 8.8)
    d.node("input", .5, .84, "CIFAR-10 RGB · [B,3,32,32]",
           "Resize 256 → CenterCrop 224 → Normalize ImageNet · [B,3,224,224]",
           "input", .85, .10, 14, 11.5)
    columns = [
        ("r", .19, [("ResNet18 · backbone", "Stem + layer1–4\n[B,512,7,7]", "frozen"),
                     ("Global average pooling", "Flatten → [B,512]", "op"),
                     ("fc mới", "Linear 512 → 10", "head"),
                     ("Logits [B,10]", "Chưa qua Softmax", "input")]),
        ("v", .50, [("VGG16 · features", "5 block Conv/ReLU/MaxPool\n[B,512,7,7]", "frozen"),
                     ("AvgPool → Flatten → FC1 → FC2", "25088 → 4096 → 4096\nGiữ classifier[0:6] pretrained", "frozen"),
                     ("classifier[6] mới", "Linear 4096 → 10", "head"),
                     ("Logits [B,10]", "Chưa qua Softmax", "input")]),
        ("d", .81, [("DenseNet121 · features", "4 dense blocks + 3 transitions\n[B,1024,7,7]", "frozen"),
                     ("ReLU → GAP → Flatten", "Vector [B,1024]", "op"),
                     ("classifier mới", "Linear 1024 → 10", "head"),
                     ("Logits [B,10]", "Chưa qua Softmax", "input")]),
    ]
    for prefix, x, rows in columns:
        d.chain(prefix, x, rows, .65, .16, .275, .112)
        d.path([(.5, .786), (.5, .75), (x, .75), (x, .711)])
        d.edges.append(("input", f"{prefix}0", False))
    d.label(.5, .055, "Đọc luồng chi tiết và phạm vi mở khóa ở từng hình bên dưới.", 11.5)
    d.save()


def residual_detail(d):
    d.label(.755, .837, "Bên trong một BasicBlock", 16, weight="bold")
    d.node("bi", .755, .758, "Input x", "[B,Cin,H,W]", "input", .29, .065)
    d.node("bc1", .755, .652, "Conv 3×3 → BN → ReLU", "stride s · Cout kênh", "op", .29, .075)
    d.node("bc2", .755, .539, "Conv 3×3 → BN", "stride 1 · Cout kênh", "op", .29, .075)
    d.node("badd", .755, .426, "Cộng theo phần tử", "F(x) + S(x)", "input", .29, .07)
    d.node("bout", .755, .318, "ReLU → output", "[B,Cout,Hout,Wout]", "op", .29, .07)
    for a, b in zip(("bi", "bc1", "bc2", "badd"), ("bc1", "bc2", "badd", "bout")):
        d.arrow(a, b)
    d.arrow("bi", "badd", .955, "#3F80BC", True)
    d.label(.938, .592, "Nhánh tắt S(x)", 10.5, color="#2F679B")
    d.ax.texts[-1].set_rotation(90)
    d.label(.755, .218, "S(x) = identity khi shape khớp.\nKhi đổi shape: Conv 1×1 + BatchNorm.", 11.2)
    d.label(.755, .134, "layer2–4: block đầu dùng stride 2.\nlayer1 và block còn lại dùng stride 1.", 11.2)


def resnet_flow():
    d = Diagram("model_resnet18_flow", "ResNet18 · Luồng residual và kích thước tensor",
                "IMAGENET1K_V1 · 4 stage, mỗi stage 2 BasicBlock · thay fc thành 10 lớp", 12.8)
    d.legend()
    d.chain("r", .272, [
        ("Input đã preprocessing", "[B,3,224,224]", "input"),
        ("Stem · Conv 7×7 / s2 → BN → ReLU", "[B,64,112,112]", "frozen"),
        ("MaxPool 3×3 / s2", "[B,64,56,56]", "op"),
        ("layer1 · 2 BasicBlock", "64 kênh · [B,64,56,56]", "frozen"),
        ("layer2 · 2 BasicBlock", "128 kênh · [B,128,28,28]", "frozen"),
        ("layer3 · 2 BasicBlock", "256 kênh · [B,256,14,14]", "frozen"),
        ("layer4 · 2 BasicBlock", "512 kênh · [B,512,7,7]", "tune"),
        ("AdaptiveAvgPool2d(1)", "[B,512,1,1]", "op"),
        ("Flatten từ chiều 1", "[B,512]", "op"),
        ("fc mới · Linear 512 → 10", "Logits [B,10] · không Softmax", "head"),
    ], height=.056)
    residual_detail(d)
    d.label(.5, .067, "Feature extraction: chỉ fc mới trainable. Partial fine-tuning: layer4 + fc mới.", 11.2)
    d.label(.5, .037, "Mọi BatchNorm backbone giữ running statistics cố định theo đặc tả LAB2.", 10.5)
    d.save()


def vgg_flow():
    d = Diagram("model_vgg16_flow", "VGG16 · Luồng convolution và classifier",
                "VGG16 chuẩn · 13 Conv + 3 Linear · giữ FC1/FC2 pretrained, chỉ thay Linear cuối", 12.8)
    d.legend()
    d.chain("v", .272, [
        ("Input đã preprocessing", "[B,3,224,224]", "input"),
        ("Block 1 · features[0:5]", "2 Conv 64 + ReLU → MaxPool\n[B,64,112,112]", "frozen"),
        ("Block 2 · features[5:10]", "2 Conv 128 + ReLU → MaxPool\n[B,128,56,56]", "frozen"),
        ("Block 3 · features[10:17]", "3 Conv 256 + ReLU → MaxPool\n[B,256,28,28]", "frozen"),
        ("Block 4 · features[17:24]", "3 Conv 512 + ReLU → MaxPool\n[B,512,14,14]", "frozen"),
        ("Block 5 · features[24:31]", "3 Conv 512 + ReLU → MaxPool\n[B,512,7,7]", "tune"),
        ("AdaptiveAvgPool(7×7) → Flatten", "[B,512,7,7] → [B,25088]", "op"),
        ("FC1 · classifier[0:3]", "Linear 25088→4096 → ReLU → Dropout\n[B,4096]", "frozen"),
        ("FC2 · classifier[3:6]", "Linear 4096→4096 → ReLU → Dropout\n[B,4096]", "frozen"),
        ("classifier[6] mới · Linear 4096 → 10", "Logits [B,10] · không Softmax", "head"),
    ], height=.058)
    d.label(.755, .835, "Các phép toán trong conv block", 15.5, weight="bold")
    d.node("kernel", .755, .744, "Mỗi convolution", "Kernel 3×3 · stride 1 · padding 1\nReLU sau từng Conv", "op", .36, .095)
    d.node("pool", .755, .620, "Cuối mỗi block", "MaxPool 2×2 · stride 2\nGiảm một nửa chiều cao/rộng", "op", .36, .095)
    d.arrow("kernel", "pool")
    d.label(.755, .509, "Gradient trong partial fine-tuning", 15, weight="bold")
    d.node("gv1", .755, .425, "Block 5 mở khóa", "features[24:]", "tune", .34, .065)
    d.node("gv2", .755, .324, "FC1 / FC2 đóng băng", "classifier[0:6] · giữ autograd", "frozen", .34, .065)
    d.node("gv3", .755, .223, "Head mới → loss", "classifier[6] trainable", "head", .34, .065)
    d.arrow("gv1", "gv2")
    d.arrow("gv2", "gv3")
    d.arrow("gv3", "gv1", .957, "#B47B21", True)
    d.label(.755, .128, "Không no_grad()/detach giữa block 5 và loss.\nDropout pretrained giữ eval() ở cả hai mode.", 10.9)
    d.label(.5, .058, "Feature extraction: chỉ classifier[6]. Fine-tune: features[24:] + classifier[6].", 11.2)
    d.save()


def densenet_flow():
    d = Diagram("model_densenet121_flow", "DenseNet121 · Luồng nối đặc trưng",
                "IMAGENET1K_V1 · dense blocks [6,12,24,16] · growth rate k=32 · thay classifier", 14)
    d.legend()
    d.chain("d", .272, [
        ("Input đã preprocessing", "[B,3,224,224]", "input"),
        ("Stem · Conv 7×7/s2 → BN/ReLU → Pool", "64 kênh · [B,64,56,56]", "frozen"),
        ("denseblock1 · 6 dense layers", "64 + 6×32 = 256 · [B,256,56,56]", "frozen"),
        ("transition1 · Conv 1×1 → AvgPool", "256→128 kênh · [B,128,28,28]", "frozen"),
        ("denseblock2 · 12 dense layers", "128 + 12×32 = 512 · [B,512,28,28]", "frozen"),
        ("transition2 · Conv 1×1 → AvgPool", "512→256 kênh · [B,256,14,14]", "frozen"),
        ("denseblock3 · 24 dense layers", "256 + 24×32 = 1024 · [B,1024,14,14]", "frozen"),
        ("transition3 · Conv 1×1 → AvgPool", "1024→512 kênh · [B,512,7,7]", "frozen"),
        ("denseblock4 · 16 dense layers", "512 + 16×32 = 1024 · [B,1024,7,7]", "tune"),
        ("norm5 → ReLU", "[B,1024,7,7]", "tune"),
        ("Global average pooling → Flatten", "[B,1024,1,1] → [B,1024]", "op"),
        ("classifier mới · Linear 1024 → 10", "Logits [B,10] · không Softmax", "head"),
    ], height=.0485)
    d.label(.755, .836, "Một bước trong dense block", 16, weight="bold")
    d.node("di", .755, .756, "Nối đặc trưng đã có", "x = concat(x0, …, xℓ−1)\n[B,C,H,W]", "input", .30, .082)
    d.node("dc1", .755, .629, "BN → ReLU → Conv 1×1", "Bottleneck 4k = 128 kênh", "op", .30, .078)
    d.node("dc2", .755, .505, "BN → ReLU → Conv 3×3", "Thêm k = 32 kênh mới", "op", .30, .078)
    d.node("dcat", .755, .381, "Concatenate theo chiều kênh", "concat(x, đặc trưng mới)\n[B,C+32,H,W]", "input", .30, .082)
    for a, b in zip(("di", "dc1", "dc2"), ("dc1", "dc2", "dcat")):
        d.arrow(a, b)
    d.arrow("di", "dcat", .957, "#3F80BC", True)
    d.label(.939, .568, "Giữ đặc trưng cũ", 10.5, color="#2F679B")
    d.ax.texts[-1].set_rotation(90)
    d.label(.755, .272, "Dense layer sinh 32 kênh mới. Dense block\nnối chúng với các đặc trưng đã có.", 11)
    d.label(.755, .178, "Transition: BN → ReLU → Conv 1×1\n→ AvgPool 2×2/s2; giảm kênh và phân giải.", 11)
    d.label(.5, .065, "Feature extraction: chỉ classifier. Fine-tune: denseblock4 + norm5 + classifier.", 11)
    d.label(.5, .036, "norm5 có affine trainable khi fine-tune; running statistics vẫn cố định.", 10.5)
    d.save()


def training_modes():
    d = Diagram("model_training_modes", "Hai chế độ huấn luyện · 6 cấu hình",
                "ImageNet → train head → best checkpoint → mở stage cuối và fine-tune tiếp", 13.2)
    d.label(.275, .875, "1 · Feature extraction", 17, weight="bold")
    d.label(.725, .875, "2 · Fine-tune tiếp", 17, weight="bold")
    rows = [
        (.72, "ResNet18", ("Stem + layer1–3", "layer4", "AvgPool → Flatten", "fc · 512→10")),
        (.45, "VGG16", ("features[0:24]", "features[24:]", "AvgPool → Flatten → FC1/FC2", "classifier[6] · 4096→10")),
        (.18, "DenseNet121", ("Stem → transition3", "denseblock4 + norm5", "ReLU → GAP → Flatten", "classifier · 1024→10")),
    ]
    for row, architecture, titles in rows:
        d.label(.5, row+.124, architecture, 15, weight="bold")
        for mode, x in (("fe", .275), ("ft", .725)):
            for index, title in enumerate(titles):
                if index == 3:
                    kind = "head"
                    body = "Train head 10 lớp" if mode == "fe" else "Giữ head đã học · tiếp tục train"
                elif index == 0:
                    kind = "frozen"
                    body = "ImageNet · frozen" if mode == "fe" else "Load best FE · frozen"
                elif index == 2:
                    kind = "frozen" if architecture == "VGG16" else "op"
                    body = "classifier[0:6] frozen · Dropout eval" if architecture == "VGG16" else "Các phép toán giữ nguyên"
                elif index == 1 and mode == "ft":
                    kind, body = "tune", "Trainable"
                else:
                    kind, body = "frozen", "Frozen"
                key = f"{architecture}_{mode}_{index}"
                d.node(key, x, row+.075-index*.05, title, body, kind,
                       .39, .039, 11.8, 10)
                if index:
                    d.arrow(f"{architecture}_{mode}_{index-1}", key)
        d.path([(.474, row-.075), (.505, row-.075),
                (.505, row+.075), (.526, row+.075)], "#3F80BC", True)
        d.edges.append((f"{architecture}_fe_3", f"{architecture}_ft_0", True))
        d.label(.489, row, "Load best FE", 10.2, color="#2F679B")
        d.ax.texts[-1].set_rotation(90)
    d.label(.5, .041, "Mọi BatchNorm giữ running statistics; affine của stage mở khóa vẫn nhận gradient.", 10.4)
    d.label(.5, .019, "Checkpoint chuyển toàn bộ backbone + head đã học · fine-tune không thay lại head.", 10.1)
    d.save()


def training_workflow():
    d = Diagram("training_workflow", "Luồng huấn luyện · train head → fine-tune tiếp",
                "CIFAR-10 · 45.000 train / 5.000 validation · mỗi kiến trúc có hai giai đoạn nối tiếp", 12.8)
    d.label(.272, .865, "Chuỗi cho từng kiến trúc", 16, weight="bold")
    d.chain("phase", .272, [
        ("Đọc Config_parameter, split và seed", "Dùng cùng dữ liệu / preprocessing", "input"),
        ("ImageNet → head 10 lớp", "Đóng băng backbone", "input"),
        ("Giai đoạn 1 · train head", "Chỉ head nhận cập nhật", "head"),
        ("Validation → chọn best FE", "Lưu checkpoint của run 01 / 03 / 05", "input"),
        ("Load toàn bộ best checkpoint FE", "Giữ backbone + head đã học", "input"),
        ("Mở stage cuối + tạo optimizer mới", "LR head / backbone theo cấu hình", "tune"),
        ("Giai đoạn 2 · fine-tune tiếp", "Head + stage cuối nhận cập nhật", "tune"),
        ("Validation → lưu best fine-tune", "Checkpoint riêng run 02 / 04 / 06", "head"),
        ("So sánh 6 run → CSV + một HTML offline", "reports/report_all_models.html", "input"),
    ], top=.796, bottom=.145, height=.056)
    d.label(.755, .865, "Vòng lặp epoch dùng chung", 16, weight="bold")
    steps = [
        ("model.train()", "Duy trì policy BatchNorm / Dropout", "op"),
        ("Forward → CrossEntropyLoss", "zero_grad → autocast nếu CUDA", "op"),
        ("Backward → clip → optimizer step", "AMP: unscale trước clipping", "op"),
        ("model.eval() → validation 5.000 ảnh", "inference_mode · không augmentation", "op"),
        ("Ghi metrics, thời gian, bộ nhớ", "TensorBoard + history theo epoch", "op"),
        ("Scheduler → lưu best / last → xét dừng", "Epoch cuối vẫn được lưu đầy đủ", "input"),
    ]
    d.chain("epoch", .755, steps, top=.772, bottom=.272, width=.365, height=.066)
    d.arrow("epoch5", "epoch0", .972, "#3F80BC", True)
    d.label(.95, .525, "Chưa dừng: epoch tiếp", 10.3, color="#2F679B")
    d.ax.texts[-1].set_rotation(90)
    d.node("resume", .755, .145, "Resume cùng run từ last_checkpoint.pt",
           "Khôi phục model + optimizer/scheduler/scaler\n+ epoch, RNG và trạng thái dừng sớm", "input", .38, .095)
    d.label(.5, .06, "Fine-tune là giai đoạn tiếp theo từ best FE; resume là tiếp tục một giai đoạn đang dở.", 11)
    d.label(.5, .032, "Official test chỉ dùng ở bước demo sau khi chốt mô hình bằng validation.", 10.6)
    d.save()


def demo_workflow():
    d = Diagram("demo_workflow", "Demo · test CIFAR-10 và xuất báo cáo PDF",
                "Chốt model bằng validation trước · nạp một checkpoint · test một lượt · xuất cùng bộ artifact", 11.8)
    d.label(.272, .866, "Nhánh checkpoint đã chọn", 16, weight="bold")
    d.label(.755, .866, "Nhánh dữ liệu official test", 16, weight="bold")
    d.chain("model", .272, [
        ("Global best từ validation", "checkpoints/checkpoint_best_global.pt", "input"),
        ("Đọc metadata → dựng CNN", "Architecture / mode / classes / transform_spec", "op"),
        ("Load strict → model.eval()", "pretrained=False · đủ backbone + head", "op"),
    ], top=.79, bottom=.59, height=.065)
    d.chain("data", .755, [
        ("CIFAR10(train=False)", "10.000 ảnh RGB 32×32 · class order chuẩn", "input"),
        ("Eval transform từ checkpoint", "Resize 256 → CenterCrop 224 → Normalize", "op"),
        ("DataLoader · shuffle=False", "[B,3,224,224] · drop_last=False", "op"),
    ], top=.79, bottom=.59, height=.065)
    d.node("infer", .5, .476, "inference_mode() · một lượt qua 10.000 ảnh",
           "Logits → loss / probabilities / predictions · không cập nhật trọng số", "head", .82, .072)
    for name in ("model2", "data2"):
        d.arrow(name, "infer")
    d.node("records", .272, .365, "JSON + CSV của cùng lần test",
           "sample_index · true/pred · prob_0 … prob_9", "input", .445, .065)
    d.node("figures", .755, .365, "Confusion matrix · ROC · ảnh đúng/sai",
           "Dùng cùng labels / probabilities và ảnh RGB gốc", "input", .445, .065)
    d.arrow("infer", "records")
    d.arrow("infer", "figures")
    d.node("check", .5, .252, "Đối chiếu metrics, CSV và hình",
           "10.000 hàng · support/CM = 10.000 · đúng class order / checkpoint hash", "op", .82, .070)
    d.arrow("records", "check")
    d.arrow("figures", "check")
    d.node("pdf", .5, .141, "PredictPDF.py → reports/Predict_Report.pdf",
           "Báo cáo tiếng Việt từ cùng JSON / CSV / PNG · dựng lại được không inference lại", "head", .82, .075)
    d.arrow("check", "pdf")
    d.label(.5, .058, "PDF/ảnh/CSV phản ánh official test; HTML training giữ kết quả train/validation riêng.", 10.8)
    d.label(.5, .029, "Không chọn lại model, checkpoint hoặc hyperparameters bằng kết quả test.", 10.5)
    d.save()


if __name__ == "__main__":
    overview()
    resnet_flow()
    vgg_flow()
    densenet_flow()
    training_modes()
    training_workflow()
    demo_workflow()
