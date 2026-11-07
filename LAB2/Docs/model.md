# Practice 2 — Đặc tả mô hình và chiến lược Transfer Learning

*ResNet18 · VGG16 · DenseNet121 *

> **Mục đích tài liệu:** Quy định chính xác những mô hình mà nhóm yêu cầu AI xây dựng trong `Model.py`: nạp trọng số ImageNet, thay tầng phân loại cho Fashion-MNIST và cấu hình hai chế độ Feature Extraction / Partial Fine-tuning

## 1. Yêu cầu chung

Nhóm lựa chọn ba kiến trúc CNN đã được huấn luyện trên ImageNet để khảo sát khả năng chuyển giao đặc trưng sang bài toán phân loại 10 lớp Fashion-MNIST. Cả ba mô hình phải giữ nguyên cấu trúc backbone gốc; chỉ thay classifier đầu ra và chủ động mở khóa một phần mạng trong thí nghiệm fine-tuning.

| Quy định | Yêu cầu bắt buộc |
|---|---|
| Nguồn mô hình | `torchvision.models` sử dụng trọng số pretrained  |
| Phiên bản trọng số | Cố định `IMAGENET1K_V1` cho cả ba kiến trúc |
| Đầu vào | Tensor `float32` hoặc tensor tương thích mixed precision, shape **`[B, 3, 224, 224]`** |
| Đầu ra | 10 logits dạng `[B, 10]` không gắn `Softmax` vào cuối mô hình |
| Cách xây dựng | Mỗi lượt chạy tạo một model mới từ cùng pretrained weights, không tái sử dụng model đã fine-tune |
| Phạm vi | Không tự ý đổi backbone, thay kiến trúc bằng model nhỏ hơn hoặc train from scratch |

Đầu vào, thứ tự nhãn và phép chuẩn hóa được định nghĩa tại [`data.md`](data.md). `Model.py` chỉ chịu trách nhiệm tạo và cấu hình mô hình, không tải Fashion-MNIST hoặc tổ chức training loop.

## 2. Ba kiến trúc được lựa chọn

![Luồng xử lý của ba mô hình pretrained](assets/model_overview.png)

*Hình 1. Luồng đặc trưng của ba kiến trúc. Tất cả sử dụng cùng đầu vào và trả về `[B, 10]`, nhưng kích thước đặc trưng trước classification head khác nhau.*

| Kiến trúc | Đặc trưng cần khảo sát | Tầng phải thay | Classification head mới | Tổng tham số sau thay head¹ |
|---|---|---|---|---:|
| **ResNet18** | Kết nối tắt giúp thông tin đi qua các residual blocks | `model.fc` | `nn.Linear(512, 10)` | 11.181.642 |
| **VGG16** | Các khối convolution xếp tuần tự, classifier gốc rất lớn | `model.classifier[6]` | `nn.Linear(4096, 10)` | 134.301.514 |
| **DenseNet121** | Các tầng dùng lại feature maps thông qua dense connections | `model.classifier` | `nn.Linear(1024, 10)` | 6.964.106 |

*¹ Số tham số của các kiến trúc torchvision tiêu chuẩn sau khi thay tầng 1.000 lớp ImageNet bằng tầng 10 lớp; được kiểm tra bằng `sum(p.numel() for p in model.parameters())`. Đây là **quy mô mô hình**, không phải số tham số được phép cập nhật hoặc dung lượng RAM thực tế.*

### 2.1. ResNet18

Đường truyền chính gồm stem, bốn stage `layer1`–`layer4`, global average pooling và `fc`. Residual blocks có nhánh cộng tắt:

$$
\mathbf{y} = \mathbf{x} + F(\mathbf{x})
$$

Nhóm giữ nguyên toàn bộ backbone pretrained và thay **duy nhất** `fc` để tạo 10 logits. Với input `[B, 3, 224, 224]`, vector ngay trước `fc` có dạng `[B, 512]`.

### 2.2. VGG16

VGG16 sử dụng các block convolution tuần tự, sau đó đi qua `avgpool` và các tầng fully-connected trong `classifier`. Nhóm sử dụng `vgg16` tiêu chuẩn, không phải `vgg16_bn`, giữ nguyên `classifier[0]` đến `classifier[5]` và chỉ thay `classifier[6]` từ 1.000 đầu ra thành 10 đầu ra. Đặc trưng đầu vào tầng cuối có dạng `[B, 4096]`.

### 2.3. DenseNet121

DenseNet121 truyền các feature maps từ những tầng trước đến tầng sau bằng phép nối kênh:

$$
\mathbf{x}_{\ell} = H_{\ell}([\mathbf{x}_0,\mathbf{x}_1,\ldots,\mathbf{x}_{\ell-1}])
$$

Nhóm giữ nguyên `model.features` pretrained và thay `model.classifier` thành `nn.Linear(1024, 10)`. Sau global average pooling, vector đặc trưng có dạng `[B, 1024]`.

## 3. Hai chiến lược cần triển khai

Mỗi kiến trúc phải được chạy độc lập với hai chế độ sau

![So sánh Feature Extraction và Partial Fine-tuning](assets/model_training_modes.png)

*Hình 2. Màu xám: tham số được đóng băng*

*màu vàng: tầng cuối backbone được mở khóa*

*màu xanh: classification head mới luôn được huấn luyện.*

| Thành phần | `feature_extract` | `fine_tune` |
|---|---|---|
| Backbone giai đoạn đầu | Đóng băng | Đóng băng |
| Backbone giai đoạn cuối | Đóng băng | Mở khóa theo bảng dưới |
| Classification head mới | Trainable| Trainable |
| Trọng số khởi tạo | Pretrained ImageNet | pretrained ImageNet |

Phạm vi mở khóa trong chế độ `fine_tune` được chốt như sau:

| Kiến trúc | Các module được phép cập nhật khi fine-tune | Những phần còn lại |
|---|---|---|
| ResNet18 | `model.layer4` và `model.fc` mới | Giữ nguyên |
| VGG16 | `model.features[24:]` và `model.classifier[6]` mới | Giữ nguyên các block convolution đầu và classifier pretrained trước tầng cuối |
| DenseNet121 | `model.features.denseblock4`, `model.features.norm5` và `model.classifier` mới | Giữ nguyên |

**Quy tắc bắt buộc khi đóng băng:**

- Đặt `requires_grad=False` cho **toàn bộ tham số pretrained trước**, sau đó thay head mới và chỉ mở khóa các module có tên trong bảng. Không chỉ freeze backbone rồi vô tình để những tầng fully-connected pretrained của VGG16 tiếp tục cập nhật.
- `requires_grad=False` **không tự đóng băng running mean/variance của BatchNorm**. Để hành vi nhất quán với batch size nhỏ, nhóm chọn giữ **mọi BatchNorm của backbone ở chế độ `eval()`** trong cả hai chiến lược; các tham số affine thuộc module đã mở khóa vẫn có thể nhận gradient. Sau mỗi lần gọi `model.train()`, cơ chế này phải được áp dụng lại.
- Với VGG16, `Dropout` nằm trong phần `classifier` pretrained đã đóng băng phải ở chế độ `eval()` để đường truyền đặc trưng cố định, trong khi head mới vẫn train bình thường.
- **Không gọi `model.eval()` cho toàn bộ model trong training.** Việc đánh giá trên validation/test mới chuyển toàn bộ model sang `eval()` theo quy trình ở [`trainning.md`](trainning.md).

Trong chế độ `feature_extract`, chỉ có head mới nhận gradient. Ở `fine_tune`, cả head và stage cuối backbone nhận gradient; các nhóm tham số này phải được nhận diện riêng để training có thể gán learning rate phù hợp mà không làm thay đổi lựa chọn kiến trúc.

## 4. Danh sách 6 thí nghiệm cốt lõi

Tên cấu hình phải ổn định xuyên suốt source, checkpoint, TensorBoard và báo cáo.

| Run ID | Backbone | Chế độ | Phần được huấn luyện |
|---|---|---|---|
| `01_resnet18_feature_extract` | ResNet18 | Feature extraction | `fc` mới |
| `02_resnet18_fine_tune` | ResNet18 | Fine-tuning | `layer4` + `fc` mới |
| `03_vgg16_feature_extract` | VGG16 | Feature extraction | `classifier[6]` mới |
| `04_vgg16_fine_tune` | VGG16 | Fine-tuning | `features[24:]` + `classifier[6]` mới |
| `05_densenet121_feature_extract` | DenseNet121 | Feature extraction | `classifier` mới |
| `06_densenet121_fine_tune` | DenseNet121 | Fine-tuning | `denseblock4` + `norm5` + `classifier` mới |

Đây là **sáu thí nghiệm bắt buộc**, không phải sáu kiến trúc khác nhau. Thí nghiệm bổ sung về learning rate, batch size hoặc mức độ unfreeze (nếu có) phải có ID riêng và được mô tả tại [`trainning.md`](trainning.md).


**Quy định triển khai:**

1. Dùng API `weights=...`, **không dùng** `pretrained=True`. Cố định đúng các enum `ResNet18_Weights.IMAGENET1K_V1`, `VGG16_Weights.IMAGENET1K_V1` và `DenseNet121_Weights.IMAGENET1K_V1`; không dùng weights của VGG16 biến thể `IMAGENET1K_FEATURES`.
2. Kiểm tra giá trị `architecture`, `mode`, `num_classes` và báo lỗi rõ ràng nếu không hợp lệ; với bài LAB này `num_classes=10`.
3. Sau khi thay head và cấu hình freeze, cho phép kiểm tra `total_params` và `trainable_params`; không hard-code các giá trị này vào kết quả xuất ra.
4. Phải có cách lấy riêng hai nhóm **trainable head** và **trainable backbone**, để `Train.py` xây dựng optimizer đúng. Không đưa các tham số `requires_grad=False` vào danh sách cần tối ưu.
5. Mỗi lần gọi `build_model()` phải khởi tạo một model độc lập. Việc tái lập trọng số head mới bằng seed sẽ do phần điều phối thí nghiệm quản lý.
6. Khi không tải được pretrained weights (mạng lỗi hoặc chưa có cache), báo lỗi và hướng dẫn chuẩn bị weights; **không tự fallback sang `weights=None`** rồi tiếp tục báo cáo là transfer learning.

Hợp đồng đầu ra quan trọng nhất:

```python
x = torch.randn(2, 3, 224, 224)
logits = model(x)       # [2, 10] — chưa qua Softmax
```

`CrossEntropyLoss` nhận trực tiếp `logits`. Việc tính loss, optimizer, scheduler, checkpoint và đánh giá thuộc tài liệu khác; không triển khai trong `Model.py`.

## 6. Tiêu chí nghiệm thu phần mô hình

AI chỉ được xem `Model.py` là hoàn thành khi toàn bộ điều kiện sau được kiểm chứng:

- [ ] Cả ba model tải đúng **ImageNet-pretrained weights `IMAGENET1K_V1`**, không phải mạng khởi tạo ngẫu nhiên.
- [ ] Thay đúng `fc`, `classifier[6]` và `classifier` tương ứng; cả ba trả về shape `[B, 10]` với batch size bất kỳ phù hợp bộ nhớ.
- [ ] Với `feature_extract`, **chỉ head mới** có `requires_grad=True`; các tham số gốc đều được đóng băng.
- [ ] Với `fine_tune`, **chỉ head và stage cuối được chỉ định** có `requires_grad=True`; những tầng khác giữ nguyên.
- [ ] Các BatchNorm của backbone giữ running statistics cố định khi training; các Dropout pretrained đã đóng băng trong VGG16 không tạo nhiễu ngẫu nhiên.
- [ ] Cùng một ảnh có thể chạy forward ở cả 6 cấu hình; output hữu hạn, không có `NaN` hoặc `Inf`.
- [ ] Kiểm tra backward trên một batch: head có gradient ở cả hai chế độ; stage cuối có gradient **chỉ trong `fine_tune`**; tham số đóng băng không có gradient.
- [ ] In được tổng số tham số, số tham số trainable và danh sách module được mở khóa; kết quả khớp với cấu hình.
- [ ] Kiểm tra `train()/eval()` chuyển chế độ đúng, đặc biệt với BatchNorm/Dropout thuộc phần frozen.
- [ ] Không chỉnh sửa pipeline đầu vào đã chốt trong [`data.md`](data.md) hoặc tự thay kiến trúc vì giới hạn phần cứng.

**Đầu ra yêu cầu:** `Model.py` chứa hàm tạo và cấu hình model; sáu run ID tương ứng ở trên; một bài kiểm tra tối thiểu về forward, freeze/unfreeze và gradient. Kết quả accuracy hay tốc độ không được dự đoán trước trong tài liệu này; chúng sẽ được thu thập và phân tích tại [`evaluation.md`](evaluation.md).

---

*Tham chiếu API chính thức: [ResNet18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) · [VGG16](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html) · [DenseNet121](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html).*
