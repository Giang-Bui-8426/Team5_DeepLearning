# Coursework — Dữ liệu cho Knowledge Distillation

> **Trạng thái: chờ nhóm duyệt.** Dataset, split và EDA **kế thừa nguyên** từ LAB2; tài liệu này chỉ ghi phần khác biệt do distillation. Không sao chép bảng lớp, preprocessing hay EDA — xem [LAB2/Docs/data.md](../../LAB2/Docs/data.md).

## 1. Nguồn dữ liệu

| Nguồn | Vai trò | Trạng thái trên máy này |
|---|---|---|
| **CIFAR-10** qua `torchvision.datasets.CIFAR10` | Ảnh và nhãn cứng cho student và teacher | Chưa có (`LAB2/data/prepared/` không tồn tại) |
| `LAB2/data/splits.npz` | Split train fit / validation cố định, seed 42 | Chưa có — bị loại khỏi Git |
| Checkpoint teacher `LAB2/checkpoints/checkpoint_best_global.pt` (run `02_resnet18_fine_tune`, SHA-256 `b55b143e…a7`) | Sinh soft label | Chưa có — bị loại khỏi Git |

Dùng lại cùng split với LAB2 là bắt buộc: teacher đã chọn checkpoint bằng validation này, nên student phải đánh giá trên đúng validation/test đó để so sánh hợp lệ.

## 2. Trường dữ liệu

| Trường | Kiểu / shape | Nguồn | Ghi chú |
|---|---|---|---|
| `image` | `uint8` `[32,32,3]` | CIFAR-10 | Định nghĩa đầy đủ: [LAB2 data.md §1](../../LAB2/Docs/data.md#1-dataset-và-nhãn) |
| `label` | `int` 0–9 | CIFAR-10 | Thứ tự lớp giữ nguyên như LAB2 |
| `teacher_logits` | `float32` `[10]` mỗi ảnh | Teacher (mới) | Chưa softmax; temperature áp dụng khi train, thuộc đặc tả phương pháp |

### 2.1. Hai đầu vào khác nhau

| | Teacher | Student |
|---|---|---|
| Kích thước | Resize cạnh ngắn 256 → center crop 224×224, theo [LAB2 data.md §3](../../LAB2/Docs/data.md#3-hợp-đồng-preprocessing) | Giữ nguyên 32×32 |
| Normalize | ImageNet mean/std (bắt buộc với weights pretrained) | **Thống kê CIFAR-10 trên 45.000 ảnh train fit** đã tính ở [LAB2 data.md §4.2](../../LAB2/Docs/data.md#42-màu-sắc-và-đặc-trưng-theo-lớp): mean `(0.49184, 0.48254, 0.44682)`, std `(0.24705, 0.24354, 0.26169)` |

Student train từ đầu nên normalize theo dataset, cùng tiền lệ LAB1 ([LAB1 data.md](../../LAB1/docs/data.md)). Quy định "không thay ImageNet mean/std" của LAB2 chỉ áp dụng cho mạng pretrained. Nếu student đổi sang mạng pretrained thì phải quay lại ImageNet. Không đưa ảnh 224×224 vào student.

### 2.2. Soft label tính online

Logits teacher được **tính trong lúc train**, trên chính batch đã augmentation mà student nhận, không tính trước:

- Teacher đã train với crop, flip, ColorJitter `p=0.8` và RandomErasing ([LAB2 trainning.md §2.1](../../LAB2/Docs/trainning.md#21-augmentation)); soft label tính trước trên ảnh gốc không khớp ảnh student thật sự thấy.
- Chi phí ước lượng: LAB2 forward 10.000 ảnh test mất 11.8 s, cả vòng inference 38.9 s trên RTX 3050 → thêm khoảng 1–3 phút cho 45.000 ảnh mỗi epoch. Đây là suy ra từ số đo test, chưa đo trong training.
- Hệ quả: phép biến đổi hình học áp lên ảnh 32×32 **một lần**, rồi tách hai nhánh — giữ 32×32 cho student, phóng lên 224 cho teacher — để hai mạng nhìn cùng nội dung. Thứ tự và tham số augmentation thuộc đặc tả phương pháp (mục 7).

## 3. Quy mô dự kiến

| Tập | Số ảnh | Dùng cho |
|---|---:|---|
| Train fit | 45.000 | Train student; sinh soft label |
| Validation | 5.000 | Chọn checkpoint student và cấu hình KD |
| Test | 10.000 | Đánh giá cuối, một lần |

Logits teacher sinh online nên không lưu thành file. Baseline student train thường chạy 3 seed trên cùng split này (quy trình tại [lab-description.md §4.3](lab-description.md#43-quy-trình-chốt-ngưỡng)); split không đổi giữa các seed.

**Rò rỉ dữ liệu:** không dùng ảnh test hay logits teacher trên test để train hoặc chốt ngưỡng. Teacher được fine-tune chỉ trên train fit nên logits của teacher trên train fit thường tự tin hơn trên ảnh chưa thấy; đây là đặc điểm cần ghi nhận khi phân tích, không phải leakage.

## 4. Quyền sử dụng

Đã đối chiếu nguồn gốc ngày 17/11/2026.

| Thành phần | Điều khoản | Áp dụng cho coursework |
|---|---|---|
| CIFAR-10 | [Trang dataset](https://www.cs.toronto.edu/~kriz/cifar.html) không ghi giấy phép; chỉ yêu cầu trích dẫn khi sử dụng: Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. | Bắt buộc trích dẫn báo cáo này trong tài liệu và báo cáo cuối |
| Mã nguồn torchvision | [BSD 3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | Được dùng tự do; không sửa hay phân phối lại thư viện |
| Weights `ResNet18_Weights.IMAGENET1K_V1` | [Tài liệu torchvision](https://docs.pytorch.org/vision/stable/models.html): weights "may have their own licenses or terms and conditions derived from the dataset used for training"; người dùng tự xác định quyền sử dụng | Weights train trên ImageNet; [điều khoản ImageNet](https://image-net.org/download.php) giới hạn **"non-commercial research and educational purposes"** |

**Kết luận:** coursework là mục đích học tập, phi thương mại nên nằm trong phạm vi cho phép. Điều khoản ImageNet viết cho *database*; việc nó ràng buộc trực tiếp đến weights hay không, torchvision không khẳng định. Vì vậy coi giới hạn phi thương mại là ràng buộc của cả teacher và student được distill từ teacher. Không dùng các checkpoint này cho mục đích thương mại và không phân phối lại dataset.

## 5. Phương án thay thế

| Rủi ro | Phương án |
|---|---|
| Không lấy được checkpoint teacher từ người đã chạy LAB2 | Train lại run 01 → 02 bằng `LAB2/train.py`; LAB2 ghi nhận tổng phase khoảng 87 phút trên RTX 3050. Phải kiểm tra lại test accuracy và cập nhật hash |
| Không lấy được `splits.npz` | Sinh lại bằng `LAB2/EDA.py` (stratified, seed 42), rồi đối chiếu số ảnh mỗi lớp với LAB2 trước khi dùng |
| Khoảng cách teacher–student quá nhỏ: `G < 2σ` theo [lab-description.md §4.3](lab-description.md#43-quy-trình-chốt-ngưỡng) | Chuyển sang **CIFAR-100** — cùng định dạng 32×32, 100 lớp, khó hơn. Checkpoint LAB2 chỉ có 10 lớp nên phải fine-tune teacher mới bằng pipeline LAB2 và tạo split mới |
| Forward teacher online làm training quá chậm hoặc thiếu VRAM 6 GB | Tính trước logits một lần trên ảnh không augmentation, lưu theo chỉ số trong `splits.npz` (45.000 × 10 × 4 byte ≈ 1.8 MB); ghi rõ giới hạn lệch augmentation nêu ở §2.2 |
