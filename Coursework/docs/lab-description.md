# Coursework — Knowledge Distillation: nén CNN phân loại CIFAR-10 cho suy luận thời gian thực

> **Trạng thái: chờ nhóm duyệt.** Tài liệu sở hữu mục 1–4 (đề tài, hướng chính, người dùng, mục tiêu). Mục 6–10 (phạm vi, phương pháp, sản phẩm, đánh giá, kế hoạch) thuộc thành viên còn lại; tài liệu này không lặp các nội dung đó.

## 1. Tên đề tài

**Knowledge Distillation từ ResNet18 fine-tuned sang CNN nhỏ để phân loại ảnh CIFAR-10 thời gian thực trên CPU laptop.**

| Thành phần | Nội dung |
|---|---|
| Công nghệ / hành động | Knowledge Distillation (teacher → student) |
| Bài toán | Phân loại ảnh 10 lớp |
| Phạm vi | CIFAR-10; teacher là checkpoint đã chọn của LAB2; thiết bị đích là laptop RTX 3050 6 GB, chỉ số chính đo trên CPU của máy này |

## 2. Hướng chính

**Realtime.** Distillation nhằm giảm chi phí suy luận để mô hình đạt tốc độ thời gian thực khi không dùng GPU, trong khi giữ chất lượng dự đoán gần teacher. Tốc độ và kích thước là mục tiêu; chất lượng dự đoán là ràng buộc.

## 3. Người dùng và vấn đề

**Người dùng:** người cần chạy phân loại ảnh trên laptop phổ thông mà không phụ thuộc GPU — máy không có GPU rời, hoặc GPU đang bận cho tác vụ khác.

**Khó khăn hiện tại** — căn cứ kết quả thật của LAB2 ([evaluation.md](../../LAB2/Docs/evaluation.md)):

| Quan sát từ LAB2 | Hệ quả khi triển khai |
|---|---|
| Mô hình tốt nhất (ResNet18 FT) đạt test accuracy 92.91% với 11,181,642 tham số | Kích thước lớn so với bài toán ảnh 32×32 |
| Đầu vào phải resize lên 224×224 theo preprocessing ImageNet ([data.md §3](../../LAB2/Docs/data.md#3-hợp-đồng-preprocessing)) trong khi ảnh gốc 32×32 | Mỗi ảnh xử lý gấp 49 lần số pixel gốc |
| Trên GPU RTX 3050, forward FP32 10.000 ảnh test mất 11.789 s (batch 32), tức ~1.2 ms/ảnh | Trên GPU, teacher nhiều khả năng đã đủ nhanh — vì vậy chỉ số chính là **CPU**, nơi chi phí trên trở thành nút thắt |
| LAB2 chưa đo latency trên CPU | Chưa kết luận teacher "chậm" trên CPU trước khi đo baseline (mục 4.3) |

Train trực tiếp một CNN nhỏ trên nhãn cứng thường kém hơn mô hình lớn; distillation dùng phân phối xác suất của teacher (soft label) để thu hẹp khoảng cách này. Mức thu hẹp thực tế là giả thuyết, kiểm chứng bằng ngưỡng ở mục 4.2.

## 4. Mục tiêu

### 4.1. Đầu ra cần đạt

- Một mô hình **student** đã distill, kèm checkpoint và metadata tái lập.
- Bảng so sánh **teacher / student train thường / student KD** trên cùng split và cùng giao thức test.
- Số đo latency trên laptop RTX 3050 cho teacher và student: CPU là chỉ số chính, GPU báo cáo kèm.

### 4.2. Chỉ số và ngưỡng nghiệm thu

| Mục tiêu | Chỉ số | Ngưỡng | Trạng thái |
|---|---|---|---|
| Realtime | Latency **toàn quy trình** (ảnh 32×32 → nhãn), CPU, batch 1, p95 | ≤ **33.3 ms/ảnh** (30 FPS) | Chốt |
| Mô hình nhỏ | Tổng tham số student | ≤ 10% teacher, tức ≤ **1,118,164** | Chốt |
| KD có tác dụng | Validation accuracy student KD | ≥ μ + 2σ, với μ, σ là trung bình và độ lệch chuẩn của student train thường qua 3 seed | Công thức chốt; số điền sau baseline |
| Giữ chất lượng | Khoảng cách tới teacher, `G = acc(teacher) − μ` | Student KD thu hẹp ≥ 50% khoảng `G`: `acc(KD) ≥ μ + 0.5·G` | Công thức chốt; số điền sau baseline |

Hai ngưỡng sau xét trên **validation**, cùng giao thức chọn checkpoint của LAB2; test chỉ dùng xác nhận cuối. Accuracy và macro F1 giữ định nghĩa của LAB2 để so sánh được với teacher. GPU latency và dung lượng checkpoint chỉ báo cáo, không đặt ngưỡng.

### 4.3. Quy trình chốt ngưỡng

1. **Đo baseline trên laptop RTX 3050**, ghi tên CPU, phiên bản PyTorch và số luồng CPU:
   - Teacher, CPU, batch 1: warm-up 50 lần, đo 500 lần, `torch.inference_mode()`, cố định `torch.set_num_threads`; lấy trung vị và p95. Ghi riêng **chỉ forward** và **toàn quy trình** (gồm resize 224 của teacher).
   - Student train thường (không KD), **3 seed**, cùng kiến trúc và ngân sách train với student KD → μ, σ trên validation.
2. **Tính và khóa ngưỡng** theo bảng 4.2, ghi vào tài liệu tiến độ kèm ngày **trước khi** train student KD. Không sửa ngưỡng sau khi thấy kết quả KD.
3. **Nhánh dự phòng:** nếu `G < 2σ` — student train thường đã gần teacher — KD không đủ khoảng để chứng minh tác dụng; chuyển sang CIFAR-100 theo [data.md §5](data.md#5-phương-án-thay-thế).

Cách cài đặt phép đo latency và kiến trúc student thuộc tài liệu đánh giá/phương pháp (mục 7, 9).

## Tài liệu chi tiết

| Nội dung | Tài liệu |
|---|---|
| Dữ liệu, soft label của teacher, quyền sử dụng | [data.md](data.md) |
| Phạm vi, phương pháp, sản phẩm, đánh giá, kế hoạch (mục 6–10) | `⟨thành viên còn lại — đặt link khi có file⟩` |
