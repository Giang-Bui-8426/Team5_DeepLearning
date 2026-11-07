# Tiến độ thực hiện và kế hoạch hoàn thiện

## 01. Tổng quan tiến độ

Đối chiếu các file source hiện tại với tài liệu LAB1. Trạng thái **đã có code** chỉ xác nhận chức năng được triển khai trong source, không mặc nhiên xác nhận một lần chạy mới đã hoàn tất.

| Hạng mục | Tình trạng theo source | Minh chứng / phạm vi xác nhận |
|:--|:--|:--|
| Khám phá dữ liệu (EDA) | **Đã có chương trình và báo cáo trước đây** | [EDA_Report](../reports/eda_report.html) |
| Thiết kế kiến trúc | **Đã triển khai** | [ModelMLP.py](../ModelMLP.py) có Plain MLP, ResidualBlock và được lựa chọn trong [Train.py](../Train.py) |
| Huấn luyện và TensorBoard | **Đã triển khai trong source** | [Train.py](../Train.py) ghi scalar theo batch/epoch, hình ROC/confusion và flush logs |
| Resume, checkpoint, Early Stopping | **Đã cập nhật** | [Train.py](../Train.py) khôi phục optimizer/scheduler/bộ đếm; ghi epoch trước khi dừng |
| HTML huấn luyện và so sánh | **Đã tạo báo cáo** | [Report_Tranning](../reports/report_all_models.html) |
| Kiểm thử trên test và PDF | **Đã có** | Thực hiện trên [Predict.py](../Predict.py) và lưu kết quả ở [Predict_Report.pdf](../reports/Predict_Report.pdf) |

Kết quả validation đã ghi tại [evaluation.md](evaluation.md) và kết quả test tại [demo.md](demo.md) là các số liệu được tài liệu hiện có trình bày

## 02. Những điểm cần xử lý

Các mục dưới đây được xác định từ source đã đối chiếu; không tự kết luận rằng chúng đã ảnh hưởng tới các lượt train trước.

| Ưu tiên | Vấn đề | Yêu cầu kiểm tra hoặc chỉnh sửa | Điều kiện xác nhận |
|:--:|:--|:--|:--|
| **P0** | [Predict.py](../Predict.py) dùng `Normalize((0,286,), (0,353,))`, khác `Train.py` | Đổi thành `Normalize((0.2860,), (0.3530,))` và kiểm tra một batch đầu vào | Test nhận tensor `[B, 1, 28, 28]`; inference chạy hết 10.000 ảnh |
| **P1** | Source mới thay đổi dropout/weight decay/LR so với một số bảng kết quả cũ | Đối chiếu `config` và `best_metrics` trong checkpoint với từng lần chạy trước khi cập nhật [evaluation.md](evaluation.md) | Không gắn số liệu cũ cho cấu hình chưa được chạy xác nhận |
| **P1** | [Train.py](../Train.py) và [Predict.py](../Predict.py) phụ thuộc `TrainReport.py` / `PredictPDF.py` | Kiểm tra hai module có trong repository và cài đủ thư viện khi chạy | Training xuất HTML; prediction xuất PDF không báo thiếu module |
| **P1** | `Train.py` dùng đường dẫn tương đối `./data`, `./logs`, `./reports`, `./checkpoints` | Chạy từ thư mục `LAB1/` hoặc thống nhất đường dẫn theo vị trí source | Không phát sinh thư mục kết quả sai vị trí |
| **P2** | Chỉ số chia train/validation được tái tạo từ seed nhưng chưa lưu riêng trong [Train.py](../Train.py) | Lưu hoặc kiểm chứng lại danh sách chỉ số khi cần đối chiếu nhiều lần chạy | Phép chia 50.000/10.000 trùng nhau giữa các run |

> **Nguyên tắc của nhóm:** nếu sửa mã ảnh hưởng đến dữ liệu hoặc quá trình train, cần xác định checkpoint và kết quả nào phải chạy lại; không tự giữ nguyên kết luận của cấu hình cũ.

## 03. Kế hoạch kiểm chứng

### Giai đoạn A — Dữ liệu và source
- [ ] Kiểm tra mô hình trả về logits `[B, 10]` với cả sáu cấu hình; xác nhận activation thực tế trong [ModelMLP.py](../ModelMLP.py).
- [ ] Kiểm tra `TrainReport.py`, `PredictPDF.py` và các thư viện cần thiết có sẵn trong repository/môi trường chạy.

**Đầu ra:** preprocessing và kiến trúc nhất quán với checkpoint trước khi kiểm thử.

### Giai đoạn B — Huấn luyện, checkpoint và HTML

- [ ] Chạy thử resume từ `last_checkpoint.pt`; xác nhận epoch, optimizer, scheduler, `best_val_loss`, `bad_epochs` được khôi phục.
- [ ] Kiểm tra Early Stopping không làm mất log, `history.json` hoặc checkpoint của epoch cuối.
- [ ] Đối chiếu `reports/best_models.csv` với `best_metrics` của mỗi `checkpoint_best.pt`, sau đó xác nhận `checkpoint_best_global.pt` được chọn bằng validation accuracy.
- [ ] Mở `reports/report_all_models.html`, xác nhận bảng so sánh và sáu tab khi đã có lịch sử đầy đủ; thử mở offline trên máy khác.

**Đầu ra:** checkpoint và báo cáo HTML truy ngược được đúng cấu hình, epoch và kết quả validation.

### Giai đoạn C — Kiểm thử và bàn giao

- [ ] Chạy [Predict.py](../Predict.py) trên tập test chính thức sau khi sửa preprocessing.
- [ ] Kiểm tra `reports/test/test_metrics.json`, `test_predictions.csv`, `confusion_matrix.png`, `roc_ovr.png`, `prediction_examples.png`.
- [ ] Mở `reports/Predict_Report.pdf`, kiểm tra metrics, classification report, hình, tiếng Việt và số trang.
- [ ] Cập nhật [evaluation.md](evaluation.md), [demo.md](demo.md) chỉ khi có đầu ra xác thực từ phiên bản code tương ứng.
- [ ] Kiểm tra liên kết Markdown, đường dẫn source và hình ảnh trong `docs/assets/` trên GitHub/VS Code.

**Đầu ra:** kết quả test độc lập, báo cáo PDF/HTML sử dụng được và tài liệu thống nhất với source.

## 04. Tiêu chí hoàn thành bài LAB

- [ ] 6 cấu hình có checkpoint và bảng validation tương ứng
- [ ] Phân tích và đánh giá đươc sử khác biệt và hiệu quả của từng mô hình
- [ ] HTML tổng hợp và PDF dự đoán được mở, kiểm tra đầy đủ nội dung.
- [ ] Các kết quả trong tài liệu có thể truy ngược về cấu hình và file output của lần chạy tương ứng.
