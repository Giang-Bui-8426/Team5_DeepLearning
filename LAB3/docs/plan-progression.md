# Tiến độ thực hiện và kế hoạch hoàn thiện — LAB3

## 01. Tổng quan tiến độ

Đối chiếu các file source hiện tại với tài liệu LAB3. Trạng thái **đã có code** chỉ xác nhận chức năng được triển khai trong source, không mặc nhiên xác nhận một lần chạy mới đã hoàn tất.

| Hạng mục | Tình trạng theo source | Minh chứng / phạm vi xác nhận |
|:---|:---|:---|
| Khảo sát Hugging Face Hub và Dataset IMDB | Chưa triển khai | — |
| Cài đặt thư viện transformers/datasets/evaluate | Chưa triển khai | — |
| Tiền xử lý và tokenize văn bản (distilbert-base-uncased) | Chưa triển khai | — |
| Bài 1: Suy luận cảm xúc pretrained | Chưa triển khai | — |
| Bài 2: Fine-tuning distilbert-base-uncased trên IMDB | Chưa triển khai | — |
| Validation, checkpoint, Early Stopping | Chưa triển khai | — |
| HTML huấn luyện và so sánh | Chưa tạo báo cáo | — |
| Kiểm thử trên test và báo cáo HTML | Chưa có | — |

## 02. Những điểm cần xử lý

Các mục dưới đây được xác định từ đặc tả LAB3; không tự kết luận rằng chúng đã ảnh hưởng tới các lượt train trước.

| Ưu tiên | Vấn đề | Yêu cầu kiểm tra hoặc chỉnh sửa | Điều kiện xác nhận |
|:---:|:---|:---|:---|
| P0 | Cài đặt thư viện `transformers`, `datasets`, `evaluate` | Kiểm tra `requirements.txt` đã bổ sung các thư viện này chưa | `pip install -r requirements.txt` thành công |
| P0 | Tải Dataset IMDB và phân chia train/validation/test | Xác định Dataset IMDB từ Hugging Face Hub, chia 20k/5k/25k | Dataset tải được, có 2 nhãn rõ ràng |
| P1 | Tokenizer và preprocessing khớp với `distilbert-base-uncased` | Kiểm tra `input_ids`, `attention_mask`, `labels` có đúng shape `[B, 256]` | DataLoader trả batch hợp lệ |
| P1 | `Predict.py` và `Train.py` sử dụng đường dẫn tương đối | Chạy từ thư mục `LAB3/` hoặc thống nhất đường dẫn theo vị trí source | Không phát sinh thư mục kết quả sai vị trí |
| P2 | Báo cáo HTML tổng hợp và báo cáo test | Xuất `report_all_models.html` và `Predict_Report.html` | File HTML mở được ngoại tuyến, đủ bảng và hình |

> **Nguyên tắc của nhóm:** nếu sửa mã ảnh hưởng đến dữ liệu hoặc quá trình train, cần xác định checkpoint và kết quả nào phải chạy lại; không tự giữ nguyên kết luận của cấu hình cũ.

## 03. Kế hoạch kiểm chứng

### Giai đoạn A — Dữ liệu và source
- [ ] Kiểm tra mô hình trả về logits `[B, 2]` với tokenizer `distilbert-base-uncased`.
- [ ] Kiểm tra `TrainReport.py`, `PredictPDF.py` và các thư viện cần thiết có sẵn trong môi trường chạy.

**Đầu ra:** preprocessing và kiến trúc nhất quán với checkpoint trước khi kiểm thử.

### Giai đoạn B — Huấn luyện, checkpoint và HTML
- [ ] Chạy thử resume từ `last_checkpoint.pt`; xác nhận epoch, optimizer, scheduler được khôi phục.
- [ ] Kiểm tra Early Stopping không làm mất log, `history.json` hoặc checkpoint của epoch cuối.
- [ ] Đối chiếu `reports/best_models.csv` với `best_metrics` của mỗi `checkpoint_best.pt`.
- [ ] Mở `reports/report_all_models.html`, xác nhận bảng so sánh khi đã có lịch sử đầy đủ; thử mở offline trên máy khác.

**Đầu ra:** checkpoint và báo cáo HTML truy ngược được đúng cấu hình, epoch và kết quả validation.

### Giai đoạn C — Kiểm thử và bàn giao
- [ ] Chạy `Predict.py` trên tập test IMDB chính thức.
- [ ] Kiểm tra `reports/test/test_metrics.json`, `test_predictions.csv`, `confusion_matrix.png`.
- [ ] Mở `reports/Predict_Report.html`, kiểm tra metrics, classification report, hình, tiếng Việt.
- [ ] Cập nhật [evaluation.md](evaluation.md), [demo.md](demo.md) chỉ khi có đầu ra xác thực từ phiên bản code tương ứng.
- [ ] Kiểm tra liên kết Markdown, đường dẫn source và hình ảnh trong `docs/assets/`.

**Đầu ra:** kết quả test độc lập, báo cáo HTML/PDF sử dụng được và tài liệu thống nhất với source.

## 04. Tiêu chí hoàn thành bài LAB

- [ ] 2 bài tập có checkpoint và bảng validation tương ứng (Bài 1 không cần train).
- [ ] Phân tích và đánh giá được sự khác biệt và hiệu quả của từng bài tập.
- [ ] HTML tổng hợp và HTML dự đoán được mở, kiểm tra đầy đủ nội dung.
- [ ] Các kết quả trong tài liệu có thể truy ngược về cấu hình và file output của lần chạy tương ứng.