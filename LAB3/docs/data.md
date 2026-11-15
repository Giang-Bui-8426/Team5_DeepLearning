# Dữ liệu và phân tích khám phá — Phân tích cảm xúc Hugging Face

*Báo cáo dữ liệu sử dụng cho phân tích cảm xúc văn bản bằng mô hình pretrained Hugging Face*

## 1. Thư viện và Dataset

LAB3 sử dụng các thư viện chính sau:

| Thư viện | Vai trò |
|---|---|
| `transformers` | Nạp mô hình pretrained và tokenizer từ Hugging Face Hub |
| `datasets` | Tải Dataset IMDB, chia split và xử lý dữ liệu |
| `evaluate` | Tính toán metrics (accuracy, F1, precision, recall) |

Dataset chính: **IMDB** (Internet Movie Database Reviews) — tập đánh giá phim nhị phân (positive / negative) từ Hugging Face Hub.

| Thuộc tính | Giá trị |
|---|---|
| Số lượng mẫu | 25.000 train, 25.000 test (tổng 50.000) |
| Nhãn | 0 = negative, 1 = positive |
| Cân bằng lớp | 12.500 mẫu/lớp trong train, 12.500 mẫu/lớp trong test |
| Văn bản | Tiếng Anh, review phim, đã được làm sạch cơ bản |

## 2. Phân chia dữ liệu

| Tập dữ liệu | Số lượng | Mục đích sử dụng |
|---|---:|---|
| **Train fit** | 20.000 | Cập nhật trọng số mô hình khi fine-tuning |
| **Validation** | 5.000 | Theo dõi loss, metrics, Early Stopping và chọn checkpoint |
| **Test chính thức** | 25.000 | Đánh giá cuối cùng sau khi khóa lựa chọn bằng validation |

Phép chia cố định (seed = 42), được lưu lại để tái lập. Không dùng test để chọn mô hình hoặc điều chỉnh hyperparameter.

## 3. Tiền xử lý văn bản

| Bước xử lý | Train | Validation / Test |
|---|---|---|
| Tokenizer | `DistilBertTokenizer` từ `distilbert-base-uncased` | Cùng tokenizer với train |
| Truncation | `True`, `max_length=256` | `True`, cùng `max_length` |
| Padding | `True`, padding đến `max_length` hoặc longest trong batch | `True`, cùng cách làm train |
| Tensor đầu ra | `input_ids`, `attention_mask`, `labels` | `input_ids`, `attention_mask`, `labels` |
| Lowercase | Được xử lý bởi tokenizer (`uncased`) | Được xử lý bởi tokenizer (`uncased`) |

Tokenizer được nạp cùng lúc với mô hình pretrained từ Hugging Face Hub. Không tự viết regex tách từ hay stemmer thay thế tokenizer của mô hình.

## 4. Kết quả khảo sát dữ liệu (EDA)

EDA cho LAB3 bao gồm:

- **Phân phối nhãn:** Số lượng mẫu positive/negative trên train fit, validation và test (đảm bảo cân bằng hoặc ghi chép lệch lớp).
- **Độ dài văn bản:** Số token trung bình, tối đa, tối thiểu sau khi tokenize với `distilbert-base-uncased`.
- **Từ đặc trưng:** Các từ xuất hiện nhiều nhất trong từng lớp cảm xúc (tuỳ chọn, phục vụ phân tích).

Kết quả EDA được lưu tại `reports/eda_stats.json` và trình bày HTML tại `reports/eda_report.html`. Hình minh họa nằm trong `Docs/assets/`.

## 5. Phạm vi sử dụng kết quả EDA

Giữ cùng phép chia dữ liệu và tokenizer cho cả hai bài tập. Các giá trị đã quan sát ở EDA không thay thế kết quả validation hoặc test; phần đối chiếu kết quả được trình bày tại [evaluation.md](evaluation.md) và [demo.md](demo.md).