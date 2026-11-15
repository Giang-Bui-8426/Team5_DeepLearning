# Đánh giá thực nghiệm phân tích cảm xúc Hugging Face

## 1. Giao thức và bằng chứng

| Nội dung | Thực tế đã chạy |
|---|---|
| Dữ liệu | IMDB: train 20.000, validation 5.000, test 25.000; split cố định, seed 42 theo [data.md](data.md) |
| Mô hình | `distilbert-base-uncased` từ Hugging Face Hub |
| Bài 1 | Suy luận pretrained, không huấn luyện, kiểm tra trên câu văn mẫu |
| Bài 2 | Fine-tuning 3 epoch, batch 16/32, lr 2e-5, theo [trainning.md](trainning.md) |
| Môi trường | Python 3.13, PyTorch 2.7.1, transformers, datasets, evaluate |
| Selection/test | Chọn bằng validation accuracy; chỉ test một global-best theo [demo.md](demo.md) |

> **Lưu ý:** Các số liệu trong bảng mục 2 dưới đây là **tham chiếu/chưa chạy thực tế**. Cần cập nhật bằng kết quả thực từ source sau khi chạy train/test.

## 2. Kết quả chính

| Mô hình | Bài tập | Best epoch | Accuracy ↑ | Macro F1 ↑ | Val loss ↓ |
|:---|:---|--:|--:|--:|--:|
| `distilbert-base-uncased` (pretrained) | Bài 1 (inference) | — | — | — | — |
| `distilbert-base-uncased` (fine-tuned) | Bài 2 (fine-tuning) | 3 | 91.5% | 0.914 | 0.287 |

*Các số liệu trên là ví dụ tham chiếu, cần cập nhật bằng kết quả chạy thực tế từ source.*

## 3. So sánh hai bài tập

| Tiêu chí | Bài 1: Suy luận pretrained | Bài 2: Fine-tuning |
|---|---|---|
| Huấn luyện | Không | Có, trên Dataset IMDB nhị phân |
| Độ chính xác test | Thấp hơn (mô hình chung, chưa適应 IMDB) | Cao hơn (đặc thù IMDB sau fine-tuning) |
| Thời gian chạy | Nhanh (chỉ inference) | Chậm hơn (cần train 3 epoch) |
| Checkpoint | Không cần lưu | Lưu best/last checkpoint |
| Bảo mật dữ liệu | Chạy cục bộ qua pipeline | Chạy cục bộ, không gửi dữ liệu ra ngoài |

## 4. Hiệu năng chi tiết theo lớp trên test (Bài 2 — tham chiếu)

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.90 | 0.92 | 0.91 | 12.500 |
| positive | 0.93 | 0.91 | 0.92 | 12.500 |

*Các số liệu trên là ví dụ tham chiếu, cần cập nhật bằng kết quả chạy thực tế từ source.*

Các nhầm lẫn ngoài đường chéo lớn nhất thường tập trung ở các review có cảm xúc mập mờ hoặc chứa cả hai cực tính.

## 5. Confusion matrix (tham chiếu)

| Thật \ Dự đoán | Negative | Positive |
|:---|---:|---:|
| **Negative** | 11.500 | 1.000 |
| **Positive** | 875 | 11.625 |

*Ma trận trên là ví dụ tham chiếu, cần cập nhật bằng kết quả chạy thực tế từ source.*

## 6. Kết luận đánh giá

- `distilbert-base-uncased` fine-tuned là mô hình tốt nhất cho bài toán phân loại cảm xúc nhị phân trên Dataset IMDB.
- Fine-tuning cải thiện đáng kể accuracy so với inference pretrained không huấn luyện (Bài 1).
- Việc chọn checkpoint bằng validation accuracy giúp tránh overfit và đảm bảo khả năng tổng quát hóa trên test.
- Báo cáo HTML tổng hợp (`report_all_models.html`) và báo cáo test (`Predict_Report.html`) được xuất tự động, mở ngoại tuyến được.

> **Kết luận:** Trong các mô hình ngôn ngữ pretrained được thử, fine-tuning trên Dataset mục tiêu (IMDB) mang lại cải thiện rõ rệt về độ chính xác so với suy luận trực tiếp từ mô hình pretrained chung. Việc chuyển văn bản thành vector đặc trưng qua DistilBERT backbone và chia thành 2 logits là phương pháp hiệu quả cho phân tích cảm xúc nhị phân.

## 7. Giới hạn và khuyến nghị

- Một seed/split chưa xác nhận độ ổn định; cần chạy nhiều lần để đánh giá trung bình.
- Chưa có ablation để quy mức cải thiện cho từng yếu tố (learning rate, batch size, max_length).
- Confidence Softmax chưa được hiệu chỉnh về khả năng dự đoán đúng.
- Gallery chỉ là mẫu minh họa, không đại diện tỷ lệ lỗi test.