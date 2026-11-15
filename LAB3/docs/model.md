# Thiết kế mô hình — Phân tích cảm xúc với Hugging Face

## 1. Hướng thiết kế

Nhóm sử dụng mô hình ngôn ngữ pretrained `distilbert-base-uncased` từ Hugging Face Hub để phân loại cảm xúc văn bản. Mô hình bao gồm backbone Transformer (DistilBERT) và đầu ra phân loại (classification head).

| Thành phần | Mô tả |
|---|---|
| Backbone | `distilbert-base-uncased` (6 lớp Transformer, 66M tham số) |
| Tokenizer | `DistilBertTokenizer` (vocab từ BERT, uncased) |
| Classification head | `nn.Linear(hidden_size, 2)` — 2 logits cho negative/positive |
| Đầu ra | Logits `[B, 2]` cho phân loại nhị phân |
| Softmax | Không đặt trong `forward`; xác suất và nhãn dự đoán xử lý ở bước inference |

Source tham khảo: lớp `Model` trong LAB3/[model.py](../../model.py) bọc `AutoModelForSequenceClassification` từ `transformers`.

## 2. Luồng xử lý của mô hình pretrained

```
Văn bản đầu vào → Tokenizer (input_ids, attention_mask) → DistilBERT backbone → [CLS] hidden state → Classification head → Logits [B, 2] → Softmax → Nhãn dự đoán
```

Các bước chính:
1. **Tokenizer:** Chuyển văn bản thô thành `input_ids` và `attention_mask`, áp dụng truncation (`max_length=256`) và padding.
2. **Mô hình:** Đưa tensor qua DistilBERT backbone, lấy hidden state của token `[CLS]` (vector biểu diễn câu), chiếu qua Linear thành 2 logits.
3. **Inference:** Áp dụng Softmax lên logits để lấy xác suất, chọn `argmax` làm nhãn dự đoán (0 = negative, 1 = positive).

## 3. Hai chế độ sử dụng mô hình

| Chế độ | Mô tả | Bài tập |
|---|---|---|
| **Inference (pipeline)** | Chỉ nạp weight pretrained, chạy suy luận trên câu văn mẫu, không cập nhật trọng số | Bài 1 |
| **Fine-tuning** | Mở Gradient cho toàn bộ mô hình, cập nhật trọng số trên Dataset IMDB nhị phân | Bài 2 |

Việc nạp pretrained phải dùng đúng `distilbert-base-uncased` từ Hugging Face Hub. Khi cache/tải weight lỗi phải báo lỗi, không tiếp tục bằng trọng số ngẫu nhiên.

## 4. Tham số mô hình

| Tham số | Giá trị |
|---|---|
| `model_name_or_path` | `distilbert-base-uncased` |
| `num_labels` | 2 (negative, positive) |
| `hidden_size` | 768 |
| `max_length` | 256 |
| `dropout` | 0.1 (mặc định của DistilBERT) |

## 5. Kiểm tra tính nhất quán của kiến trúc

- [ ] Khởi tạo mô hình từ `AutoModelForSequenceClassification` với `num_labels=2`, forward với `[2, 256]` trả về logits `[2, 2]`.
- [ ] Xác nhận tokenizer tạo ra `input_ids` và `attention_mask` cùng shape `[B, 256]`, không có giá trị `None`.
- [ ] Đối chiếu `config` lưu trong checkpoint với source và kết quả tại [evaluation.md](evaluation.md) trước khi nhận xét sự khác nhau giữa các mô hình.
- [ ] Nạp lại checkpoint đầy đủ bằng `strict=True` tái tạo đúng kiến trúc và đầu ra.

Các cấu hình có thể thay đổi nhiều tham số cùng lúc. Phần so sánh kết quả thuộc [evaluation.md](evaluation.md), không mặc nhiên quy mọi chênh lệch cho một thay đổi kiến trúc riêng lẻ.