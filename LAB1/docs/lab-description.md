# LAB 1 · MLP

### Huấn luyện và so sánh các kiến trúc Multi-Layer Perceptron

> **Tổng quan dự án**  
> Xây dựng quy trình phân loại Fashion-MNIST bằng mạng MLP. Tiến hành phân tích dữ liệu, huấn luyện 6 cấu hình, đánh giá trên validation và lựa chọn checkpoint để kiểm thử trên tập test

## 1. Mục tiêu bài LAB

> Thay đổi, trải nghiệm các hyperparameter và network layer để huấn luyện những mô hình khác nhau về độ rộng, độ sâu, cách chuẩn hóa và các yếu tố ảnh hưởng đến chất lượng dự đoán. Quan sát khả năng hội tụ, mức độ ổn định và sự đánh đổi giữa các cấu hình.

Phạm vi thực nghiệm:
- Thống nhất dữ liệu và quy trình đánh giá giữa 6 cấu hình, chi tiết tại [data.md](data.md) và [trainning.md](trainning.md).
- Chọn checkpoint bằng validation accuracy. Kết quả trình bày tại [evaluation.md](evaluation.md) và [demo.md](demo.md).

## 2. Quy trình thực hiện

```mermaid
flowchart LR
    A[Khảo sát Fashion-MNIST] --> B[Chia dữ liệu và tiền xử lý]
    B --> C[Huấn luyện 6 cấu hình MLP]
    C --> D[Validation và theo dõi TensorBoard]
    D --> E[Chọn best checkpoint]
    E --> F[Kiểm thử trên tập test chính thức]
    D --> G[Báo cáo HTML và so sánh mô hình]
    F --> H[Báo cáo kết quả dự đoán]
```

Báo cáo EDA được trình bày ở  [EDA_report.html](../reports/EDA_report.html). Quy trình train và chọn checkpoint nằm trong [Train.py](../Train.py), phần demo và kết quả demo được trình bày ở [Predict.py](../Predict.py) và [Predict_Report.pdf](../reports/Predict_Report.pdf) .

## 3. Phạm vi các mô hình

Các thí nghiệm được định nghĩa trong [Train.py](../Train.py) và cùng sử dụng lớp `MLP` trong [ModelMLP.py](../ModelMLP.py).

| Thí nghiệm | Hướng so sánh |
|---|---|
| **01 · Baseline MLP** | Kiến trúc cơ sở làm mốc tham chiếu |
| **02 · Wide MLP** | Tăng số neuron ở các tầng ẩn |
| **03 · Deep MLP** | Tăng số tầng fully-connected |
| **04 · BatchNorm MLP** | Bổ sung Batch Normalization |
| **05 · Medium MLP** | Kết hợp mạng nhiều tầng và rộng, BatchNorm và cấu hình tối ưu riêng |
| **06 · Residual MLP** | Sử dụng LayerNorm và residual/skip connections |

Chi tiết về kiến trúc, tham số và các điểm cần đối chiếu với source đặt tại [model.md](model.md) và [trainning.md](trainning.md). Không quy mọi chênh lệch kết quả cho riêng kiến trúc khi một cấu hình thay đổi đồng thời nhiều hyperparameter.

## 4. Source và tài liệu liên quan

| Thành phần | Source | Tài liệu chuyên trách |
|---|---|---|
| Dữ liệu và EDA | [EDA.py](../EDA.py), [Train.py](../Train.py) | [data.md](data.md) |
| Xây dựng MLP | [ModelMLP.py](../ModelMLP.py) | [model.md](model.md) |
| Huấn luyện, TensorBoard, checkpoint | [Train.py](../Train.py) | [trainning.md](trainning.md) |
| Ghi lịch sử và báo cáo HTML | [Train.py](../Train.py), [TrainReport.py](../TrainReport.py) | [trainning.md](trainning.md), [evaluation.md](evaluation.md) |
| Đánh giá và so sánh sáu mô hình | [Train.py](../Train.py), [TrainReport.py](../TrainReport.py) | [evaluation.md](evaluation.md) |
| Dự đoán và xuất báo cáo PDF | [Predict.py](../Predict.py), [PredictPDF.py](../PredictPDF.py) | [demo.md](demo.md) |
| Kiểm tra tính nhất quán và tiến độ | — | [plan-progression.md](plan-progression.md) |

Phần cài đặt, hướng dẫn chạy và cấu trúc repository được trình bày trong [README.md](../README.md); không lặp lại tại đây.
