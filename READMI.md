# Deep Learning

Repository lưu trữ source code, tài liệu và kết quả thực nghiệm của các bài LAB môn Deep Learning.

## Các bài LAB

| Bài LAB | Nội dung | Tài liệu |
|---|---|---|
| **LAB1 — MLP** | Phân loại Fashion-MNIST, huấn luyện và so sánh 6 cấu hình MLP | [LAB1](LAB1/docs/lab-description.md) |
| **LAB2 — Transfer Learning** | Feature extraction và fine-tuning các mô hình CNN pretrained | [LAB2](LAB2/) |

## Cấu trúc repository

```text
Deep-Learning/
├── .gitignore
├── .gitattributes
├── README.md
├── requirements.txt           # Danh sách thư viện (khi được bổ sung)
├── LAB1/
│   ├── EDA.py                 # Phân tích dữ liệu
│   ├── ModelMLP.py            # Định nghĩa kiến trúc MLP
│   ├── Train.py               # Huấn luyện và đánh giá validation
│   ├── TrainReport.py         # Xuất báo cáo huấn luyện
│   ├── Predict.py             # Kiểm thử mô hình trên tập test
│   ├── PredictPDF.py          # Xuất báo cáo kiểm thử PDF
│   ├── checkpoints/           # Checkpoint best của từng mô hình và best global
│   ├── reports/               # EDA HTML, HTML tổng hợp, biểu đồ, CSV, JSON, PDF test
│   │   └── test/              # Metrics, dự đoán và hình đánh giá trên tập test
│   └── docs/                  # Tài liệu Markdown và hình minh họa
└── LAB2/                      # Source, tài liệu và kết quả của LAB2
```

Các thư mục `data/`, `logs/`, checkpoint `last_checkpoint.pt` và báo cáo HTML riêng từng mô hình được loại khỏi Git theo [`.gitignore`](.gitignore). Chỉ những file đã tạo và commit mới xuất hiện trên GitHub.

## Tài liệu và báo cáo LAB1

| Nội dung | Liên kết |
|---|---|
| Tổng quan LAB1 | [lab-description.md](LAB1/docs/lab-description.md) |
| Dữ liệu và EDA | [data.md](LAB1/docs/data.md) · [eda_report.html](LAB1/reports/eda_report.html) |
| Kiến trúc và huấn luyện | [model.md](LAB1/docs/model.md) · [trainning.md](LAB1/docs/trainning.md) |
| Đánh giá và so sánh 6 mô hình | [evaluation.md](LAB1/docs/evaluation.md) · [report_all_models.html](LAB1/reports/report_all_models.html) |
| Demo và kết quả test | [demo.md](LAB1/docs/demo.md) · [Predict_Report.pdf](LAB1/reports/Predict_Report.pdf) |
| Tiến độ thực hiện | [plan-progression.md](LAB1/docs/plan-progression.md) |
