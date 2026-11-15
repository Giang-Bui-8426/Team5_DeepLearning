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
├── .gitattributes
├── READMI.md
├── requirements.txt           # Thư viện dùng cho LAB2
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
└── LAB2/
    ├── EDA.py                 # Phân tích dữ liệu
    ├── model.py               # CNN pretrained, head và freeze policy
    ├── data.py                # Dataset, split và preprocessing dùng chung
    ├── metrics.py             # Metrics và đọc/ghi artifact dùng chung
    ├── train.py               # Sáu run, checkpoint/resume và TensorBoard
    ├── TrainReport.py         # History và một báo cáo HTML tổng hợp
    ├── Predict.py             # Test checkpoint đã chọn, xuất JSON/CSV/PNG
    ├── PredictPDF.py          # Dựng PDF từ bộ kết quả đã lưu
    ├── Evaluate.py            # Đối chiếu checkpoint/validation, xuất CSV và hình đánh giá
    ├── tests.py               # Kiểm tra chức năng và tái lập resume
    ├── Docs/                  # Đặc tả và hình minh họa
    ├── data/                  # CIFAR-10 và splits.npz
    ├── checkpoints/           # Sinh khi train: best/last theo run, global best
    ├── logs/                  # Sinh khi train: TensorBoard theo run
    ├── reports/               # EDA, history, CSV, HTML và PDF
    │   └── test/              # Sinh khi test: metrics, dự đoán và hình
    ├── smoke/                 # Output kiểm tra chức năng, tách khỏi thực nghiệm
    └── archives/              # Giữ output lượt bị ngắt khi thay đổi cấu hình
```

Trong LAB2, dữ liệu, logs, smoke và last checkpoint được loại khỏi Git theo [LAB2/.gitignore](LAB2/.gitignore). Chỉ những file đã tạo và commit mới xuất hiện trên GitHub.

## Chạy LAB2

Đặc tả tại [LAB2/Docs](LAB2/Docs/lab-description.md), thư viện tại [requirements.txt](requirements.txt). Dữ liệu cần có ở `LAB2/data/prepared`, split và output EDA theo [data.md](LAB2/Docs/data.md).

Chạy từ gốc repository:

```powershell
python LAB2/tests.py
python LAB2/train.py --run all --device auto
python LAB2/Predict.py --device auto
python LAB2/Evaluate.py --device auto
```

Training nạp weights ImageNet; torchvision tải weights nếu cache chưa có. `train.py --smoke` dùng khởi tạo ngẫu nhiên, ít ảnh train/validation và output riêng để kiểm tra chức năng. Test chính thức chỉ chạy sau khi đủ sáu run hợp lệ và có `reports/selection.json`.

Kết quả LAB2 đã chạy: [phân tích thực nghiệm](LAB2/Docs/evaluation.md) · [HTML training](LAB2/reports/report_all_models.html) · [PDF test](LAB2/reports/Predict_Report.pdf). Trạng thái và bằng chứng nghiệm thu tại [plan-progress](LAB2/Docs/plan-progress).

Dựng lại báo cáo từ artifact đã lưu:

```powershell
python LAB2/TrainReport.py LAB2/reports
python LAB2/PredictPDF.py --report-dir LAB2/reports/test --out LAB2/reports/Predict_Report_rebuilt.pdf
```

Khi dùng `--output-root` khác, truyền đường dẫn checkpoint/output tương ứng cho bước test và báo cáo. Run đã có output dùng `--resume` với cùng cấu hình hoặc chọn root mới.

## Tài liệu và báo cáo LAB1

| Nội dung | Liên kết |
|---|---|
| Tổng quan LAB1 | [lab-description.md](LAB1/docs/lab-description.md) |
| Dữ liệu và EDA | [data.md](LAB1/docs/data.md) · [eda_report.html](LAB1/reports/eda_report.html) |
| Kiến trúc và huấn luyện | [model.md](LAB1/docs/model.md) · [trainning.md](LAB1/docs/trainning.md) |
| Đánh giá và so sánh 6 mô hình | [evaluation.md](LAB1/docs/evaluation.md) · [report_all_models.html](LAB1/reports/report_all_models.html) |
| Demo và kết quả test | [demo.md](LAB1/docs/demo.md) · [Predict_Report.pdf](LAB1/reports/Predict_Report.pdf) |
| Tiến độ thực hiện | [plan-progression.md](LAB1/docs/plan-progression.md) |
