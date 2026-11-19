# Coursework — Kế hoạch và tiến độ

Đang ở giai đoạn đặc tả: bộ tài liệu bản nháp đã soạn, **chưa có EDA, source, train hay test**. Quy trình theo [rules.md](../../rules.md). Chưa nghiệm thu bước nào ngoài bước 1.

## Lộ trình

| Bước | Công việc | Điều kiện hoàn tất | Trạng thái |
|---:|---|---|---|
| 1 | [Tổng quan bài coursework](lab-description.md) | Chốt bài toán, mục tiêu và phạm vi | Đã duyệt |
| 2 | Kế hoạch và tiến độ | Chốt lộ trình và điều kiện chuyển bước | Chờ duyệt |
| 3 | [Dữ liệu](data.md) | Đặc tả dữ liệu dựa trên output EDA đã đối chiếu | Bản nháp; chờ chạy EDA |
| 4 | [Mô hình](model.md) | Luồng xử lý, input/output và cấu hình mô hình rõ ràng | Bản nháp; chờ duyệt |
| 5 | [Training](trainning.md) | Chốt cấu hình, thực thi, tracking và đầu ra training | Bản nháp; chờ duyệt; chỉ hai run chính |
| 6 | [Demo](demo.md) | Chốt quy trình test/inference và cách trình bày kết quả | Bản nháp; chờ duyệt |
| 7 | Dọn và hoàn thiện source | Docs liên quan được duyệt; source cần thiết, phụ thuộc hợp lệ, kiểm thử đạt | Chưa làm |
| 8 | Chạy train/test | Output thực tế được đối chiếu với đặc tả | Chưa làm |
| 9 | [Báo cáo đánh giá](evaluation.md) | Phân tích cả kết quả train và test đã xác minh | Bị chặn bởi bước 8 |

Các tài liệu ở bước 1–6 cần người dùng xác nhận trước khi chuyển tiếp. Chi tiết kỹ thuật và tiêu chí kiểm tra lấy từ tài liệu tương ứng.

## Bằng chứng hiện có

| Hạng mục | Hiện trạng | Nguồn đối chiếu |
|---|---|---|
| EDA | Chưa chạy; chưa tải dataset | `reports/eda_report.html`, `reports/eda_stats.json` (chưa có) |
| Source | Chưa viết | `tests.py` (chưa có) |
| Smoke | Chưa chạy | — |
| Pretrained | Không áp dụng: mô hình train từ khởi tạo ngẫu nhiên | [model.md](model.md) |
| Official train | Chưa chạy hai run chính | `reports/run_summary.csv`, `reports/selection.json` (chưa có) |
| Official test | Chưa chạy | `reports/test/test_metrics.json` (chưa có) |
| Báo cáo | Chưa có HTML/PDF | `reports/report_all_models.html`, `reports/Predict_Report.pdf` (chưa có) |
| Evaluation | Chưa thể viết vì chưa có output | [evaluation.md](evaluation.md) |

Số tham số trong [model.md](model.md) mục 6 là tính theo công thức, chưa đếm từ source. Smoke, khi có, dùng ít cửa sổ và output riêng, không tạo global checkpoint chính thức và không dùng làm kết quả thực nghiệm.

## Cập nhật tiến độ

Sau mỗi bước, cập nhật trạng thái ngay trong bảng lộ trình và dẫn bằng chứng kiểm tra. Khi dọn source, xác định file cần giữ/xóa và kiểm tra phụ thuộc trước khi gỡ.
