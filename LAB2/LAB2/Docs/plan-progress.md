# LAB2 — Kế hoạch và tiến độ

Đã hoàn tất sáu run chính thức với `num_workers=4`, test global-best và viết [evaluation.md](evaluation.md) từ kết quả đã đối chiếu theo [rules.md](../../rules.md). Nghiệm thu ngày 11/11/2026.

## Lộ trình

| Bước | Công việc | Điều kiện hoàn tất | Trạng thái |
|---:|---|---|---|
| 1 | [Tổng quan bài LAB](lab-description.md) | Chốt bài toán, mục tiêu và phạm vi | Đã duyệt |
| 2 | Kế hoạch và tiến độ | Chốt lộ trình và điều kiện chuyển bước | Đã duyệt |
| 3 | [Dữ liệu](data.md) | Đặc tả dữ liệu dựa trên output EDA đã đối chiếu | Đã duyệt |
| 4 | [Mô hình](model.md) | Luồng xử lý, input/output và cách điều chỉnh mô hình rõ ràng | Đã duyệt |
| 5 | [Training](trainning.md) | Chốt cấu hình, thực thi, tracking và đầu ra training | Đã duyệt; chỉ sáu run chính |
| 6 | [Demo](demo.md) | Chốt quy trình test/inference và cách trình bày kết quả | Đã duyệt |
| 7 | Dọn và hoàn thiện source | Docs liên quan được duyệt; source cần thiết, phụ thuộc hợp lệ, kiểm thử đạt | Hoàn tất; kiểm tra chức năng và seed với 4 workers đạt |
| 8 | Chạy train/test | Output thực tế được đối chiếu với đặc tả | Hoàn tất sáu run, test và kiểm tra checkpoint/lineage |
| 9 | [Báo cáo đánh giá](evaluation.md) | Phân tích cả kết quả train và test đã xác minh | Hoàn tất; số liệu, hình và liên kết đã kiểm tra |

Các tài liệu ở bước 1–6 cần người dùng xác nhận trước khi chuyển tiếp. Chi tiết kỹ thuật và tiêu chí kiểm tra lấy từ tài liệu tương ứng.

## Bằng chứng hiện có

| Hạng mục | Hiện trạng | Nguồn đối chiếu |
|---|---|---|
| EDA | Đã đối chiếu trong bước đặc tả dữ liệu | [HTML](../reports/eda_report.html), [JSON](../reports/eda_stats.json) |
| Source | Kiểm tra gradient/freeze, chuyển giai đoạn, resume và artifact đạt; bổ sung kiểm tra tái lập augmentation với 4 workers đạt | [tests.py](../tests.py), [kiểm tra chức năng trước thực nghiệm](../smoke/implementation_20261110_final/reports/implementation_checks.json) |
| CUDA smoke | Sáu run, mỗi run 2 epoch trên subset; cả sáu mode chạy được batch 16 với AMP | [Summary](../smoke/implementation_20261110_final/reports/run_summary.csv), [kiểm tra batch 16](../smoke/implementation_20261110_final/reports/batch16_check.json) |
| Pretrained | Ba run FE đã dùng ImageNet V1; ba run FT tiếp tục từ đúng parent best | [Metadata và lịch sử](../reports/report_all_models.html), [đối chiếu lineage](../reports/validation_verification.json) |
| Official train | Đủ sáu run hợp lệ, 56 epoch hoàn thành; cùng split, workers 4; đã chọn global-best bằng validation | [Summary](../reports/run_summary.csv), [selection](../reports/selection.json) |
| Official test | Đủ 10.000 dự đoán từ checkpoint đã chọn; JSON/CSV/CM/per-class nhất quán | [Metrics](../reports/test/test_metrics.json), [dự đoán](../reports/test/test_predictions.csv) |
| Báo cáo | HTML offline đủ 7 tab; desktop/mobile và CSV/SVG/PNG đạt; PDF chính thức 6 trang đã mở kiểm tra, dựng lại được từ bundle | [HTML](../reports/report_all_models.html), [PDF](../reports/Predict_Report.pdf), [biên bản kiểm tra](../reports/validation_verification.json) |
| Evaluation | Phân tích đủ sáu run, hội tụ, ba cặp FE → FT, tài nguyên và test; bảy hình tối thiểu 200 dpi, liên kết hợp lệ | [Báo cáo](evaluation.md), [CSV validation](../reports/validation_best_metrics.csv), [CSV tài nguyên](../reports/resource_comparison.csv) |

Smoke dùng khởi tạo ngẫu nhiên và tập con train/validation, không tạo global checkpoint chính thức; metrics smoke không dùng làm kết quả thực nghiệm. Bộ bằng chứng nằm riêng trong `LAB2/smoke`.

Các lượt ngắt/resume được giữ trong `LAB2/archives`. Ảnh hưởng của Windows sleep đến số đo thời gian và phạm vi so sánh được ghi tại [execution_events.json](../reports/execution_events.json) và phân tích trong `evaluation.md`.

## Cập nhật tiến độ

Sau mỗi bước, cập nhật trạng thái ngay trong bảng lộ trình và dẫn bằng chứng kiểm tra. Khi dọn source, xác định file cần giữ/xóa và kiểm tra phụ thuộc trước khi gỡ.
