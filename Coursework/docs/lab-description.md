# Coursework — Topic 8: Dự báo điện năng thiết bị gia dụng bằng Transformer Encoder

> **Trạng thái: đã chốt ngày 19/11/2026.** Tài liệu sở hữu mục 1–4 (đề tài, hướng chính, người dùng, mục tiêu). Dữ liệu (mục 5) thuộc [data.md](data.md); mục 6–10 thuộc thành viên còn lại. Tài liệu này không lặp các nội dung đó.

## 1. Tên đề tài

**Dự báo điện năng thiết bị gia dụng 10 phút tới từ chuỗi thời gian đa biến bằng Transformer Encoder, đối chứng với LSTM, trên dữ liệu UCI Appliances Energy Prediction.**

| Thành phần | Nội dung |
|---|---|
| Công nghệ / hành động | Transformer Encoder cho hồi quy; LSTM làm baseline học sâu; phân tích attention map |
| Bài toán | Hồi quy chuỗi thời gian đa biến (multivariate time-series regression): dự báo `Appliances` (Wh) tại mốc kế tiếp |
| Phạm vi | Một ngôi nhà tiết kiệm năng lượng ở Bỉ, 11/01–27/05/2016, bước 10 phút, 25 đặc trưng cảm biến/thời tiết ([data.md §1](data.md#1-dataset-và-trường-dữ-liệu)) |

## 2. Hướng chính

**AI dự đoán.** Mục tiêu là dự báo chính xác lượng điện tiêu thụ ngắn hạn từ lịch sử đa biến, và giải thích được mô hình dựa vào mốc thời gian nào thông qua attention. Tốc độ suy luận không phải ràng buộc: mỗi dự báo có 10 phút trước khi mốc tiếp theo đến.

## 3. Người dùng và vấn đề

**Người dùng:** hộ gia đình và hệ thống quản lý năng lượng tòa nhà (BEMS) cần biết trước tải điện thiết bị trong thời gian ngắn để dời lịch thiết bị, cảnh báo tiêu thụ bất thường hoặc tham gia điều chỉnh phụ tải.

**Khó khăn hiện tại**, căn cứ EDA ([data.md §4](data.md#4-kết-quả-eda-và-ý-nghĩa-sử-dụng)):

| Quan sát | Hệ quả |
|---|---|
| Tiêu thụ nền thấp (trung vị 60 Wh) xen các đỉnh ngắn đến 1080 Wh; 10% mốc cao nhất chiếm 37.1% năng lượng | Phần cần dự báo nhất, tức các đỉnh, cũng là phần khó nhất |
| Dự báo "giữ nguyên giá trị trước" (persistence) đã đạt R² 0.481 trên validation nhưng luôn trễ một bước ở đỉnh | Cách đơn giản nhất bỏ lỡ đúng thời điểm tải tăng |
| Không cảm biến nào tương quan mạnh với mục tiêu (\|Spearman\| lớn nhất 0.354) | Không thể dự báo tốt bằng quy tắc hay hồi quy tuyến tính trên một vài cảm biến |
| Train là mùa đông, test là cuối xuân; nhiệt độ lệch mạnh giữa hai tập | Mô hình phải tổng quát sang điều kiện chưa thấy |

Lý do chọn Transformer Encoder và kỳ vọng so với LSTM nằm ở [model.md §1](model.md#1-kiến-trúc-lựa-chọn). Đó là giả thuyết, được kiểm chứng bằng các ngưỡng ở mục 4.2.

## 4. Mục tiêu

### 4.1. Đầu ra cần đạt

- Mô hình **Transformer Encoder** đã train, dự báo `Appliances` 1 bước, có checkpoint tái lập được.
- Mô hình **LSTM** baseline đã train.
- Bảng so sánh **persistence / LSTM / Transformer** trên validation và test.
- **Attention map** của Transformer, đối chiếu với tự tương quan đã đo trong EDA.

### 4.2. Chỉ số và ngưỡng nghiệm thu

Metric tính trên **Wh** (sau khi đảo mọi biến đổi mục tiêu), dự báo 1 bước. Mốc so sánh là persistence trên validation, đã đo trong [data.md §4.7](data.md#47-baseline-tham-chiếu-trên-validation): **RMSE 66.43 Wh, MAE 26.16 Wh, R² 0.481**.

| Mục tiêu | Chỉ số | Ngưỡng | Trạng thái |
|---|---|---|---|
| Vượt baseline không học máy | MAE và RMSE validation của Transformer | Thấp hơn persistence **≥ 5%**: MAE ≤ **24.85 Wh**, RMSE ≤ **63.11 Wh** | Chốt 19/11/2026 |
| Giải thích được biến động | R² validation của Transformer | > **0.481** (persistence) | Chốt 19/11/2026 |
| So sánh Transformer với LSTM | MAE và RMSE validation | Với 1 seed: chỉ báo cáo chênh lệch, không kết luận mô hình nào tốt hơn. Kết luận "tốt hơn" cần 3 seed mỗi mô hình (khảo sát seed ở [trainning.md §9](trainning.md#9-khảo-sát-mở-rộng-tùy-chọn)) và chênh lệch MAE trung bình > tổng hai độ lệch chuẩn | Công thức chốt; số điền sau train |
| Dự báo đỉnh | MAE trên các mốc có `Appliances` ≥ p90 (196 Wh) | Báo cáo cho cả ba phương pháp, không đặt ngưỡng | Chỉ báo cáo |
| Attention | Trọng số attention trung bình theo độ trễ | Báo cáo định tính: đối chiếu với ACF (lag gần và lag 144); không đặt ngưỡng | Chỉ báo cáo |

Ngưỡng xét trên **validation**; test chỉ dùng một lần để xác nhận cuối và báo cáo cả RMSE, MAE, R². Vì giữa train và test có lệch mùa ([data.md §4.6](data.md#46-lệch-phân-phối-theo-mùa-giữa-các-tập)), sai số test cao hơn validation là rủi ro đã biết; khi đó phải ghi nhận, không được đổi split hay ngưỡng.

### 4.3. Quy trình chốt ngưỡng

1. Mức cải thiện 5% so với persistence ở bảng 4.2 đã chốt ngày **19/11/2026**, **trước khi** train Transformer.
2. Không sửa ngưỡng sau khi đã thấy kết quả validation hoặc test.
3. Nếu Transformer không đạt ngưỡng, kết quả vẫn được báo cáo như một phát hiện hợp lệ, kèm phân tích nguyên nhân.

## Tài liệu chi tiết

| Nội dung | Tài liệu |
|---|---|
| Dữ liệu, EDA, split, quyền sử dụng (mục 5) | [data.md](data.md) · [báo cáo EDA](../reports/eda_report.html) |
| Phương pháp: baseline và kiến trúc (mục 7) | [model.md](model.md) |
| Cấu hình train, đánh giá | [trainning.md](trainning.md) |
| Kế hoạch, tiến độ và rủi ro (mục 10) | [plan-progress.md](plan-progress.md) |
