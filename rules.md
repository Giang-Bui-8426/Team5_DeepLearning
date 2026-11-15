# AI Agent Rules — Project Workflow

> **Vai trò:** Điều phối quá trình đặc tả → triển khai → chạy thực nghiệm → đánh giá dự án Software/AI/ML.
>
> **Nguyên tắc:** File rule này chỉ quy định **cách làm việc và điều kiện chuyển bước**. Chi tiết kỹ thuật, cấu hình, thiết kế và yêu cầu đầu ra thuộc về các tài liệu đặc tả tương ứng; không lặp chúng tại đây.

## 1. Nguyên tắc nền tảng

1. **Đọc trước khi làm:** Xem tài liệu tổng quan repository (ưu tiên [READMI.md](READMI.md) theo bản gốc; nếu không tồn tại, tìm README thực tế), các đặc tả đã duyệt và source liên quan.
2. **Đúng yêu cầu, đúng phạm vi:** Chỉ sửa các phần cần thiết. Không tự refactor, đổi interface, tái cấu trúc, format hàng loạt hoặc thêm tính năng ngoài yêu cầu.
3. **Tài liệu là nguồn yêu cầu:** Triển khai theo phiên bản đặc tả đã được người dùng xác nhận. Nếu source, tài liệu và yêu cầu mới mâu thuẫn, nêu rõ để xử lý; không tự quyết định âm thầm.
4. **Không suy đoán thành kết quả:** Phân biệt dữ liệu quan sát, giả thuyết, kế hoạch và kết quả đã kiểm chứng. Không bịa metrics, log, checkpoint, báo cáo hay trạng thái thực thi.
5. **Không làm mất dữ liệu:** Không tự xóa/ghi đè dữ liệu, checkpoint, output, thay đổi chưa commit; không tự commit/push hoặc chạy thao tác Git có tính phá hủy.
6. **Ưu tiên tối giản:** Code, tài liệu và phản hồi chỉ giữ nội dung cần thiết, chính xác, có thể kiểm chứng. Không trùng lặp nội dung giữa các file.

## 2. Quy trình bắt buộc cho dự án/LAB

**Khảo sát → Đặc tả & duyệt → Triển khai → Chạy train/test → Viết evaluation → Bàn giao.**

### Giai đoạn A — Khảo sát

- Hiểu bài toán, repository, giới hạn tài nguyên, dữ liệu đầu vào và đầu ra mong đợi.
- Xác định file đặc tả đang có, trạng thái đã duyệt và phần việc cần thực hiện.
- Không tự tạo nội dung kỹ thuật chưa có căn cứ; nêu rõ thông tin còn thiếu.

### Giai đoạn B — Viết và duyệt đặc tả

Với dự án/LAB mới, thực hiện **lần lượt** theo thứ tự dưới đây. **Viết một tài liệu → người dùng review/xác nhận → mới chuyển sang tài liệu tiếp theo.** Không tự triển khai code trước khi các đặc tả liên quan được duyệt.
- Các tài liệu đã được thực hiện sẽ là cơ sở để thực hiện trong các file sau ( ví dụ : có thể dựa vào các phân tích EDA để kết hợp với kiến thức của bạn để đề xuất các tham số , công nghệ phù hợp cho dự án )
- Thực hiện và nghiệm thu theo từng pha trong `plan-progression.md`. **Không đánh dấu `[x]`** chỉ vì code/file đã được tạo; phải có bằng chứng đạt yêu cầu của pha đó.
- Với tác vụ phức tạp, nêu ngắn gọn: **vấn đề → nguyên nhân hoặc giả thuyết → file bị ảnh hưởng → cách kiểm chứng**.
- Phân biệt sự thật đã quan sát, giả thuyết và phương án đề xuất.
- Chọn **cách sửa nhỏ nhất giải quyết được nguyên nhân**; tránh sửa nhiều thứ cùng lúc khiến khó truy lỗi.
- TRÁNH TÌNH TRẠNG GHI TRÙNG NÔI DUNG GIỮA CÁC TÀI LIỆU . YÊU CÂU CÁC TÀI LIỆU ĐẢM NHẬN ĐÚNG NỘI DUNG KHÔNG NÓI LẠI NHỮNG NỘI DUNG ĐÃ ĐƯỢC ĐỀ CẬP Ở NƠI KHÁC LÀM TÀI LIỆU LAN MAN VÀ PHÌNH TO 
- YÊU CẦU CÁC TÀI LIỆU SẠCH , CHÍNH XÁC , KHÔNG LAN MAN , THÔNG TIN VÀ GIẢ THUYÊT ĐÃ ĐƯỢC KIỂM CHỨNG
- Khi thực hiện tài liệu phía sau có thể sửa những tài liệu trước nếu thật sự cần thiết , nhưng những thông tin sửa ở những tài liệu đó cần được người dùng xác nhận mới được sửa

| Tài liệu | Phạm vi chịu trách nhiệm |
|---|---|
| `Lab-description.md` | Bài toán, mục tiêu và phạm vi tổng quan. |
| `Plan-Progress.md` | Lộ trình, công việc, phụ thuộc, tiến độ và điều kiện nghiệm thu. |
| `Data.md` | Đặc tả dữ liệu và kết luận hữu ích từ EDA. |
| `Model.md` | Thiết kế luồng xử lí hay fine-tuning của model một cách rõ ràng và trực quan bằng hình ảnh . Các thông tin về input,output cần được hiển thị rõ ràng . Những thông tin như tại sao sử dụng model này , điểm khác biệt mà dự án sẽ thực hiện so với model bình thường và khả năng về hiệu quả sau khi thực hiện |
| `Trainning.md` | Toàn bộ yêu cầu cấu hình, thực thi, tracking và output của training. |
| `Demo.md` | Yêu cầu quy trình test/inference, đánh giá và hình thức trình bày đầu ra. |

**Phân định trách nhiệm:** Mỗi nội dung chi tiết chỉ có **một file sở hữu**. File khác cần dùng thì dẫn liên kết đến file sở hữu, không sao chép nội dung. `Plan-Progress.md` chỉ theo dõi tiến độ, không thay thế các đặc tả.

**Các nguồn và đường dẫn phải giữ khi làm việc:**

- [READMI.md](READMI.md): tài liệu tổng quan được trỏ trong bản rule gốc (xác minh file có thật; nếu repo dùng `README.md`, dùng đường dẫn thực tế).
- [EDA.py](LAB1/EDA.py): source tham chiếu để **custom EDA theo dữ liệu thực tế** và tạo output HTML phục vụ `Data.md`.
- [TrainReport.py](LAB1/TrainReport.py): source/form tham chiếu để **custom báo cáo training HTML** theo yêu cầu trong `Trainning.md`.
- `plan-progression.md`: tham chiếu theo tên trong bản gốc để nghiệm thu từng pha; đối chiếu với `Plan-Progress.md` của repo trước khi sử dụng, không tự tạo hai tài liệu trùng chức năng.

**Lưu ý:** Các đường dẫn trên là tham chiếu từ rule gốc, **không phải khẳng định file tồn tại trong repository hiện tại**. Khi triển khai, kiểm tra đường dẫn thật và giữ nguyên liên kết hợp lệ.

### Giai đoạn C — Triển khai

- Chỉ triển khai nội dung đã được duyệt; tuân thủ source, convention và cấu trúc hiện có.
- Xác định trước các file cần sửa, thực hiện minimal diff và kiểm tra diff sau khi sửa.
- Chạy kiểm tra phù hợp với phạm vi thay đổi trước khi chuyển sang thực nghiệm; không gọi smoke test là kết quả chính thức.
- Nếu tác vụ chỉ là sửa lỗi nhỏ, làm đúng phần được giao; **không bắt buộc khởi động lại quy trình đặc tả LAB**.

### Giai đoạn D — Chạy thực nghiệm và viết `evaluation.md`

Đối với một pipeline AI/ML hoàn chỉnh, **viết báo cáo đánh giá sau khi triển khai code và thực sự chạy cả training lẫn testing**:

1. Chạy **train** theo `Trainning.md`; lưu output thực tế (log, metrics, checkpoint, báo cáo HTML và các artifact liên quan).
2. Chạy **test/evaluation** theo `Demo.md` bằng checkpoint hợp lệ; lưu output thực tế (metrics, dự đoán, biểu đồ, báo cáo và artifact liên quan).
3. **Đối chiếu output của cả hai lần chạy**, kiểm tra chúng có cùng run/checkpoint, dữ liệu và cấu hình tương ứng; phân biệt train/validation/test.
4. **Sau đó** tạo/cập nhật `evaluation.md` dựa trên bằng chứng có thật, gồm: thông tin các run và nguồn output; kết quả train/validation; kết quả test; nhận xét, vấn đề, giới hạn và kết luận so với tiêu chí nghiệm thu.
5. Dẫn link tương đối đến log, checkpoint, hình và HTML/PDF thực tế; **không chép lại toàn bộ nội dung** từ `Trainning.md`, `Demo.md` hoặc các báo cáo đã sinh.

**Điều kiện bắt buộc:** Nếu train hoặc test chưa chạy thành công, thiếu output hay chưa xác minh checkpoint, **không viết kết luận thực nghiệm như đã hoàn thành**. Ghi rõ trạng thái bị chặn, bằng chứng hiện có và bước cần thực hiện; không dựng metrics để lấp chỗ trống. Không đánh dấu pha tương ứng `[x]` khi chưa đủ bằng chứng.

### Giai đoạn E — Bàn giao

- Cập nhật tiến độ trong tài liệu theo dõi thực tế của repo dựa trên bằng chứng nghiệm thu.
- Báo cáo ngắn: **Đã làm gì → File/output liên quan → Đã chạy/đã kiểm tra → Kết quả → Phần chưa hoàn thành**.
- Không tuyên bố hoàn thành khi chưa thực hiện đủ bước bắt buộc của tác vụ.

## 3. Quy tắc chất lượng và an toàn

- Kiểm chứng đúng luồng đã sửa và tính nhất quán giữa source, cấu hình, checkpoint, log và báo cáo.
- HTML/Markdown/PDF phải sử dụng **dữ liệu thực**; kiểm tra đường dẫn, khả năng hiển thị và chế độ offline nếu đặc tả yêu cầu.
- Không đưa secret vào source/log/report; không tự cài phụ thuộc, tải tài nguyên lớn hoặc dùng dịch vụ trả phí khi chưa đánh giá tác động và có chấp thuận cần thiết.
- Không ngầm thay đổi split, test set hay metric để làm đẹp kết quả. Không đưa kết quả test vào quá trình lựa chọn best checkpoint nếu đặc tả không cho phép.
- Trao đổi bằng tiếng Việt, ngắn gọn và rõ trạng thái **đã thực hiện / chưa thực hiện / cần xác nhận**.

## 4. Điều kiện hoàn tất

Một công việc chỉ được đánh dấu hoàn tất khi: **(1)** đúng phạm vi và đặc tả đã duyệt, **(2)** có bằng chứng kiểm chứng phù hợp, **(3)** các output/link được xác minh, **(4)** tiến độ và bàn giao phản ánh đúng thực tế. Riêng pipeline AI/ML hoàn chỉnh, phải có **output train + output test + `evaluation.md` dựa trên hai nguồn đó**.
