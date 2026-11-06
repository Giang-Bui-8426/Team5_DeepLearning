# Tiến độ thực hiện và kế hoạch hoàn thiện


## 01. Tổng quan tiến độ
Quy trình thực nghiệm : khảo sát dữ liệu, thiết kế nhiều cấu hình MLP, huấn luyện và theo dõi bằng TensorBoard, lựa chọn mô hình theo validation accuracy và chuẩn bị bước đánh giá trên tập test chính thức. Kiểm tra tính nhất quán của checkpoint, chuẩn hóa kết quả đánh giá và hoàn thiện bằng chứng cho báo cáo

| Hạng mục | Tình trạng hiện tại | Minh chứng / phạm vi xác nhận |
|:--|:--|:--|
| Khám phá dữ liệu (EDA) | **Đã thực hiện** | Có `eda_report.html`, kết quả phân bố lớp, pixel, PCA và t-SNE |
| Thiết kế kiến trúc | **Đã triển khai** | `ModelMLP.py` có Plain MLP và ResidualBlock; `Train.py` định nghĩa 6 cấu hình |
| Huấn luyện và theo dõi | **Đã có kết quả thực nghiệm** | Ảnh TensorBoard thể hiện train loss, validation loss, metrics, learning rate |
| Quản lý checkpoint | **Đã có code; cần kiểm tra thêm** | Có cơ chế `last_checkpoint.pt`, `checkpoint_best.pt`, checkpoint global; logic resume còn điểm chưa chuẩn |
| So sánh mô hình | **Đã có code và biểu đồ** | `compare_models()` tạo CSV, biểu đồ metrics, heatmap và accuracy–loss |
| Kiểm thử trên test | **Đã chuẩn bị source; chưa xác nhận kết quả** | Đã có bản `Predict.py` đánh giá toàn tập test; chưa có output test thực tế được gửi kèm |

### Luồng công việc của nhóm

```mermaid
flowchart LR
    A[Khám phá dữ liệu] --> B[Thiết kế 6 MLP]
    B --> C[Huấn luyện & TensorBoard]
    C --> D[Chọn best checkpoint]
    D --> E[Kiểm thử official test]
    E --> F[Tổng hợp báo cáo & demo]
    classDef done fill:#e5f3eb,stroke:#41956b,color:#174a34
    classDef review fill:#fff3da,stroke:#c88e32,color:#654009
    classDef todo fill:#e9edf5,stroke:#71819b,color:#26364b
    class A,B,C done
    class D,E review
    class F todo
```

*Sơ đồ phản ánh mức độ hoàn thiện có bằng chứng, không đại diện cho tỷ lệ phần trăm tiến độ được đo chính xác.*

## 02. Những điểm cần rà soát trước khi chốt kết quả

Nhóm nên xử lý các mục dưới đây trước khi chạy thêm những thí nghiệm mới hoặc công bố kết quả cuối**. Các vấn đề này được xác định trực tiếp từ source hiện tại, không phải khẳng định đã gây sai lệch toàn bộ các lần chạy trước.

| Ưu tiên | Điểm cần rà soát | Hướng xử lý | Điều kiện xác nhận |
|:--:|:--|:--|:--|
| **P0** | `load_checkpoint()` tạo lại scheduler trong biến cục bộ thay vì cập nhật scheduler đang dùng | Load `scheduler_state_dict` vào đúng đối tượng scheduler được truyền vào | Khi resume, lịch learning rate tiếp nối đúng từ checkpoint |
| **P0** | Early Stopping có thể `break` trước bước ghi TensorBoard và lưu checkpoint của epoch vừa đánh giá | Hoàn thành ghi metrics và checkpoint rồi mới kiểm tra điều kiện dừng | Không mất epoch cuối trên log và checkpoint |
| **P1** | `best_loss` và số epoch không cải thiện (`bad`) bị đặt lại khi resume | Lưu và phục hồi trạng thái Early Stopping, hoặc ghi rõ resume không bảo toàn quy tắc dừng | Resume không làm sai lệch logic dừng đã thiết lập |
| **P1** | Tên biểu đồ vẫn ghi “5 MLP Models” khi đang chạy 6 model | Thay bằng tiêu đề tự động dựa trên `len(results)` | Tên biểu đồ, số đường và bảng kết quả thống nhất |
| **P1** | Chưa có bằng chứng test chính thức của model được chọn | Chạy file `Predict.py` với global-best checkpoint, lưu JSON/CSV/hình | Có báo cáo test độc lập, không dùng test để chọn model |
| **P2** | `activation="relu"` đang thực thi `nn.LeakyReLU()` | Giữ và đổi tên cho đúng, hoặc sửa layer rồi train lại có kiểm soát | Tên cấu hình và kiến trúc thực tế không mâu thuẫn |
| **P2** | Tính tái lập của lần train còn phụ thuộc RNG, DataLoader và môi trường | Cố định môi trường, lưu seed và kiểm thử lại nếu cần | Có đủ thông tin để nhóm khác lặp lại thí nghiệm |

> **Nguyên tắc của nhóm:** nếu chỉ thay mã ghi log hoặc sửa một lỗi ảnh hưởng đến quá trình train, nhóm cần ghi rõ phiên bản và xác định có cần chạy lại hay không

## 03. Kế hoạch hoàn thiện theo thứ tự ưu tiên

### Giai đoạn A — Thiết lập pipeline

**Mục tiêu:** dữ liệu, checkpoint được quản lý nhất quán kể cả khi training bị gián đoạn.

- [ ] Sửa luồng resume của scheduler; kiểm tra `optimizer` và `scheduler` có được load đồng bộ hay không.
- [ ] Chuyển kiểm tra Early Stopping xuống sau phần lưu kết quả epoch; kiểm tra trường hợp dừng ở epoch cuối.
- [ ] Kiểm tra lại các file `last_checkpoint.pt`, `checkpoint_best.pt` và `checkpoint_best_global.pt` sau khi chạy.
- [ ] Đảm bảo log cũ và log mới không bị ghép vào cùng một run khi thay đổi cấu hình thí nghiệm.

**Đầu ra mong đợi:** một lần chạy có thể tạm dừng và tiếp tục mà không mất checkpoint hợp lệ hoặc làm sai lệch lịch learning rate.

### Giai đoạn B — Chốt kết quả validation

**Mục tiêu:** Bảng so sánh sáu model hoàn chỉnh

- [ ] Đối chiếu `reports/best_models.csv` với trường `best_metrics` trong từng `checkpoint_best.pt`.
- [ ] Kiểm tra `checkpoint_best_global.pt` được sao chép từ đúng model có validation accuracy cao nhất.
- [ ] Đồng bộ tên mô hình và tiêu đề các hình, thống nhất 6 cấu hình trong toàn bộ báo cáo.
- [ ] Chọn hình đại diện: learning curves, so sánh metrics và ROC–AUC; ghi rõ validation set và tiêu chí chọn checkpoint.

**Đầu ra mong đợi:** bảng kết quả validation có thể truy ngược đến checkpoint thực tế của từng model.

### Giai đoạn C — Đánh giá mô hình cuối trên test set

**Mục tiêu:** đo khả năng tổng quát hóa của mô hình đã được chọn, không dùng test set để điều chỉnh hyperparameter.

- [ ] Load global-best checkpoint bằng phiên bản `Predict.py` dùng để kiểm thử toàn tập dữ liệu.
- [ ] In hoặc xuất cấu hình thực tế của model đi kèm checkpoint trước khi đánh giá.
- [ ] Chạy một lượt trên 10.000 ảnh test chính thức; lưu test loss, accuracy, macro-precision, macro-recall, macro-F1, FPR và ROC–AUC.
- [ ] Xuất confusion matrix và ảnh ví dụ dự đoán đúng/sai để minh họa hạn chế của mô hình.

**Đầu ra mong đợi:** Thấy được sự hiệu quả của model với bài toàn dự đoán

### Giai đoạn D — Hoàn thiện báo cáo và trình bày

**Mục tiêu:** các file tài liệu có vai trò rõ ràng, nội dung được đối chiếu với source và hình ảnh đủ dễ đọc khi trình chiếu.

- [ ] Bảo đảm mỗi ảnh có chú thích đủ: biểu đồ nói về gì, trên tập dữ liệu nào và dùng kết quả của epoch/checkpoint nào.
- [ ] Kiểm tra đường dẫn hình trong Markdown và cách hiển thị trên GitHub/VS Code.
- [ ] Chạy thử quy trình demo từ một thư mục project sạch với hướng dẫn đã viết.

**Đầu ra mong đợi:** bộ hồ sơ đủ để người đọc hiểu nhóm đã thực hiện gì, đánh giá ra sao và có thể chạy lại phần demo mà không cần suy đoán.

## 04. Tiêu chí hoàn thành bài LAB

Nhóm sẽ xem project là sẵn sàng nộp khi các điều kiện sau đều được xác nhận:

- [ ] Sáu cấu hình có checkpoint và bảng validation tương ứng, không nhầm giữa best epoch và epoch cuối.
- [ ] Việc resume và Early Stopping đã được kiểm tra, trạng thái huấn luyện không bị mất do thứ tự lưu sai.
- [ ] Kết quả test được ghi riêng và không bị sử dụng để lựa chọn model.
- [ ] Các biểu đồ chính có chữ, chú thích và tên mô hình dễ đọc
- [ ] Source, môi trường chạy và tài liệu mô tả đủ rõ để tái lập và trình bày.
