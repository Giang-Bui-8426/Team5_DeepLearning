# Coursework — Knowledge Distillation: nén CNN phân loại CIFAR-10 cho suy luận thời gian thực
## 6. Phạm vi

> **Trạng thái: chờ nhóm duyệt.** Tài liệu sở hữu mục 1–4 (đề tài, hướng chính, người dùng, mục tiêu). Mục 6–10 (phạm vi, phương pháp, sản phẩm, đánh giá, kế hoạch) thuộc thành viên còn lại; tài liệu này không lặp các nội dung đó.

### 6.1. Phần bắt buộc

| # | Hạng mục | Ghi chú |
|---|---|---|
| B1 | Teacher cố định: checkpoint `02_resnet18_fine_tune` của LAB2 | Không train lại, trừ khi mất checkpoint (xem [data.md §5](data.md#5-phương-án-thay-thế)) |
| B2 | Một kiến trúc student train từ đầu trên ảnh 32×32 | Trần tham số theo [§4.2](#42-chỉ-số-và-ngưỡng-nghiệm-thu) |
| B3 | Student **train thường** (nhãn cứng), 3 seed | Cung cấp μ, σ để khóa ngưỡng |
| B4 | Student **KD** (nhãn cứng + soft label online từ teacher), 3 seed | Cùng split, augmentation, ngân sách train với B3 |
| B5 | Đo latency CPU của teacher và student theo quy trình [§4.3](#43-quy-trình-chốt-ngưỡng) | Forward riêng và toàn quy trình; GPU báo cáo kèm, không có ngưỡng |
| B6 | Test chính thức: teacher, student train thường tốt nhất, student KD tốt nhất | Mỗi mô hình một lần, chọn bằng validation |
| B7 | Khóa ngưỡng ghi ngày **trước** khi train KD | Không sửa sau khi thấy kết quả |

### 6.2. Phần mở rộng (chỉ làm khi B1–B7 xong)

| # | Mở rộng | Giá trị | Chi phí |
|---|---|---|---|
| M1 | Lưới nhỏ nhiệt độ T và trọng số α | Hiểu độ nhạy của KD | Mỗi tổ hợp là một lần train; đã tính trong ngân sách ở [§7.5](#75-quy-trình-thực-hiện) |
| M2 | Quét độ rộng student (hệ số kênh 0.5×, 1×, 2×) | Đường cong kích thước – accuracy – latency | Thêm run train |
| M3 | Lượng tử hóa INT8 hoặc xuất ONNX cho CPU | Giảm thêm latency | Rủi ro tương thích trên máy |
| M4 | Demo realtime từ webcam hoặc thư mục ảnh | Dễ trình bày | Không ảnh hưởng nghiệm thu |
| M5 | McNemar giữa student KD và train thường trên test | Kiểm định khác biệt theo cặp | Chỉ cần dự đoán đã lưu |
| M6 | Calibration (ECE) | So độ tin cậy confidence giữa teacher và student | Chỉ cần xác suất đã lưu |

### 6.3. Ngoài phạm vi

- Teacher khác hoặc ensemble nhiều teacher; distillation đặc trưng trung gian (hint/attention) làm kết quả chính.
- Dataset khác làm kết quả chính. CIFAR-100 chỉ xuất hiện như nhánh dự phòng khi `G < 2σ`.
- Triển khai trên điện thoại hoặc thiết bị nhúng; đo công suất tiêu thụ.
- Sử dụng thương mại (xem [data.md §4](data.md)).

### 6.4. Quy tắc giữ phạm vi

Nếu thiếu thời gian, bỏ theo thứ tự M6 → M5 → M4 → M3 → M2 → M1. B1–B7 không rút gọn; nếu không hoàn thành thì báo cáo ghi rõ phần thiếu thay vì thay bằng số ước lượng.

## 7. Phương pháp

### 7.1. Baseline

| Mô hình | Vai trò | Huấn luyện |
|---|---|---|
| **Teacher**: ResNet18 FT (LAB2) | Mốc chất lượng và mốc latency | Không train lại; tham chiếu validation 93.22%, test 92.91% từ [LAB2 evaluation.md](../../LAB2/Docs/evaluation.md), phải xác nhận lại khi nạp |
| **Student train thường** | Đối chứng của KD | Chỉ nhãn cứng, 3 seed, cùng kiến trúc, split, augmentation và ngân sách với student KD |

Student train thường là baseline duy nhất để kết luận KD có tác dụng; so với teacher chỉ để biết khoảng cách `G`.

### 7.2. Giải pháp KD

Student học đồng thời nhãn cứng và phân phối xác suất làm mềm của teacher:

`L = α · T² · KL( softmax(z_teacher / T) ‖ softmax(z_student / T) ) + (1 − α) · CE(z_student, y)`

- Teacher: `eval()`, `torch.inference_mode()`, trọng số cố định, logits chưa softmax.
- Student suy luận ở `T = 1`; temperature chỉ dùng khi tính loss.
- Soft label **tính online** trên batch đã augmentation: biến đổi hình học áp một lần lên ảnh 32×32 rồi tách hai nhánh (student giữ 32×32; teacher resize 256, center crop 224, ImageNet mean/std). Lý do và chi phí: [data.md §2.2](data.md#22-soft-label-tính-online).
- Student normalize theo CIFAR-10 mean/std của train fit ([data.md §2.1](data.md#21-hai-đầu-vào-khác-nhau)).

### 7.3. Kiến trúc student (đề xuất)

CNN kiểu VGG nhỏ, không dùng pretrained, nhận `[B,3,32,32]`, trả logits `[B,10]`:

| Khối | Cấu trúc | Kênh |
|---|---|---|
| 1 | 2× (Conv3×3 → BN → ReLU), MaxPool 2 | 32 |
| 2 | 2× (Conv3×3 → BN → ReLU), MaxPool 2 | 64 |
| 3 | 2× (Conv3×3 → BN → ReLU), MaxPool 2 | 128 |
| Head | Global average pooling → Linear(128, 10) | — |

Tính tay khoảng 288.7 nghìn tham số (conv không bias, có BN), thấp hơn xa trần tham số ở [§4.2](#42-chỉ-số-và-ngưỡng-nghiệm-thu). Con số này **phải được đếm lại bằng `model.parameters()`** khi cài đặt. Kiến trúc có thể tăng độ rộng nếu validation của student train thường quá thấp; thay đổi phải thực hiện **trước** khi khóa ngưỡng.

### 7.4. Cấu hình huấn luyện (đề xuất khởi điểm)

| Mục | Đề xuất | Ghi chú |
|---|---|---|
| Optimizer | SGD momentum 0.9, nesterov, weight decay 5e-4 | Chốt chung cho train thường và KD |
| Learning rate | Cosine decay từ 0.05 | Khảo sát nhanh trên baseline, không tune riêng cho KD |
| Batch / epoch | 128 / 30 | Epoch chốt sau khi đo thời gian thực của vòng KD |
| Augmentation | Tập đã duyệt ở LAB2 trainning.md §2.1, áp trên ảnh 32×32 | **Cần xác nhận:** nếu LAB2 định nghĩa tham số ở 224×224 thì phải quy đổi tương đương sang 32×32 và ghi lại |
| Precision | Student AMP FP16; teacher FP32 | Teacher chuyển AMP chỉ khi logits đối chiếu khớp FP32 |
| Chọn checkpoint | Validation accuracy; hòa thì macro F1 → loss → run ID | Giữ quy tắc LAB2 |
| Seed | 3 seed, cùng danh sách cho train thường và KD | Split không đổi |

Cách tổ chức script (cấu hình, run ID, lưu checkpoint, TensorBoard) theo `LAB1/Train.py` và `LAB2/train.py`, đã dùng ở các lab trước.

### 7.5. Quy trình thực hiện

1. **Xác nhận teacher:** nạp strict, đối chiếu hash theo [data.md §1](data.md#1-nguồn-dữ-liệu), tính lại validation accuracy và so với số LAB2 ở §7.1.
2. **Đo baseline** (mục 4.3 bước 1): latency teacher; student train thường 3 seed → μ, σ, `G`.
3. **Khóa ngưỡng:** tính các ngưỡng, ghi vào [mục 10](#10-kế-hoạch-phụ-thuộc-và-rủi-ro) kèm ngày và commit.
4. **Kiểm tra nhánh dự phòng:** nếu `G < 2σ`, chuyển CIFAR-100 theo [data.md §5](data.md#5-phương-án-thay-thế).
5. **Chọn T, α (M1):** lưới nhỏ, ví dụ T ∈ {2, 4}, α ∈ {0.5, 0.9}, chạy 1 seed, chọn bằng validation. Rồi chạy 3 seed với cấu hình đã chọn.
6. **Test:** khóa cả ba checkpoint (teacher, train thường tốt nhất, KD tốt nhất), test mỗi mô hình một lần.

Việc chọn T, α trên validation là ngân sách tuning mà student train thường không có; đây là thiên lệch nhẹ có lợi cho KD và phải ghi trong báo cáo (xem [§9.5](#95-bộ-tình-huống-và-cách-kết-luận)).

### 7.6. Giả thuyết cần kiểm chứng

- H1: student KD vượt student train thường theo ngưỡng "KD có tác dụng" ở §4.2.
- H2: student KD thu hẹp khoảng `G` tới teacher theo ngưỡng "Giữ chất lượng" ở §4.2.
- H3: student đạt ngưỡng realtime ở §4.2 và nhanh hơn rõ rệt so với teacher. **Chưa kiểm chứng**: LAB2 chưa đo CPU nên chưa biết teacher đã đạt ngưỡng hay chưa.

Chưa có kết quả nào trong tài liệu này; mọi giả thuyết có thể bị bác bỏ và báo cáo vẫn hợp lệ nếu ghi đúng kết quả.

## 8. Sản phẩm

### 8.1. Danh mục

| Nhóm | Sản phẩm | Nội dung và điều kiện |
|---|---|---|
| Mô hình | `student.py` | Lớp student (32×32 → logits `[B,10]`), hàm đếm tham số, không có Softmax trong `forward` |
| Mô hình | Checkpoint student train thường và student KD (best theo validation) | Kèm metadata: seed, cấu hình, hash teacher, hash split, ngày; nhỏ, dự kiến vài MB |
| Mã train | `Distill.py` | Chế độ `baseline` và `kd` dùng chung vòng lặp; chỉ khác hàm loss |
| Mã đo | `Benchmark.py` | Đo latency theo [lab-description.md §4.3](#43-quy-trình-chốt-ngưỡng); xuất JSON và hình |
| Mã test | `Predict.py` | Test một checkpoint, lưu dự đoán, metrics, hình theo định dạng LAB2 ([demo.md](../../LAB2/Docs/demo.md)) |
| Kiểm thử | `tests.py` | Kiểm tra mục 8.3 |
| Dữ liệu mẫu | Danh sách chỉ số ảnh dùng đo latency; vài ảnh minh họa | Không đóng gói CIFAR-10, `splits.npz`, checkpoint teacher (bị loại khỏi Git) |
| Kết quả | `thresholds.json`, `baseline_runs.csv`, `kd_runs.csv`, `latency.json`, `test_*.json/csv`, hình so sánh | Truy được về cùng run ID |
| Báo cáo | `Coursework_Report` (PDF hoặc HTML) | Một bảng ba mô hình; phần kết luận bám ngưỡng |
| Tài liệu | Bộ docs mục 1–10, README hướng dẫn chạy lại | Link chéo hợp lệ |
| Demo (mở rộng) | Chương trình realtime đọc webcam/thư mục, hiển thị nhãn và FPS | Chỉ khi xong B1–B7 |

### 8.2. Điều kiện bàn giao

1. Người khác chạy lại được từ README, kể cả bước lấy checkpoint teacher và split (hoặc cách sinh lại).
2. Mọi con số trong báo cáo truy được về file kết quả; số chưa đo ghi `N/A` kèm lý do.
3. Ngưỡng nghiệm thu trong báo cáo trùng `thresholds.json`, kèm ngày khóa.
4. Kết quả không đạt ngưỡng vẫn được báo cáo đầy đủ; không sửa ngưỡng.

### 8.3. Kiểm thử trước khi chạy chính thức

| Kiểm tra | Kỳ vọng |
|---|---|
| Nạp teacher strict, hash đúng | Khớp SHA-256 trong [data.md §1](data.md#1-nguồn-dữ-liệu) |
| Validation accuracy teacher tính lại | Khớp số LAB2 ở [§7.1](#71-baseline) trong sai số số học |
| Hai nhánh từ cùng một ảnh augmentation | Với cùng seed, nhánh teacher bằng kết quả resize/normalize trực tiếp từ ảnh 32×32 đã augmentation |
| Teacher không nhận gradient, BN ở `eval()` | Tham số teacher không đổi sau một bước train |
| Số tham số student | Không vượt trần ở [§4.2](#42-chỉ-số-và-ngưỡng-nghiệm-thu) |
| Train thường và KD dùng cùng split, seed, augmentation, epoch | Đối chiếu từ cấu hình đã lưu |
| Logits hữu hạn, output `[B,10]` | Đạt với batch cuối |

## 9. Đánh giá

### 9.1. Chỉ số

| Nhóm | Chỉ số | Dùng cho |
|---|---|---|
| Chất lượng | Accuracy, macro F1, Cross-Entropy, recall từng lớp, confusion matrix | Validation (chọn, nghiệm thu) và test (xác nhận) |
| Tương quan với teacher | Tỷ lệ dự đoán trùng teacher; trong các ảnh teacher sai, tỷ lệ student cũng sai | Chỉ mô tả, test |
| Hiệu năng | Latency CPU batch 1: trung vị, p95; forward riêng và toàn quy trình | Ngưỡng realtime |
| Chi phí | Tổng tham số; dung lượng checkpoint; GPU latency | Báo cáo kèm |
| Ổn định | Trung bình và độ lệch chuẩn qua 3 seed | Baseline và KD |

Mọi metric của một run lấy từ **cùng best epoch**.

### 9.2. Chia dữ liệu và vai trò

| Tập | Vai trò |
|---|---|
| Train fit 45.000 | Train student; soft label online |
| Validation 5.000 | Chọn checkpoint và cấu hình KD; **nghiệm thu hai ngưỡng chất lượng** |
| Test 10.000 | Xác nhận cuối cho ba mô hình, mỗi mô hình một lần; không dùng để chọn hoặc sửa ngưỡng |
| Ảnh đo latency | Tập cố định lấy từ validation (ví dụ 500 ảnh); dữ liệu ảnh không ảnh hưởng thời gian nhưng cố định để tái lập |

### 9.3. Quy ước tính ngưỡng (cần nhóm chốt cùng với việc khóa ngưỡng)

| Điểm chưa nêu trong mục 4.2 | Đề xuất |
|---|---|
| Độ lệch chuẩn σ | Độ lệch chuẩn mẫu (ddof = 1) của 3 validation accuracy baseline |
| Giá trị `acc(KD)` so với ngưỡng | **Trung bình 3 seed KD**; báo cáo thêm seed thấp nhất và cao nhất |
| `acc(teacher)` trong `G` | Validation accuracy teacher tính lại ở bước xác nhận, đối chiếu số LAB2 |
| Làm tròn | Giữ nguyên số gốc khi so sánh; làm tròn chỉ khi hiển thị |

Với n = 3, σ ước lượng rất thiếu chắc chắn; điều này phải nêu khi kết luận.

### 9.4. Quy trình đo latency

Triển khai thuộc `Benchmark.py`, theo [§4.3](#43-quy-trình-chốt-ngưỡng):

- Giao thức đo (warm-up, số lần đo, `inference_mode`, số luồng, trung vị và p95) theo [§4.3](#43-quy-trình-chốt-ngưỡng). Bổ sung ở đây:
- Ranh giới **toàn quy trình**: từ tensor `uint8 [32,32,3]` trong bộ nhớ đến nhãn dự đoán (`argmax`). Không tính đọc đĩa. Teacher gồm resize 256, crop 224, normalize; student gồm normalize.
- Dùng `time.perf_counter`; ghi số luồng, tên CPU, phiên bản PyTorch, hệ điều hành, chế độ nguồn.
- Điều kiện đo: cắm sạc, chế độ hiệu năng cao, tắt ứng dụng nặng; lặp **3 phiên đo độc lập** cách nhau, báo cáo trung vị của p95 và khoảng giữa các phiên.
- Nếu phát hiện máy ngủ, giảm xung hay phiên đo lệch rõ rệt, loại phiên đó và ghi lý do trong sự kiện thực thi, giống cách LAB2 xử lý.
- GPU: cùng quy trình, thêm `torch.cuda.synchronize()`; chỉ báo cáo.

### 9.5. Bộ tình huống và cách kết luận

| Tình huống | Nội dung | Trạng thái |
|---|---|---|
| S0 | Xác nhận teacher, đo latency baseline | Bắt buộc |
| S1 | Student train thường 3 seed → μ, σ, `G` | Bắt buộc |
| S2 | Student KD 3 seed trên validation | Bắt buộc |
| S3 | Test ba mô hình đã khóa, phân tích lớp yếu và cặp nhầm so với teacher | Bắt buộc |
| S4 | Nhánh CIFAR-100 nếu `G < 2σ` | Có điều kiện |
| S5 | McNemar, calibration, quét độ rộng | Mở rộng |

**Bảng kết luận**

Bốn ngưỡng lấy nguyên từ [§4.2](#42-chỉ-số-và-ngưỡng-nghiệm-thu) (giá trị đã khóa nằm ở [§10.3](#103-khóa-ngưỡng-điền-ở-m4)); không đặt lại ở đây.

| Ngưỡng | Chưa đạt thì |
|---|---|
| Realtime | Báo cáo số đo thật, không đổi ngưỡng |
| Mô hình nhỏ | Đổi kiến trúc trước khi khóa; sau khi khóa thì ghi không đạt |
| KD có tác dụng | Kết luận KD chưa chứng minh được tác dụng |
| Giữ chất lượng | Ghi mức thu hẹp thực tế |

**Giới hạn cần nêu**

- Ba seed và một split: kết luận chỉ cho lần chạy này.
- Lưới T, α chọn bằng validation là ngân sách tuning mà baseline không có, thiên lệch nhẹ có lợi cho KD.
- Nếu teacher đã đạt ngưỡng realtime trên CPU, mục tiêu tốc độ vẫn đạt nhưng không chứng minh được lợi ích của distillation; khi đó báo cáo thêm tỷ lệ tăng tốc student/teacher.
- Teacher logits trên train fit tự tin hơn trên ảnh chưa thấy ([data.md §3](data.md)); không suy ra nguyên nhân thay đổi hành vi student từ một lần chạy.
- Latency phản ánh một máy, một cấu hình luồng; không ngoại suy sang CPU khác.
- Confidence Softmax chưa hiệu chỉnh.

## 10. Kế hoạch, phụ thuộc và rủi ro

### 10.1. Phân công

| Việc | Người phụ trách | Phụ thuộc |
|---|---|---|
|| `⟨tên⟩` |  |
|| `⟨tên⟩` |  |
| | `⟨tên⟩`| |
|| `⟨tên⟩` |  |
| | `⟨tên⟩` |  |

### 10.2. Mốc

Tài liệu chỉ ghi điều kiện hoàn tất; ngày giờ do nhóm gán.

| Mốc | Công việc | Điều kiện hoàn tất | Ngày |
|---|---|---|---|
| M1 | Docs 10 mục, rà chéo, link hợp lệ | Hai người duyệt | Deadline coursework (trưa thứ Ba) |
| M2 | Lấy hoặc sinh lại teacher, split, data; chạy kiểm tra 8.3 | Hash và accuracy teacher khớp | `⟨…⟩` |
| M3 | Đo baseline: latency teacher, student train thường 3 seed | μ, σ, `G` có file kết quả | `⟨…⟩` |
| M4 | **Khóa ngưỡng** | Bảng 10.3 đã điền, đã commit | `⟨…⟩` |
| M5 | Student KD 3 seed (kèm lưới T, α) | Có `kd_runs.csv` | `⟨…⟩` |
| M6 | Test ba mô hình, báo cáo | Báo cáo bám bảng kết luận | `⟨…⟩` |

### 10.3. Khóa ngưỡng (điền ở M4)

| Mục | Giá trị | Ghi chú |
|---|---|---|
| Ngày khóa / commit | `⟨…⟩` | Trước khi train KD |
| `acc(teacher)` validation | `⟨…⟩` | Tính lại, đối chiếu số LAB2 |
| μ, σ (3 seed train thường) | `⟨…⟩` | ddof = 1 nếu nhóm đồng ý |
| `G = acc(teacher) − μ` | `⟨…⟩` | |
| Ngưỡng KD có tác dụng `μ + 2σ` | `⟨…⟩` | |
| Ngưỡng giữ chất lượng `μ + 0.5·G` | `⟨…⟩` | |
| Điều kiện nhánh dự phòng `G < 2σ` | `⟨đúng/sai⟩` | |
| Teacher CPU latency (trung vị, p95) | `⟨…⟩` | Để biết mục tiêu realtime có ý nghĩa không |

### 10.4. Phụ thuộc

| Phụ thuộc | Hiện trạng theo [data.md §1](data.md#1-nguồn-dữ-liệu) | Cách xử lý |
|---|---|---|
| CIFAR-10 tại `LAB2/data/prepared/` | Chưa có trên máy | Tải qua torchvision |
| `LAB2/data/splits.npz` | Bị loại khỏi Git | Xin từ người chạy LAB2 hoặc sinh lại bằng `LAB2/EDA.py` rồi đối chiếu số ảnh mỗi lớp |
| Checkpoint teacher | Bị loại khỏi Git | Xin từ người chạy LAB2; nếu mất, train lại run 01 → 02 |
| Mã LAB2 (`model.py`, preprocessing) | Có trong repo | Import hoặc sao chép tối thiểu; không sửa LAB2 |
| GPU RTX 3050 6 GB và CPU của máy đo | Một máy | Mọi số đo latency ghi tên máy |

### 10.5. Rủi ro

| Rủi ro | Ảnh hưởng | Cách xử lý |
|---|---|---|
| Thiếu checkpoint hoặc split | Chặn toàn bộ thí nghiệm | Làm M2 sớm nhất; phương án dự phòng ở mục 10.4 |
| `G < 2σ` | Không đủ khoảng chứng minh KD | Chuyển CIFAR-100; cần fine-tune teacher mới, thêm thời gian lớn |
| Teacher đã đạt realtime trên CPU | Mục tiêu tốc độ không phân biệt được KD | Báo cáo tỷ lệ tăng tốc; nêu giới hạn |
| Teacher online làm vòng train chậm, thiếu VRAM | Không đủ 3 seed trong thời gian | Dùng logits tính trước theo [data.md §5](data.md#5-phương-án-thay-thế), ghi giới hạn lệch augmentation |
| Latency CPU nhiễu (giảm xung, tiến trình nền, máy ngủ) | Sai lệch p95 | 3 phiên đo, loại phiên lỗi có ghi lý do |
| Student quá yếu hoặc quá mạnh | KD không phân biệt được | Điều chỉnh kiến trúc **trước** khi khóa ngưỡng |
| Sửa ngưỡng sau khi thấy kết quả KD | Mất giá trị nghiệm thu | Commit `thresholds.json` ở M4, không chỉnh |
| Dùng test để chọn T, α hoặc checkpoint | Kết quả không hợp lệ | Chọn bằng validation, test mỗi mô hình một lần |
| Augmentation LAB2 không quy đổi trực tiếp sang 32×32 | Student và teacher thấy dữ liệu khác nhau | Ghi lại quy đổi, kiểm tra hai nhánh cùng ảnh |
| Hai người viết trùng nội dung | Docs rối | Mục 1–5 và 6–10 tách chủ sở hữu; rà chéo ở M1 |
| Vi phạm điều khoản ImageNet khi dùng thương mại | Rủi ro pháp lý | Giữ mục đích học tập; không phân phối lại dataset và checkpoint |

## Tài liệu chi tiết

| Nội dung | Tài liệu |
|---|---|
| Dữ liệu, soft label của teacher, quyền sử dụng (mục 5) | [data.md](data.md) |
