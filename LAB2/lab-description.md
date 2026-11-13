# LAB2 — Thực hành với mô hình CNN pretrained

## Bài toán

Phân loại ảnh CIFAR-10 thành 10 lớp bằng các mạng CNN đã được huấn luyện trước trên ImageNet.

## Mục tiêu và phạm vi

- Nạp và khám phá ResNet18, VGG16, DenseNet121 từ `torchvision.models`.
- Với mỗi kiến trúc, train tầng phân loại trước, sau đó fine-tune tiếp từ best checkpoint của giai đoạn đó.
- Khảo sát learning rate và batch size; theo dõi quá trình huấn luyện bằng TensorBoard.
- So sánh chất lượng phân loại và chi phí tính toán, kiểm thử mô hình đã chọn và tổng hợp kết quả thực nghiệm.

## Tài liệu chi tiết

| Nội dung | Tài liệu |
|---|---|
| Lộ trình, tiến độ và nghiệm thu | [plan-progress](plan-progress) |
| Dữ liệu và kết luận từ EDA | [data.md](data.md) |
| Kiến trúc, luồng xử lý và fine-tuning | [model.md](model.md) |
| Cấu hình, thực thi và báo cáo training | [trainning.md](trainning.md) |
| Test/inference và trình bày đầu ra | [demo.md](demo.md) |
| Kết quả, phân tích và kết luận sau train/test | [evaluation.md](evaluation.md) |
