import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size1=128, hidden_size2=64, num_classes=10):
        super(MLP, self).__init__()
        
        # 1. Duỗi phẳng ảnh 2D (28x28) thành vector 1D (784 features)
        self.flatten = nn.Flatten()
        
        # 2. Định nghĩa các tầng ẩn và hàm kích hoạt (ReLU) 
        self.fc1 = nn.Linear(input_size, hidden_size1)       # Tầng ẩn 1: 784 -> 128
        self.relu1 = nn.ReLU()                                # Hàm kích hoạt ReLU
        self.dropout1 = nn.Dropout(p=0.2)                     # Regularization chống overfitting (Slide 8 Chap 2)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)     # Tầng ẩn 2: 128 -> 64
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        
        # 3. Tầng ngõ ra Output Layer 
        # Trả về Logits trực tiếp (khi kết hợp với CrossEntropyLoss thì không cần Softmax ở đây)
        self.out = nn.Linear(hidden_size2, num_classes)      # 64 -> 10

    def forward(self, x):
        # Flatten ảnh
        x = self.flatten(x)
        
        # Lan truyền tiến qua Hidden Layer 1
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Lan truyền tiến qua Hidden Layer 2
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
    
        # Ngõ ra
        logits = self.out(x)
        return logits