import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import io

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class datasets():
    def __init__(self, path, transform=None):
        self.data = pd.read_parquet(path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_bytes = self.data.iloc[idx]["image"]["bytes"]
        label = self.data.iloc[idx]["label"]

        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        if self.transform:
            image = self.transform(image)

        return image, label


train_path = r"C:\Users\30982\Downloads\train-00000-of-00001.parquet"
test_path = r"C:\Users\30982\Downloads\test-00000-of-00001.parquet"
train_dataset = datasets(train_path, transform=transform)
test_dataset = datasets(test_path, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# 使用 test_dataset 创建测试数据加载器
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class FCNN(nn.Module):
    def __init__(self, channel=1, mid_channel=32, final_channel=64):
        # 调用父类的 __init__ 方法
        super(FCNN, self).__init__()
        self.channel = channel
        self.mid_channel = mid_channel
        self.final_channel = final_channel

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.channel, self.mid_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(self.mid_channel, self.final_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(final_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(7 * 7 * self.final_channel, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


model = FCNN()
criterion = nn.CrossEntropyLoss()
# 修正优化器类名的大小写
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for image, label in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, loss:{total_loss / len(train_dataloader)}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total}%")