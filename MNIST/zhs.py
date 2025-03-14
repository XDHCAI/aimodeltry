import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.6,), (0.6,))
])

# 使用MNIST数据集
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 定义网络结构
class FCNN(nn.Module):
    def __init__(self, channel=1, mid_channel=128, final_channel=256):
        super(FCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, mid_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(mid_channel, final_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(final_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(7 * 7 * final_channel, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCNN().to(device)
model_file = 'model.pth'

# 判断是否存在已训练好的模型，若有则加载，若无则训练
if os.path.exists(model_file) :
    print("找到已保存的模型，直接加载模型参数")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
else:
    print("未找到模型，开始训练")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(train_dataloader):.4f}")

    # 保存训练好的模型参数
    torch.save(model.state_dict(), model_file)
    print("模型已保存")

# 推理部分：加载图片并进行预测
png_image = Image.open("C:\\Users\zheng hao shuai\Pictures\Screenshots\屏幕截图 2025-03-12 215438.png")
jpg_image = png_image.convert('RGB')
jpg_image.save("output.jpg")

model.eval()
image = Image.open("output.jpg").convert('L')
# 如果图片背景和MNIST数据集相反，可以反转颜色
image = ImageOps.invert(image)
image = transform(image)
image = image.unsqueeze(0)

# 显示预处理后的图片
image_vis = image.squeeze().cpu().numpy()
plt.imshow(image_vis, cmap='gray')
plt.title("预处理后的图片")
plt.show()

# 进行预测
with torch.no_grad():
    outputs = model(image.to(device))
    predicted_class = outputs.argmax(dim=1)
    print("预测数字:", predicted_class.item())
