import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from datetime import datetime

batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


transformer = transforms.Compose([transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5], std=[0.5])])

train_datasets = datasets.FashionMNIST(root='./fashion_mnist/',
                                       train=True,
                                       transform=transformer,
                                       download=True)

test_datasets = datasets.FashionMNIST(root='./fashion_mnist/',
                                      train=False,
                                      transform=transformer,
                                      download=True)

train_loader = DataLoader(dataset=train_datasets, 
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_datasets,
                         batch_size=batch_size,
                         shuffle=False)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample 
    def forward(self, x):
        identity = x 
        if self.downsample is not None: 
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity 
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, include_top=True): 
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0]) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, out_channels, num_blocks, stride=1): 
        downsample = None
        if stride!= 1 or self.in_channels!= out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [] 
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels 
        for _ in range(1,num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x

def resnet18(num_classes=2, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

model = resnet18(num_classes=10).to(device)
loss_fc = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.8)
optimizer = optim.Adam(model.parameters(), lr=0.003)

log_dir = 'pro'
os.makedirs(log_dir, exist_ok=True)

def train():
    model.train()
    total_loss = 0.0
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        out = model(input)
        loss = loss_fc(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader) 

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.3f}%") 
    return accuracy

checkpoint_dir = 'pro/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

best_accuracy = 0.0

for epoch in range(20):
    print(f'epoch:{epoch+1}')
    train_loss = train()
    current_accuracy = test() 

    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'best_model_epoch_{epoch+1}_acc_{current_accuracy:.2f}.pth'
        )
        torch.save(model.state_dict(), checkpoint_path)
        
        log_entry = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"New best model saved: Accuracy {current_accuracy:.2f}% "
            f"at epoch {epoch+1}\n"
        )
        with open(os.path.join('pro', 'training.log'), 'a') as f:
            f.write(log_entry)