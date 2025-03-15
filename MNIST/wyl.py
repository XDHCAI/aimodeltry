import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm


transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
train_dataset = FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)


train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=64,shuffle=False)

class FCNN(nn.Module):
    def __init__(self,channel=1,mid_channel=32,final_channel=64):
        super(FCNN,self).__init__()
        self.channel = channel
        self.mid_channel = mid_channel
        self.final_channel=final_channel

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.channel,self.mid_channel,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(self.mid_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(self.mid_channel,self.final_channel,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(self.final_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=2)
        )

        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(self.final_channel*7*7,512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,10)


    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for image,label in tqdm(train_dataloader,desc=f"Epoch {epoch+1}"):
        image,label = image.to(device),label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_dataloader)}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image,label in test_dataloader:
            image,label = image.to(device),label.to(device)
            output = model(image)
            _,predicted = torch.max(output.data,1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print(f"Epoch {epoch+1} Test Accuracy: {correct/total}")
