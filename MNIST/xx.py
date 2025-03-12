from torch import nn
from torch import optim
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, dataset
import tqdm
# import pandas as pd
# from PIL import Image
# import io

transformer = transforms.Compose([transforms.Resize(size=(28, 28)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5], std=[0.5])])

train_dataset = datasets.MNIST(root='./mnist/',
                            train=True,
                            transform=transformer,
                            download=True)

test_dataset = datasets.MNIST(root='./mnist/',
                            train=False,
                            transform=transformer,
                            download=True)

# class MyDataset(dataset):
#     def __init__(self, file_path, transform=None):
#         super(MyDataset, self).__init__()
#         self.data = pd.read_parquet(file_path)
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, index):
#         img_bytes = self.data.iloc[index]['image']['bytes']
#         label = self.data.iloc[index]['label']

#         img = Image.open(io.BytesIO(img_bytes)).convert('L')

#         if self.transform is not None:
#             img = self.transform(img)
#         return img, label


batch_size = 64

train_loader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        shuffle=False)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(1,32,5,1,2),nn.BatchNorm2d(32) , nn.ReLU(), nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,5,1,2),nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2))
       
        self.fc1 = nn.Sequential(nn.Linear(64*7*7, 1000), nn.Dropout(0.5), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(1000, 10), nn.Softmax(dim=1))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

cross_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)


def train():
    model.train()
    for i, (inputs, labels) in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        out = model(inputs)
        loss = cross_loss(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Train Loss: {total_loss / len(train_loader)}")

#测试
def test():
    model.eval()
    correct = 0
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        out = model(inputs)
        _, predicted = torch.max(out.data, 1)
        correct += (predicted == labels).sum()
    print('Test Acc: {:.6f}%'.format(correct.item() / (len(test_dataset))*100))

#训练20轮
for epoch in range(20):
    print('epoch:{}'.format(epoch+1))
    train()
    test()