import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.io import read_image

max_iters = 10
batch_size = 100
learning_rate = 5e-4
hidden_size = 64

# Select readable jpg or png imgs path 
train_imgs_dirs = open('../../hw1/data/train_files.txt').read().splitlines()
rd_train_imgs_dirs = []
for path in train_imgs_dirs:
    try:
        img = read_image('../../hw1/data/' + path)
    except:
        continue
    rd_train_imgs_dirs.append(path)

test_imgs_dirs = open('../../hw1/data/test_files.txt').read().splitlines()
rd_test_imgs_dirs = []
for path in test_imgs_dirs:
    try:
        img = read_image('../../hw1/data/' + path)
    except:
        continue
    rd_test_imgs_dirs.append(path)

transform = transforms.Compose([transforms.Resize((224, 224))])

class SUNDataset(Dataset):
    def __init__(self, isTrainSet=True):
        if isTrainSet:
            self.img_dir = rd_train_imgs_dirs
            self.labels = np.loadtxt('../../hw1/data/train_labels.txt', np.int64)
        else:
            self.img_dir = rd_test_imgs_dirs
            self.labels = np.loadtxt('../../hw1/data/test_labels.txt', np.int64)

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img = read_image('../../hw1/data/'+self.img_dir[idx])
        img = transform(img)
        label = self.labels[idx]
        return img, label

train_data = SUNDataset()
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = SUNDataset(isTrainSet=False)
test_size = len(test_data)
test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class conv_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(conv_block,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=kernel_size,
                                out_channels=out_channels, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output

class SUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=8):
        super(SUNet, self).__init__()

        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(
                conv_block(in_channels, 16, 7, 3),
                nn.MaxPool2d(kernel_size=3),
                conv_block(16, 32, 5),
                nn.MaxPool2d(kernel_size=3),
                conv_block(32, 32, 3),
                nn.MaxPool2d(kernel_size=2),
                conv_block(32, 16, 3),
                nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Linear(in_features=256, out_features=out_channels)

    def forward(self, input):
        output = self.net(input)
        output = output.view(input.size(0), -1)
        output = self.fc(output)
        return output

model = SUNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = [] 
train_accs = []
test_losses = []
test_accs = []
for itr in range(max_iters):
    print("Itr: {:02d}".format(itr)) 
    model.train()
    total_loss = 0
    total_acc = 0
    for xb, yb in train_loader:
        xb = xb.float()
        
        optimizer.zero_grad()

        y_pred = model(xb)
        loss = criterion(y_pred, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        _, preds = torch.max(y_pred.data, 1)
        total_acc += torch.sum(preds == yb.data)

    train_acc = total_acc / len(train_loader.dataset)
    train_avg_loss = total_loss / len(train_loader.dataset)
    train_losses.append(train_avg_loss)
    train_accs.append(train_acc) 

    # Testing
    total_loss = 0
    total_acc = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.float()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            
            total_loss += loss.item() * xb.size(0)
            _, preds = torch.max(y_pred.data, 1)
            total_acc += torch.sum(preds == yb.data)
    
    test_avg_loss = total_loss / len(test_loader.dataset)
    test_acc = total_acc / len(test_loader.dataset)
    test_losses.append(test_avg_loss)
    test_accs.append(test_acc)

    print("training loss: {:.2f} \t accuracy : {:.2f}".format(train_avg_loss, train_acc))
    print("testing loss: {:.2f} \t accuracy : {:.2f} \n".format(test_avg_loss, test_acc))

plt.figure()
plt.xlabel('iters')
plt.ylabel('accuracy')
plt.plot(np.arange(max_iters), train_accs, label = 'training')
plt.plot(np.arange(max_iters), test_accs, label = 'testing')
plt.legend()
plt.show()