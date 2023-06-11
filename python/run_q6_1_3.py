import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.io
import skimage.transform

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.optim as optim

from nn import *
from q4 import *

#refered to: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

max_iters = 20
# max_iters = 3

learning_rate = 0.001
batch_size = 100

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR10(root='./data', train=True,
                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = datasets.CIFAR10(root='./data', train=False,
                          download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

train_losses = []
# test_losses = []
train_accs = []
# test_accs = []
num_train = len(trainset)
num_test = len(testset)

for itr in range(max_iters):
    print('iteration: ', itr)
    total_loss = 0
    total_acc = 0
    test_total_loss = 0
    test_total_acc = 0
    for batch_idx, (x, target) in enumerate(trainloader):
        out =  model(x)
        loss = criterion(out,target)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        _, predicted = torch.max(out.data, 1)

        total_acc += ((target==predicted).sum().item())
        total_loss += loss.item()

    batch_num = batch_idx + 1
    train_avg_loss = total_loss / batch_num # Avg loss from batches
    
    train_acc = total_acc / num_train
    print('training loss: ' + str(train_avg_loss))
    print('training accuracy: ' + str(train_acc))
    train_losses.append(train_avg_loss)
    train_accs.append(train_acc)

plt.figure(0)
plt.xlabel('iters')
plt.ylabel('loss')
plt.plot(np.arange(max_iters), train_losses, label = 'loss')
plt.legend()

plt.figure(1)
plt.xlabel('iters')
plt.ylabel('accuracy')
plt.plot(np.arange(max_iters),train_accs, label = 'accuracy')
plt.show()