import numpy as np
import scipy.io
from nn import *

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import skimage.measure
import torch.optim as optim


# NIST36 dataset
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
xb, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 10
batch_size = 100
learning_rate = 1e-2
hidden_size = 64

train_x = torch.tensor(train_x).float()
train_label = np.where(train_y == 1)[1]
train_label = torch.tensor(train_label)
xb = torch.tensor(xb).float()
valid_label = np.where(valid_y == 1)[1]
valid_label = torch.tensor(valid_label)

train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(train_x,train_label),
                                           batch_size=batch_size,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(xb,valid_label),
                                           batch_size=batch_size,
                                           shuffle=True)

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(5 * 5 * 50, 512)
        self.fc2 = torch.nn.Linear(512, 36) 

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))       
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 5* 5 * 50)    
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ConvNet()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

train_losses = []
valid_losses = []
train_accs = []
valid_accs = []
train_num = train_x.shape[0]
valid_num = xb.shape[0]
for itr in range(max_iters):
    # Training
    total_loss = 0
    total_acc = 0
    for batch_idx, (xb, labels) in enumerate(train_loader):
        xb = xb.reshape(batch_size, 1, 32, 32)
        out =  model(xb)
        
        loss = criterion(out, labels)
        
        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()
        _, predicted = torch.max(out.data, 1)

        total_acc += ((labels==predicted).sum().item())
        total_loss += loss
    
    train_acc = total_acc / train_num
    batch_num = batch_idx + 1
    train_avg_loss = total_loss / batch_num # Avg loss from batches

    # Validation
    valid_total_loss = 0
    valid_total_acc = 0
    for batch_idx, (xb, labels) in enumerate(valid_loader):
        xb = xb.reshape(batch_size, 1, 32, 32) 
        valid_out =  model(xb)
        
        valid_loss = criterion(valid_out,labels)
        
        _, valid_predicted = torch.max(valid_out.data, 1)
        
        valid_total_acc += ((labels==valid_predicted).sum().item())
        valid_total_loss += valid_loss
    
    batch_num = batch_idx + 1
    valid_avg_loss = valid_total_loss / batch_num # Avg loss from batches

    valid_acc = valid_total_acc/valid_num

    train_losses.append(train_avg_loss.item())
    valid_losses.append(valid_avg_loss.item())
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, train_avg_loss, train_acc))

print('Validation accuracy: ', valid_acc)

plt.figure(0)
plt.xlabel('iters')
plt.ylabel('loss')
plt.plot(np.arange(max_iters), train_losses, label = 'training')
plt.plot(np.arange(max_iters),valid_losses, label = 'validation')
plt.legend()
# plt.show()

plt.figure(1)
plt.xlabel('iters')
plt.ylabel('Accuracy')
plt.plot(np.arange(max_iters), train_accs, label = 'training')
plt.plot(np.arange(max_iters), valid_accs, label = 'validation')
plt.legend()
plt.show()
