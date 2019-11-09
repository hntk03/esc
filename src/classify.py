# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np

from torchvision.models import vgg16_bn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
import torch.nn as nn
import torch.optim as optim


class ESCDataset(Dataset):
    
    def __init__(self, datapath, meta, transform=None):
        
        self.data = np.load(datapath)
        self.target = pd.read_csv(meta)['target'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        data = self.data[idx]
        target = self.target[idx]
        
        if self.transform:
            data = self.transform(data)
        
        return data, target


# +
def train(net, train_loader):
    net.train()
    running_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad() 
        outputs = net(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predict = torch.max(outputs.data, 1)
        correct += (predict == labels).sum().item()
        total += labels.size(0)
        
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    
    return net, train_loss, train_acc

def valid(net, valid_loader):
    net.eval()
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        
        for batch_idx, (images, labels) in enumerate(valid_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predict = torch.max(outputs.data, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)
            
    val_loss = running_loss / len(valid_loader)
    val_acc = correct / total
    
    return net, val_loss, val_acc


# -

mean = [-30.0007]
std =  [21.8174]

transforms = Compose([ToTensor(), Normalize(mean, std)])

dataset = ESCDataset('../data/melspectrogram.npy', '../data/ESC-50/meta/esc50.csv', transforms)

num_folds = 5
num_classes = 50

num_epochs = 100
batch_size = 64
learning_rate = 0.01

device = 'cuda:2'
criterion = nn.CrossEntropyLoss()

index = np.arange(0, len(dataset))
index

# +
cv = 0.0

for fold_idx in range(num_folds):
    
    valid_idx = np.load('../data/fold' + str(fold_idx) + '.npy')
    train_idx = np.setdiff1d(index, valid_idx)
    
    print('fold {}'.format(fold_idx))
    net = vgg16_bn(num_classes=num_classes)
    net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(Subset(dataset, valid_idx), shuffle=False, batch_size=batch_size)
    
    for epoch_idx in range(num_epochs):
        
        net, train_loss, train_acc = train(net, train_loader)
        net, valid_loss, valid_acc = valid(net, valid_loader)
        
        print('train_loss {:.3f} valid loss {:.3f} train_acc {:.3f} valid_acc {:.3f}'.format(train_loss, valid_loss, train_acc, valid_acc))
        
    cv += valid_acc / num_folds
# -

print(cv)








