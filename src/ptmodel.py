from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import manifold
from matplotlib import offsetbox
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sys


class Net(nn.Module):
  def __init__(self):
    super().__init__()

    # define layers
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3)
    self.conv2 = nn.Conv2d(in_channels=9, out_channels=16, kernel_size=2)
    self.bn = nn.BatchNorm2d(num_features=9)
    self.fc1 = nn.Linear(in_features=16*2*2, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    self.fc3 = nn.Linear(in_features=60, out_features=1)
  # define forward function

  def forward(self, t, outls=False):
    # conv 1
    t = self.conv1(t)
    t = self.bn(t)
    t = F.relu(t)

    # conv 2
    t = self.conv2(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2)
    # fc1
    t = t.reshape(-1, 16*2*2)
    t = self.fc1(t)
    t = F.relu(t)
    embed = t
    # fc2
    t = self.fc2(t)
    t = F.relu(t)
    #fc3, output
    t = self.fc3(t)

    if outls:
        return(t, embed)
    else:
        return(t)


def train_epoch(model, device, train_loader, optimizer, epoch, retloss=False):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    losslist = []
    for batch_idx, data in enumerate(train_loader):
        data, target = data['X'].to(
          device, dtype=torch.float), data['Y'].to(device, dtype=torch.float)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)             # Make predictions
        loss = F.mse_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))
        if retloss:
            losslist.append(loss.item())
    if retloss:
        return(np.mean(losslist))


def test(model, device, test_loader, retloss=False):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for batch_idx, data in enumerate(test_loader):
            data, target = data['X'].to(
              device, dtype=torch.float), data['Y'].to(device, dtype=torch.float)
            output = model(data)
            # sum up batch loss
            test_loss += F.mse_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))
    if retloss:
        return(test_loss)

def gen_split(dataset,valper=.15,batchsize=50):
    train_dataset = dataset
    valinds=[]

    valinds=np.random.choice(np.arange(0,len(dataset)),int(len(dataset)*.15),replace=False)

    traininds=np.array([x for x in np.arange(0,len(dataset)) \
               if x not in valinds])
    subset_indices_train = traininds
    subset_indices_valid = valinds


    np.random.shuffle(subset_indices_train)
    np.random.shuffle(subset_indices_valid)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(valinds),
        sampler=SubsetRandomSampler(subset_indices_valid)
    )
    return(train_loader,val_loader)


class ICEsDataset(Dataset):

    def __init__(self, datadict):
        #DDR not yet implemented
        # self.ddr = datadict['DDR']
        self.ctx = np.array(datadict['CTX'])
        self.crism = np.array(datadict['CRISM'])
        self.coords = datadict['COORDS']
        assert(len(self.ctx) == len(self.crism))

    def __len__(self):
        return len(self.ctx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'X': self.ctx[idx], 'Y': self.crism[idx]}

        return sample

    def normalize(self):
        flatcris = self.crism.flatten()
        min, max = np.min(flatcris), np.max(flatcris)
        if (min != 0) & (max != 1):
                self.crism = (self.crism+abs(min))/(max+abs(min))
        else:
            print('CRISM Already Normalized!')
            sys.exit()

        flatctx = self.ctx.flatten()
        min, max = np.min(flatctx), np.max(flatctx)
        if (min != 0) & (max != 1):
                self.ctx = (self.ctx+abs(min))/(max+abs(min))
        else:
            print('CTX Already Normalized!')
            sys.exit()
