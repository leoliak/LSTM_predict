# Define options
import argparse
parser = argparse.ArgumentParser(description="Template")
#
#

# Model options
parser.add_argument('-ll', '--lstm-layers', default=1, type=int, help="LSTM layers")
parser.add_argument('-ls', '--lstm-size', default=128, type=int, help="LSTM hidden size")
parser.add_argument('-os', '--output-size', default=128, type=int, help="output layer size")
# Training options
parser.add_argument("-b", "--batch_size", default=1, type=int, help="batch size")
parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="learning rate")
parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.2, type=float, help="learning rate decay factor")
parser.add_argument('-lrde', '--learning-rate-decay-every', default=50, type=int, help="learning rate decay period")
parser.add_argument('-dw', '--data-workers', default=4, type=int, help="data loading workers")
parser.add_argument('-e', '--epochs', default=500, type=int, help="training epochs")
# Backend options
parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")

# Parse arguments
opt = parser.parse_args()

# Imports
import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import imageio

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Dataset class
class TrajectoriesDataset:  
    # Constructor
    def __init__(self, dataset):
        # Load data
        self.data = scio.loadmat(dataset)['coords'][0];
        # Compute size
        self.size = len(self.data)
    # Get size
    def __len__(self):
        return self.size
    # Get item
    def __getitem__(self, i):
        data = self.data[i].copy()
        if(torch.randn(1)>0.5):
          data = np.flip(data,0).copy() 
        x = np.random.randint( int(data.shape[0]/4),int(data.shape[0]/2)-1,1)[0];
        input= data[x:x+int(data.shape[0]/2),:]
        target = data[x+1+int(data.shape[0]/2),:]
        # print(input.shape)
        return input, target

dataset  = {'train':TrajectoriesDataset(dataset='sample_large2.mat'), 'val':TrajectoriesDataset(dataset='sample_large1.mat')}
# Create loaders
loaders = {split: DataLoader(dataset[split], batch_size = opt.batch_size, drop_last = True, shuffle = True) for split in ['train','val']}


# for data in dataset:
#     plt.plot(data[0][:,1], data[0][:,0])
#     plt.show()
# Define model
class Model(nn.Module):
    def __init__(self, input_size, lstm_size, lstm_layers):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        # self.output_size = output_size
        # Define internal modules
        self.lstm = nn.GRU(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
        self.output = nn.Linear(10, 2)
    def forward(self, x):
        # Prepare LSTM initiale state
        batch_size = x.size(0)
        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size), torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        if x.is_cuda: lstm_init = (lstm_init[0].to(device), lstm_init[0].to(device))
        lstm_init = (lstm_init[0], lstm_init[1])
        # Forward LSTM and get final state
        x, h = self.lstm(x) #, lstm_init)
        return self.output(x[:,-1,:])
opt.lstm_layers =1;
model = Model(2, 10, 1)

optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr = opt.learning_rate)

criterion = nn.MSELoss()
criterion = criterion.to(device)
# Setup CUDA
if not opt.no_cuda:
    model.to(device)
    print("Copied to CUDA")

# Start training
for epoch in range(1, 500):
    # Initialize loss/accuracy variables
    losses = {'train': 0, 'val':0}
    counts = {'train': 0, 'val':0}
    # Adjust learning rate for SGD
    if opt.optim == "SGD":
        lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # Process each split
    for split in ['train', 'val']:
        # Set network mode
        if split == 'train':
            model.train()
        else:
            model.eval()
        # Process all split batches
        # for i, (input, target) in enumerate(dataset[split]):
        for i, (input, target) in enumerate(loaders[split]):
            # plt.plot(input[:,0], input[:,1])
            # plt.show()
            # input = torch.from_numpy(input)
            input = input.float()
            input[0,:,0] = 1.15*(-240 + input[0,:,0])/480.0
            input[0,:,1] = 1.15*(-320 + input[0,:,1])/640.0
            # target = torch.from_numpy(target)
            target[0,0]  = 1.15*(-240 + target[0,0] )/480.0
            target[0,1]  = 1.15*(-320 + target[0,1] )/640.0
            target = target.float()
            # Check CUDA
            # if not opt.no_cuda:
            #     input = input.to(device)
            #     input = input.float()
            #     target = target.to(device)
            #     target = target.float()
            optimizer.zero_grad()
            with torch.set_grad_enabled(split == 'train'):
                # Forward
                output = model(input)
                loss = criterion(output, target)
                losses[split] += loss.item()
                if(split=='train'):
                    # Backward and optimize
                    loss.backward()
                    optimizer.step()
                if(split=='val'): # get some more steps
                    multiple_outputs= output.clone()
                    input_clone = input.clone()
                    for i in range(0,6):
                        input_clone = torch.cat([input_clone,output.unsqueeze(0)],1);
                        output = model(input_clone)
                        multiple_outputs = torch.cat([multiple_outputs,output],0);
            counts[split] += 1
        # print('split: {} , loss {}'.format(split,loss.item()))
        input = input.numpy()
        target = target.numpy()
        viewout = output.clone()
        if(split=='val'):
            viewout = multiple_outputs.clone()
        viewout = viewout.detach().numpy()
        if(epoch%1 == 0 and split=='val' ):
            plt.clf()
            plt.ylim((-1.8, 1.8))   # set the ylim to bottom, top
            plt.xlim((-1.8, 1.8))   # set the ylim to bottom, top
            plt.plot(input[0][:,0], input[0][:,1],'b')
            plt.plot(input[0][:,0], input[0][:,1],'b.',markersize=4)
            plt.plot(target[0][0], target[0][1],'g*')
            plt.plot(viewout[0][0], viewout[0][1],'r*')
            if(split=='val'):
                plt.plot(viewout[1][0], viewout[1][1],'y.')
                plt.plot(viewout[2][0], viewout[2][1],'y.')
                plt.plot(viewout[3][0], viewout[3][1],'y.')
            plt.title('Test Set epoch'+str(epoch))
            plt.show(block = False)
            plt.pause(0.2)
            plt.savefig('epoch_' + str(epoch + 1) + '.png' )
    # Print info at the end of the epoch
    print("Epoch {}: TrainLoss={} ValLoss={} ".format(epoch, losses['train']/counts['train'],losses['val']/counts['val'] ))


images = []
for e in range(1,epoch-1):
    img_name = 'epoch_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))

imageio.mimsave('generation_animation.gif', images, fps=5)