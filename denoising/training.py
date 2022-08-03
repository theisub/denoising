import torch
import numpy as np
from utils import MyDataSet, pad_collate, GRU
from torch.utils.data import DataLoader,BatchSampler,SequentialSampler
from matplotlib import pyplot as plt
import os
import torch.nn as nn
import tqdm
import argparse

parser = argparse.ArgumentParser(description='Args for training')

parser.add_argument('-batch_size', type=int,
                    help='A required integer to set batch_size')

parser.add_argument('-number_of_epochs', type=int, nargs='?',
                    help='An optional integer positional argument')

parser.add_argument('-train_path', type=str,
                    help='Path to train data')

parser.add_argument('-val_path', type=str,
                    help='Path to validation data')
                    
parser.add_argument('--plot_losses', action='store_true',
                    help='Option to plot losses', default=False)


dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

args = parser.parse_args()


print("Argument values:")
print('Batch size ',args.batch_size)
print('Number of epochs ',args.number_of_epochs)
print('Train path ',args.train_path)
print('Val_path ',args.val_path)
print('Plot losses: ', args.plot_losses)

model= GRU(80, 16, 80,bidirectional=True,num_directions=2)
model.to(dev)

print('Your device',torch.cuda.get_device_name(dev))

predictions = []

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_func = nn.MSELoss()


max_epochs = args.number_of_epochs if args.number_of_epochs is not None else 20
batch_size = args.batch_size if args.batch_size is not None else 256
train_path = args.train_path if args.train_path is not None else "./train/train/"
val_path = args.val_path if args.val_path is not None else "./val/val/"


dataset = MyDataSet(train_path)
dataset_val = MyDataSet(val_path)
train_loader = DataLoader(dataset, sampler=BatchSampler(SequentialSampler(dataset),batch_size=batch_size,drop_last=True),collate_fn=pad_collate)
valid_loader = DataLoader(dataset_val, sampler=BatchSampler(SequentialSampler(dataset_val),batch_size=batch_size,drop_last=True),collate_fn=pad_collate)


training_loss_plot = []
valid_loss_plot = []
best_epoch_info = {'best_training_loss':float('inf'),'best_validation_loss':float('inf'),'best_epoch_num':None}
epoch_ctr=0

for epoch in range(max_epochs):
    train_batches = 0
    valid_batches = 0
    train_loss=0.0
    valid_loss=0.0
    for noisy,clean,_,_ in train_loader:

        input = noisy.to(torch.float32).to(dev)
        output = clean.to(torch.float32).to(dev)
        loss = loss_func(input, output)

        pred = model(input).to(dev)
        optimizer.zero_grad()
        loss = loss_func(pred, output)
        train_loss +=loss.item()

        train_batches+=1

        loss.backward()
        optimizer.step()
        epoch_train_loss=train_loss/train_batches

    training_loss_plot.append(epoch_train_loss)

    for noisy,clean,_,_ in valid_loader:
        with torch.no_grad():
            inp_val = noisy.to(torch.float32).to(dev)
            out_val =  clean.to(torch.float32).to(dev)
            pred = model(inp_val).to(dev)
            optimizer.zero_grad()
            loss = loss_func(pred, out_val)
            valid_loss +=loss.item()
            valid_batches+=1
            epoch_valid_loss = valid_loss/valid_batches
    valid_loss_plot.append(epoch_valid_loss)

    epoch_ctr+=1
    print(f'Epoch {epoch_ctr}: Train loss {epoch_train_loss} ; Valid loss {epoch_valid_loss}')

    if epoch_train_loss < best_epoch_info['best_training_loss'] and epoch_valid_loss < best_epoch_info['best_validation_loss']:
        best_epoch_info['best_training_loss'] = epoch_train_loss
        best_epoch_info['best_validation_loss'] = epoch_valid_loss
        torch.save(model, 'best.pt')
        best_epoch_info['best_epoch_num']=epoch
        
    torch.save(model,'last.pt')

if (args.plot_losses):
    plt.plot(training_loss_plot,label='Training loss, MSE')
    plt.plot(valid_loss_plot,label='Validation loss, MSE')
    plt.xlabel(('Epoch. Best epoch',best_epoch_info['best_epoch_num']))
    plt.ylabel('MSE')
    plt.legend()
    plt.show()