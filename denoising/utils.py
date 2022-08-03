import numpy as np
import torch
from torchvision.datasets import DatasetFolder
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,BatchSampler,SequentialSampler

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample


class MyDataSet(torch.utils.data.Dataset):
  def __init__(self, path):
    super(MyDataSet, self).__init__()

    self.clean_data = DatasetFolder(
        root=path+'clean',
        loader=npy_loader,
        extensions=['.npy']
    )

    self.clean_data.targets= np.ones_like(self.clean_data.targets)
    self.noisy_data = DatasetFolder(
        root=path+'noisy',
        loader=npy_loader,
        extensions=['.npy']
    )

  def __len__(self):
    return len(self.noisy_data.samples)

  def __getitem__(self, index):
    if isinstance(index,slice):
        noisy = self.noisy_data[index.start:index.stop][0]
        clean = self.clean_data[index.start:index.stop][0]
    elif isinstance(index,list):
        noisy=[]
        clean=[]
        for i in index:
            noisy.append(self.noisy_data[i][0])
            clean.append(self.clean_data[i][0])
    else:
        noisy = self.noisy_data[index]
        clean = self.clean_data[index]

    return noisy, clean

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,bidirectional,num_directions):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                        batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size*num_directions, output_size, )
        self.act = nn.Tanh()
    def forward(self, x):
        pred, hidden = self.rnn(x, None)
        pred = self.act(self.linear(pred))
        return pred

def pad_collate(batch):
  (data1, data2) = zip(*batch)
  data1_lengths = [len(x) for x in data1[0]]
  data2_lengths = [len(y) for y in data2[0]]

  data1_padded = pad_sequence(data1[0], batch_first=True, padding_value=0)
  data2_padded = pad_sequence(data2[0], batch_first=True, padding_value=0)

  return data1_padded, data2_padded, data1_lengths, data2_lengths