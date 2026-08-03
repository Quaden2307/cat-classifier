from data_prep.dataset import CatDataset, eval_transform, train_transform
from torchvision.transforms import v2
import torch as torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import PIL.Image as Image
from pathlib import Path   


train = CatDataset(csv_path='data/splits.csv', split='train', transform=train_transform)
val = CatDataset(csv_path='data/splits.csv', split='val', transform=eval_transform)
test = CatDataset(csv_path='data/splits.csv', split='test', transform=eval_transform)   

train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
val_loader = DataLoader(val, batch_size=32, shuffle=False, num_workers=0, drop_last = False)
test_loader = DataLoader(test, batch_size=32, shuffle=False, num_workers=0, drop_last=False)

class CatMLP(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super(CatMLP, self).__init__()
        self.layer1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden, out_features)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x