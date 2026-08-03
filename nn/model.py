from dataset import CatDataset, eval_transform, train_transform
from torchvision.transforms import v2
import torch
import pandas as pd
import PIL.Image as Image
from pathlib import Path   

nn.Flatten(CatDataset)

train = CatDataset(csv_path='data/splits.csv', split='train', transform=train_transform)
val = CatDataset(csv_path='data/splits.csv', split='val', transform=eval_transform)
test = CatDataset(csv_path='data/splits.csv', split='test', transform=eval_transform)   

