from torchvision.transforms import v2
import torch
import pandas as pd
import PIL.Image as Image
from pathlib import Path    

eval_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    #v2.Normalize(mean, std),
])

train_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    #v2.Normalize(mean, std),
])


class CatDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path, split, transform=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['split'] == split]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]
        image_path = Path(row['image_path'])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

train = CatDataset(csv_path='data/splits.csv', split='train', transform=train_transform)
val = CatDataset(csv_path='data/splits.csv', split='val', transform=eval_transform)
test = CatDataset(csv_path='data/splits.csv', split='test', transform=eval_transform)