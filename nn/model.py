from data_prep.dataset import CatDataset, eval_transform, train_transform
from torchvision.transforms import v2
import torch as torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import PIL.Image as Image
from pathlib import Path   
from sklearn.metrics import precision_recall_fscore_support



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

model = CatMLP(in_features=224*224*3, hidden=128, out_features=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad() #reset the gradient values 
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        predictions = []
        labels = []
        for images, val_labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, val_labels)
            val_loss += loss.item()
            predictions.extend(outputs.argmax(dim=1).tolist())
            labels.extend(val_labels.tolist())

    p, r, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1, zero_division=0)
    print(f"epoch {epoch+1:2}  train {train_loss/len(train_loader):.4f}  "
        f"val {val_loss/len(val_loader):.4f}  P {p:.3f}  R {r:.3f}  F1 {f1:.3f}")



