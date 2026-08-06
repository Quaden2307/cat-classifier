from data_prep.dataset import CatDataset
from torchvision import models
from torchvision.transforms import v2
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

torch.manual_seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.RandomHorizontalFlip(p=0.5),  #train only: a mirrored ragdoll is still a ragdoll
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std),
])

eval_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std),
])

train = CatDataset(csv_path='data/splits.csv', split='train', transform=train_transform)
val = CatDataset(csv_path='data/splits.csv', split='val', transform=eval_transform)
test = CatDataset(csv_path='data/splits.csv', split='test', transform=eval_transform)

train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
val_loader = DataLoader(val, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
test_loader = DataLoader(test, batch_size=32, shuffle=False, num_workers=0, drop_last=False)

#Load resnet18 with its pretrained ImageNet weights (downloads ~45 MB on first run)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 2)

for param in model.layer4.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.61]).to(device))
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(),     'lr': 1e-3},
])

epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
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
            images, val_labels = images.to(device), val_labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, val_labels)
            val_loss += loss.item()
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            labels.extend(val_labels.cpu().tolist())

    p, r, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1, zero_division=0)
    print(f"epoch {epoch+1:2}  train {train_loss/len(train_loader):.4f}  "
          f"val {val_loss/len(val_loader):.4f}  P {p:.3f}  R {r:.3f}  F1 {f1:.3f}")

#Test block commented out while iterating 
model.eval()
predictions = []
truths = []
with torch.no_grad():
    for images, test_labels in test_loader:
        outputs = model(images.to(device))
        predictions.extend(outputs.argmax(dim=1).cpu().tolist())
        truths.extend(test_labels.tolist())

p, r, f1, _ = precision_recall_fscore_support(
    truths, predictions, average='binary', pos_label=1, zero_division=0)
print(f"\nTEST  P {p:.3f}  R {r:.3f}  F1 {f1:.3f}")
print("confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(truths, predictions))

torch.save(model.state_dict(), 'resnet/resnet18_cats.pt')
print("saved weights to resnet/resnet18_cats.pt")
