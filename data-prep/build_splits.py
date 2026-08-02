import sklearn as sklearn
import pandas as pd
import numpy as np
import torch as torch
import torchvision.transforms.v2 as v2
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split


#Creating CSV File
DATA_DIR = Path('data/cat_v1')
EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
field_names = ['id', 'image_path', 'breed', 'ragdoll', 'split']
rows = []
for breed_dir in sorted(DATA_DIR.iterdir()):
    if breed_dir.is_dir():
        breed = breed_dir.name
        ragdoll = 1 if breed == 'ragdoll' else 0
        for image_path in sorted(breed_dir.iterdir()):
            if image_path.suffix.lower() in EXTS:
                rows.append({
                    'id': image_path.stem,
                    'image_path': str(image_path),
                    'breed': breed,
                    'ragdoll': ragdoll,
                })

#Splitting 70/15/15, stratified so each pile keeps the same ragdoll proportion
labels = [r['ragdoll'] for r in rows]
train_rows, temp_rows = train_test_split(
    rows, test_size=0.30, stratify=labels, random_state=42) #--> temp_rows to split in half for val and test
 
temp_labels = [r['ragdoll'] for r in temp_rows]
val_rows, test_rows = train_test_split(
    temp_rows, test_size=0.50, stratify=temp_labels, random_state=42)

for r in train_rows:
    r['split'] = 'train'
for r in val_rows:
    r['split'] = 'val'
for r in test_rows:
    r['split'] = 'test'

#Writing CSV File
with open('data/splits.csv', 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(rows)

