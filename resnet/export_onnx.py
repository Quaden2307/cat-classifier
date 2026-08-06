"""Export the trained ResNet to ONNX for browser inference.

Run once from the project root:  python -m resnet.export_onnx
Writes frontend/model.onnx, then verifies it against PyTorch on the full test split.
"""

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import v2

from data_prep.dataset import CatDataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

eval_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std),
])

# Rebuild the architecture, then load the trained weights. CPU on purpose:
# the exported graph must not reference MPS, and export needs no gradient math.
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load('resnet/resnet18_cats.pt', map_location='cpu'))
model.eval()

dummy = torch.zeros(1, 3, 224, 224)
torch.onnx.export(
    model, (dummy,), 'frontend/model.onnx',
    input_names=['image'], output_names=['logits'],
    external_data=False,  # single self-contained file — the browser fetches exactly one URL
)

# onnxruntime-web lags the Python package on ONNX file-format ("IR") versions;
# clamping to 10 keeps the browser runtime happy and changes no math.
m = onnx.load('frontend/model.onnx')
print(f"exported frontend/model.onnx  (opset {m.opset_import[0].version}, ir_version {m.ir_version})")
if m.ir_version > 10:
    m.ir_version = 10
    onnx.save(m, 'frontend/model.onnx')
    print("clamped ir_version to 10 for onnxruntime-web")

# Parity check: every test image through both runtimes, predictions must agree.
session = ort.InferenceSession('frontend/model.onnx')
test = CatDataset(csv_path='data/splits.csv', split='test', transform=eval_transform)
loader = DataLoader(test, batch_size=1, shuffle=False)

max_diff = 0.0
agree = 0
with torch.no_grad():
    for image, label in loader:
        torch_logits = model(image).numpy()
        onnx_logits = session.run(None, {'image': image.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(torch_logits - onnx_logits).max()))
        agree += int(torch_logits.argmax() == onnx_logits.argmax())

print(f"predictions agree on {agree}/{len(test)} test images")
print(f"max logit difference: {max_diff:.2e}")
