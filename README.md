# Cat classifier

**Live demo: https://cat-classifier-five.vercel.app/**

Upload a photo of a cat and a fine-tuned resnet18 decides whether it's a ragdoll.
The model runs entirely in your browser through ONNX, so the photo never leaves
your device.

## Why I built this

I wanted to actually understand neural networks and computer vision, not just call
libraries. Before this project I wrote a digit classifier in raw NumPy with every
gradient derived by hand, no frameworks, which got 98.08% on MNIST. That covered
the math. This project is the next step: real image data, PyTorch, CNNs, and
transfer learning, ending in something deployed instead of a script that only
runs on my machine.

## How it was built

The rule I followed the whole way: change one thing at a time, so when something
breaks I know what broke it.

1. **Ported the NumPy MLP to PyTorch.** Same architecture, same MNIST data, known
   answer. If the PyTorch version didn't hit ~98% I'd know the problem was my
   PyTorch, not the model.
2. **Built the data pipeline.** 949 cat photos across 5 breeds from
   [this Kaggle dataset](https://www.kaggle.com/datasets/yapwh1208/cats-breed-dataset).
   The set claimed to be deduplicated but two images appeared in both the ragdoll
   and siamese folders with identical bytes, which would have leaked contradictory
   labels across the train/test split. Cleaned that up, split 70/15/15 stratified,
   and wrote the split to a csv so every model gets scored on the exact same test
   set. Later grew the dataset to 1,722 by adding the overlapping breeds from the
   [Oxford-IIIT Pet dataset](https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset),
   with the same dedupe-then-resplit process.
3. **MLP baseline on the cat images.** Flattened pixels into a two-layer network.
   This is supposed to be bad. It's the number the real models have to beat.
4. **Fine-tuned a pretrained resnet18.** Froze the ImageNet weights, swapped the
   final layer for a 2-class head, trained that, then unfroze the last
   convolutional block at a much lower learning rate. This is the model that ships.
5. **Deployed it.** Exported the weights to ONNX, verified the exported model gives
   identical predictions to the PyTorch one on the full test set, and built a
   static frontend that runs inference in the browser with onnxruntime-web. No
   server, nothing uploaded.

Next up: a CNN written from scratch to compare against the pretrained one, and
after that, classifying all 5 breeds instead of just ragdoll or not.

## Results

Ragdolls are 22.9% of the data, so accuracy is the wrong metric — a model that
always says "not ragdoll" is 77.1% accurate and completely useless. Everything is
scored with F1 on the ragdoll class instead.

That imbalance bit me in training too: with a normal loss the MLP collapsed to
predicting "not ragdoll" for every single image while the loss kept improving.
The fix was weighting the loss so the ragdolls pull on the gradient as hard as
the majority class.

| Model | Ragdoll F1 |
| --- | --- |
| MLP baseline | 0.52 (val, on the earlier 949-photo split) |
| resnet18 fine-tuned | **0.832** (test) |

Same data, same training loop, same loss. The gap is the architecture — the MLP
can't see spatial structure in flattened pixels, and the pretrained ResNet starts
with features learned from a million images.

Target is F1 ≥ 0.90, and 0.832 isn't that, which is why the site tells you the
judge will not always be right.

## Repo layout

- `data_prep/` — dataset class and the split builder
- `nn/` — MLP baseline
- `resnet/` — the shipped model, training and ONNX export
- `frontend/` — the site (static HTML/CSS/JS + the ONNX model)
- `docs/` — day-by-day progress log and decisions, including the failures
