# Project context

## What this project is

A cat image classifier, built in PyTorch. The images are scraped and labeled by me, not
taken from a prepared dataset. The eventual goal is a CNN.

This is a **learning project**. The point is understanding how the models work, not
shipping a product.

## How I want you to work with me

**I write the implementation code.** Default to explaining, reviewing, and verifying
rather than editing files. Ask before changing my code. Fixing environment/tooling
problems, writing throwaway diagnostic scripts, and checking my math are all welcome.

Don't refactor working code into a "cleaner" structure unless I ask. Don't add
abstractions I didn't request.

## What I already know — calibrate to this

I just finished writing an MNIST digit classifier from scratch in raw NumPy, no
frameworks. I derived every gradient by hand. It reached 98.08% test accuracy.

So I already understand, at the level of having implemented it myself:

- Forward pass through a two-layer MLP (784 → 128 ReLU → 10 softmax), He initialization
- Softmax, cross-entropy loss, and why they're paired — including how the log cancels
  the exp so the gradient collapses to `P - Y`
- Backpropagation and the chain rule, derived by hand for every parameter
- Why broadcasting forward means summing backward (bias gradients)
- That loss and gradient must normalize by the same sample count
- Mini-batch SGD: batches, epochs, shuffling, why more updates beats more exact
  gradients (my full-batch version got 92.7% in 182s; mini-batch got 98.1% in 6.4s)
- Train/validation/test splits, overfitting, the generalization gap
- Numerical gradient checking against finite differences

**Do not re-explain these basics.** Assume I know the math and go straight to what's new.

What I have NOT used before: PyTorch, any autograd framework, CNNs, real image data,
transfer learning, data augmentation.

## Environment

- Apple Silicon (arm64), macOS 15.3.1 — PyTorch MPS backend available, use `device="mps"`
- System Python is 3.9.6; installing 3.11+ for current PyTorch
- Stack: `torch`, `torchvision`

## The plan

1. **Port the MNIST MLP to PyTorch first.** Same architecture, known answer (98.08%).
   Learn `nn.Module` / `DataLoader` / optimizer loop against a verifiable target before
   introducing new data.
2. **Build the data pipeline.** Source, download, dedupe, inspect, resize, split
   train/val/test. Dedupe before splitting; split before augmenting.
3. **MLP on cat images.** New data, known architecture. Expect poor results — this is a
   baseline to beat.
4. **CNN.** New architecture, known-good data pipeline.
5. **Fine-tune a pretrained ResNet** and compare against the from-scratch CNN.

The principle: change one thing at a time, so a failure is always attributable.

## Things I specifically need to unlearn or watch

- My NumPy code was **column-major** — every matrix was `(features, samples)`, e.g. a
  batch was `784 × 64`. PyTorch is row-major, `(batch, features)`. This will feel wrong
  and I will get shapes backwards.
- `optimizer.zero_grad()` — gradients accumulate in PyTorch. My NumPy code overwrote
  them each pass, so I have no instinct for this.
- `nn.CrossEntropyLoss` takes raw logits and applies log-softmax internally. I'm used to
  computing softmax myself, so I'm likely to double-apply it.
- `model.eval()` / `torch.no_grad()` at inference time.
