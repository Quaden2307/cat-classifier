# Project context

## What this project is

A cat classifier built in PyTorch, ending in a CNN. It is both a **learning project**
and a **portfolio piece** — the point is understanding how the models work, but the
result gets deployed and shown to recruiters.

Deadline: **August 31, 2026.**

Current task: **ragdoll vs. not-ragdoll** (binary). Then 5-breed classification. Then
deployment, CNN only — the MLP is a baseline for comparison and is not shipped.

Data: 953 images across 5 breeds in `data/cat_v1`, from Kaggle (link in
`data/dataset_link`). Ragdoll is 207 of them, 21.7% — the classes are imbalanced, which
matters when scoring the model but not when building it. Success criteria live in
`docs/project-notes.md`.

Longer-form detail lives in `docs/project-notes.md` (decisions, open questions, known
data issues) and `docs/progress-log.md` (day-by-day). **`docs/` is public** — no
secrets, no local paths, nothing that shouldn't be read by a stranger.

## How I want you to work with me

**I write the implementation code.** Default to explaining, reviewing, and verifying
rather than editing files. Ask before changing my code.

You may edit **documentation and security** directly. Environment/tooling fixes,
throwaway diagnostic scripts, and checking my math are all welcome.

**Never run git commands** — no commits, no pushes. Give me the commands and I'll run
them. Never appear as a contributor on this project.

Don't refactor working code into a "cleaner" structure unless I ask. Don't add
abstractions I didn't request.

**Be concise.** Lead with the most important point and give a recommendation, not a
survey of every consideration. Long-form context belongs in `docs/`, not in a reply.

## What I already know — calibrate to this

I wrote an MNIST digit classifier from scratch in raw NumPy, no frameworks, deriving
every gradient by hand. It reached 98.08% test accuracy.

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

Define ML jargon the first time it comes up rather than using it bare. I know the math;
I don't always know the standard name for it.

## The plan

1. **Port the MNIST MLP to PyTorch.** Same architecture, known answer (98.08%). Learn
   `nn.Module` / `DataLoader` / optimizer loop against a verifiable target before
   introducing new data. ← current step
2. **Build the data pipeline.** Dedupe, inspect, resize, split train/val/test. Dedupe
   before splitting; split before augmenting. Persist the split to a manifest so every
   model is evaluated on the same test set.
3. **MLP on cat images.** New data, known architecture. Expect poor results — this is a
   baseline to beat.
4. **CNN.** New architecture, known-good data pipeline.
5. **Fine-tune a pretrained ResNet** and compare against the from-scratch CNN.

The principle: change one thing at a time, so a failure is always attributable.

Shared data prep is used by both the MLP and the CNN — same manifest, same label
mapping, same loading path. Resolution, channel count, and whether the output is
flattened are parameterized, since the two models need different tensor shapes.

## Things I specifically need to unlearn or watch

- My NumPy code was **column-major** — every matrix was `(features, samples)`, e.g. a
  batch was `784 × 64`. PyTorch is row-major, `(batch, features)`. This will feel wrong
  and I will get shapes backwards.
- `optimizer.zero_grad()` — gradients accumulate in PyTorch. My NumPy code overwrote
  them each pass, so I have no instinct for this.
- `nn.CrossEntropyLoss` takes raw logits and applies log-softmax internally. I'm used to
  computing softmax myself, so I'm likely to double-apply it.
- `model.eval()` / `torch.no_grad()` at inference time.

## Environment

Apple Silicon, macOS 15.3.1 — use `device="mps"`. Python 3.12 in `venv/`. Stack: `torch`,
`torchvision`, `numpy`, `pillow`, `matplotlib`, `scikit-learn`.
