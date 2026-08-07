# Project notes

Running notes on scope, decisions, and known issues. Day-by-day progress is tracked
separately in `progress-log.md`.

## Goal and deadline

Full project complete by **August 31, 2026**.

**Success criterion: F1 ≥ 0.90 on the ragdoll class**, with precision, recall, and a
confusion matrix reported alongside. Accuracy is not the target on its own: the classes
are imbalanced 21.7% / 78.3%, so a model that always answers "not ragdoll" scores 78.3%
without learning anything, and even a genuinely 90%-accurate model can miss nearly half
the ragdolls while hitting that number.

The model is the point. Deployment is a follow-on task once the model is done, and is
not expected to be a large effort.

## Scope

**Phase 1 — ragdoll / not-ragdoll (binary).** Current focus. Both classes come from the
existing 5-breed set, so no negative class needs sourcing — and there is no risk of the
model separating on resolution or compression artifacts instead of on the animal. The
negatives contain the two hardest confusers: siamese (also colorpoint) and maine coon
(also long-haired and large).

**Phase 2 — breed classification.** The existing `data/cat_v1` is already breed-labeled
across 5 classes. More breeds may be added later, so nothing should assume a fixed
class count.

**Phase 3 — deployment.** CNN only. The MLP is a baseline for comparison and is not
shipped.

Staged model plan (from `CLAUDE.md`): PyTorch MNIST port → data pipeline → MLP on cat
images → CNN → fine-tuned pretrained ResNet. One change at a time, so any failure is
attributable.

## Decisions made

- **Shared data-prep module** (`data_prep/`) used by both the MLP and the CNN. Shares the
  manifest, label mapping, and file loading. Split into two files: `prep.py` builds the
  manifest and is run once; `dataset.py` defines `CatDataset` and is imported. Keeping
  them separate means importing the dataset can never regenerate the split.
- **PIL, not OpenCV.** `cv2.imread` returns `None` on undecodable files instead of
  raising, and OpenCV loads BGR while all of torchvision assumes RGB. A channel swap
  would go unnoticed in the from-scratch models but would quietly degrade the
  pretrained ResNet, whose normalization is per-channel.
- **Subclass `torch.utils.data.Dataset`**, hand it to `DataLoader`. Writing
  `__len__`/`__getitem__` by hand is the part worth understanding; batching, shuffling,
  and worker parallelism come from `DataLoader`.
- **Persist the train/val/test split** to `data/splits.csv` rather than recomputing it
  per run. Schema is `id,image_path,breed,ragdoll,split`. A fixed seed is not enough once
  files are added — and data *will* be added (more breeds later). Without a stable split,
  "the CNN beat the MLP" is not a valid comparison. The `breed` column is carried even
  though phase 1 ignores it, so phase 2 does not require regenerating the file.
- **Split 70/15/15, stratified on the ragdoll label**, seeded. Stratifying keeps each
  pile at the same 21.7% ragdoll share; without it the ~143-image test pile varies by
  roughly ±4.5 ragdolls, which moves F1 on its own.
- **Center crop, not squash**, for aspect ratio — `Resize(256)` then `CenterCrop(224)`.
  224 matches what the pretrained ResNet expects at stage 5.
- **Transforms are attached per split**, not baked into the dataset class. Train, val,
  and test each get their own pipeline so augmentation can never reach val or test.
- Dedupe before splitting; split before augmenting.
- **Class-weighted loss**: `CrossEntropyLoss(weight=[1.0, 3.61])`, where 3.61 = 520/144,
  the non-ragdoll : ragdoll ratio in the training split. Each ragdoll's loss counts
  3.61×, so the 144 ragdolls pull on the gradient as hard as the 520 negatives. Without
  it the MLP collapsed to always predicting "not ragdoll" — F1 0.000 across 10 epochs
  while loss still fell, since the majority class is a cheap minimum under imbalance.
  Applies to every model trained on this split, for comparability; recompute the ratio
  if the split ever changes. To recompute: in `data/splits.csv`, take the rows where
  `split == train` and divide the count with `ragdoll == 0` by the count with
  `ragdoll == 1`. Note the printed loss becomes a weighted average, so runs with and
  without the weight are not comparable on loss.

## Open decisions

- **Where the flatten happens.** Either the dataset returns a flat vector for the MLP and
  a `(3, H, W)` tensor for the CNN, or it always returns `(3, H, W)` and the MLP starts
  with `nn.Flatten()`. The second removes a setting from the shared module and puts the
  shape change in the model that needs it. Needs deciding before the MLP is written.
- **Input resolution for the MLP.** The pipeline currently outputs 224x224, which is
  150,528 inputs — at 128 hidden units that is ~19M parameters in `W1` alone against 664
  training images. A smaller resolution for the MLP specifically (64x64 gives 12,288
  inputs, ~1.5M parameters) is worth considering. The CNN and ResNet stay at 224.
- **Normalization.** Currently commented out. Worth turning on later as a measurable
  before/after rather than assumed — see the Normalization section below.

## Known data issues

Dataset: `data/cat_v1`, **1,722 images** — the original Kaggle set (949 after
cleanup) plus the overlapping breeds from the Oxford-IIIT Pet dataset, merged
2026-08-06. Ragdoll is 395 of them (22.9%). Oxford has no domestic shorthair, so
that class remains Kaggle-only at 170.

Resolved:

- One failed download saved as HTML (`maine_coon/…dsc-8088.htm`) would have thrown on
  decode. Deleted.
- **Two images appeared in both `ragdoll/` and `siamese/` with identical bytes** —
  contradictory labels on the hardest confuser pair. The dataset was deduplicated by
  filename hash within each breed folder but not across them. Removed. A perceptual-hash
  sweep over all remaining images found no other duplicates or near-duplicates.
- Mixed formats (944 jpg/jpeg, 7 png, 1 webp) produce inconsistent channel counts —
  verified 4 channels on one PNG (alpha) and 1 on another (grayscale). `.convert('RGB')`
  in `__getitem__` normalizes all of them to 3; without it `DataLoader` cannot stack a
  batch.

Outstanding:

- Sizes are heterogeneous (roughly 1024x768 up to 4032x3024, both orientations). Only
  one apparent shared-source cluster at 1025x820. Handled by resize + center crop.
- ~575 KB average per image, 546 MB total. Every epoch decodes all of them from scratch.
  Pre-resizing into a cache directory would cut training time substantially; worth doing
  only once loading is measurably the bottleneck.

## Normalization

**Currently off.** `Normalize` is commented out of both pipelines. It buys a modest
convergence speedup and does not decide whether the MLP works, so it is not worth
blocking the baseline on.

When it goes in: compute mean/std per channel from the **training split only** —
statistics over the full dataset leak test-set information — via a run-once script whose
output is six hard-coded numbers. The constants are specific to the resize geometry and
need recomputing if that changes.

Because it is a one-line change, turning it on is a clean before/after measurement
against the un-normalized baseline rather than an assumption.

The pretrained ResNet stage is the exception and is not optional there: it requires
ImageNet's constants, `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`,
since the weights were fit under them.

## Data loading

Start with `num_workers=0` in `DataLoader` and raise it only if loading is measurably the
bottleneck. Dependencies are listed in `requirements.txt`.
