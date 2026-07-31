# Project notes

Running notes on scope, decisions, and known issues. See `CLAUDE.md` for background
and working preferences.

## Goal and deadline

Full project complete by **August 31, 2026**.

The model is the point. Deployment is a follow-on task once the model is done, and is
not expected to be a large effort.

## Scope

**Phase 1 — cat / not-cat (binary).** Current focus. Needs a negative class; the
existing data is cats only.

**Phase 2 — breed classification.** The existing `data/cat_v1` is already breed-labeled
across 5 classes. More breeds may be added later, so nothing should assume a fixed
class count.

**Phase 3 — deployment.** CNN only. The MLP is a baseline for comparison and is not
shipped.

Staged model plan (from `CLAUDE.md`): PyTorch MNIST port → data pipeline → MLP on cat
images → CNN → fine-tuned pretrained ResNet. One change at a time, so any failure is
attributable.

## Decisions made

- **Shared data-prep module** used by both the MLP and the CNN. Shares the manifest,
  label mapping, and file loading. Resolution, channel count, and whether the output is
  flattened are parameterized, since the two models need different tensor shapes.
- **PIL, not OpenCV.** `cv2.imread` returns `None` on undecodable files instead of
  raising, and OpenCV loads BGR while all of torchvision assumes RGB. A channel swap
  would go unnoticed in the from-scratch models but would quietly degrade the
  pretrained ResNet, whose normalization is per-channel.
- **Subclass `torch.utils.data.Dataset`**, hand it to `DataLoader`. Writing
  `__len__`/`__getitem__` by hand is the part worth understanding; batching, shuffling,
  and worker parallelism come from `DataLoader`.
- **Persist the train/val/test split to a manifest** (`path,label,split`) rather than
  recomputing it per run. A fixed seed is not enough once files are added — and data
  *will* be added (negatives now, more breeds later). Without a stable split, "the CNN
  beat the MLP" is not a valid comparison.
- Dedupe before splitting; split before augmenting.

## Open decisions

- Source of the negative class for phase 1. The choice defines the task: easily
  separable negatives (landscapes, cars) would let the MLP baseline score well and
  destroy the value of the MLP-vs-CNN comparison. Dogs and other animals make the CNN
  earn its result. Target ~950 to match the cat count.
- Input resolution per model. 64x64 RGB is 12,288 inputs — at 128 hidden units that is
  ~1.57M parameters in `W1` alone against ~1,500 training images. Expected for a
  baseline, but should be chosen deliberately.
- Aspect-ratio handling: squash to square (distorts) vs. center crop (discards). Either
  is defensible; it must be applied identically everywhere.

## Known data issues

Dataset: `data/cat_v1`, 953 images, 548 MB, from the Kaggle link in
`data/dataset_link`. Counts: bengal 177, domestic_shorthair 170, maine_coon 191,
ragdoll 207, siamese 208.

- `maine_coon/2003-4288-2848-dsc-8088-2e700.dsc-8088.htm` is a failed download saved as
  HTML. It will throw when a loader tries to decode it.
- Mixed formats: 944 jpg/jpeg, 7 png, 1 webp. PNGs may carry an alpha channel — convert
  to RGB explicitly or the channel count will not match the first layer.
- Sizes are heterogeneous (roughly 1024x768 up to 4032x3024, both orientations). Only
  one apparent shared-source cluster at 1025x820.
- Because the cats are messy multi-source scrapes, negatives pulled from a single clean
  source would differ systematically in resolution and compression. A model can separate
  on that alone without looking at an animal. Source negatives as messily as the cats.
- ~575 KB average per image. Pre-resizing to a cache directory avoids decoding
  multi-megapixel JPEGs every epoch.

## Normalization

Compute mean/std from the **training split only** — statistics over the full dataset
leak test-set information. The pretrained ResNet stage is the exception: it requires
ImageNet's constants, since the weights were fit under them.

## Environment

Apple Silicon, macOS 15.3.1, `device="mps"`. Python 3.12.10.

As of these notes, no virtualenv exists in this project and `torch` is not installed.

The project lives in iCloud Drive. iCloud can evict file contents and re-download on
access, which can surface as slow or intermittently failing reads once a `DataLoader`
with multiple workers is reading many files. Keep the venv and any resized cache out of
sync where possible.
