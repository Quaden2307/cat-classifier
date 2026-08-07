# Progress log

A day-by-day record of what was built, what was decided, and what went wrong. Newest
entries at the top.

Target completion: **August 31, 2026**.

## Conventions

One entry per working day. Each entry covers:

- **Done** — what actually got built or changed.
- **Decided** — choices made, with the reason. The reason is the part worth keeping.
- **Learned / hit** — bugs, surprises, dead ends. Worth recording precisely; a failure
  you can explain is more useful than one you quietly fixed.
- **Next** — what the following session picks up.

Results get numbers. "Improved accuracy" is not a log entry; "val accuracy 0.71 → 0.79
after adding random horizontal flips" is.

---

## 2026-08-06

**Done (data batch + retrain)**
- Added the overlapping breeds from the Oxford-IIIT Pet dataset (bengal, maine coon,
  ragdoll, siamese; it has no domestic shorthair): 949 → **1,722 images**, ragdoll
  395 (22.9%). Merge sweep over all 1,722: every file decodes, no cross-source or
  cross-breed duplicates; removed 2 double-copied files and one frame from each of
  2 same-cat burst pairs (4 deletions total).
- Rebuilt the split: train 1,205 / val 258 / test 259, ragdoll 22.9 / 22.9 / 23.2%.
  Test positives 31 → 60, so the noise floor tightens to roughly ±0.01 F1.
  Recomputed the class weight: 3.61 → **3.37** (929/276).
- Retrained the ship config (10 epochs, layer4 + head): val F1 settled 0.83–0.84.
  Test: **P 0.800 / R 0.867 / F1 0.832**, confusion matrix [[186, 13], [8, 52]] —
  52 of 60 ragdolls caught, 13 false alarms in 199 negatives.
- Re-exported to ONNX; parity 259/259 vs PyTorch. Site stats updated.
- Doubling the data moved test F1 0.806 → 0.832 (different test sets, but the new
  one is twice as reliable). The data lever works; next stop toward 0.90 is more
  data again and/or unfreezing layer3.
- MLP baseline not yet rerun on the new split — its 0.52 refers to the old split
  until then.

**Done**
- Experiment: doubled training to 20 epochs, same seed. Epochs 1–10 replayed the
  documented run exactly (the seed pins init, shuffle order, and flips). Epochs 11–20:
  val loss flat between 0.186 and 0.208 with no sustained trend, val F1 bouncing
  0.746–0.812 inside the established noise band, train loss 0.056 → 0.012.
  Test: **identical metrics and confusion matrix** to the 10-epoch model —
  P/R/F1 0.806, [[106, 6], [6, 25]]. Ten extra epochs moved nothing.
- Added MIT license.

**Decided**
- **Ship config stays at 10 epochs.** The model converges by ~epoch 10; further
  training only memorizes the 664 training images harder without changing a single
  test decision. The bottleneck is data, not training time — the planned data batch
  (more ragdolls + phase-2 breeds) is the next lever that can actually move F1.

---

## 2026-08-05

**Done**
- Stage 5 pulled forward: `resnet/model.py` — pretrained resnet18, every layer frozen,
  head swapped to `Linear(512, 2)`, so only 1,026 of 11.2M parameters train. ImageNet
  normalization on (mandatory here — the pretrained weights were fit under it). Same
  loop, loss, and metrics as the MLP, so the architecture is the only change.
- 5 epochs: val F1 0.658 → 0.732, still rising at the end. Test (one look):
  **P 0.718 / R 0.903 / F1 0.800**, confusion matrix [[101, 11], [3, 28]] — caught 28 of
  31 ragdolls, 11 false alarms in 112 negatives.
- Fine-tune round: unfroze `layer4` (8.4M params at lr 1e-5, head at 1e-3 — param
  groups), 10 epochs. Val F1 0.711 → **0.794**. Test:
  **P 0.788 / R 0.839 / F1 0.812**, confusion matrix [[105, 7], [5, 26]].
- Augmentation experiment: `RandomHorizontalFlip(p=0.5)`, train pipeline only. Val F1
  0.787, test **P 0.806 / R 0.806 / F1 0.806**, matrix [[106, 6], [6, 25]] — one false
  alarm traded for one miss vs. the no-flip run. **No measurable effect**; kept anyway
  (costs nothing, marginally more robust). This run's weights are what
  `resnet/resnet18_cats.pt` now holds — the ship model: **test F1 0.806**.

**Done (frontend)**
- Exported the ship model to ONNX (`resnet/export_onnx.py` → `frontend/model.onnx`,
  43 MB single file). Parity-checked against PyTorch on all 143 test images:
  predictions agree 143/143, max logit difference 4.9e-06.
- Built the frontend (`frontend/`): static HTML/CSS/JS, no framework. Upload or
  drag-drop a photo → preprocessing re-implemented in JS (resize shorter side to 256,
  center crop 224, ImageNet normalize) → onnxruntime-web runs the ONNX model in the
  visitor's browser → verdict + softmax confidence. Cozy ragdoll palette
  (cream/powder-blue/camel), keycap-style button.
- Verified the JS preprocessing is safe: simulated the browser's differences from PIL
  (bilinear resize, 8-bit quantization, floor-offset crop) in Python — predictions
  unchanged on 143/143 test images. Canvas-vs-PIL drift does not flip any decision.

**Decided**
- **Inference runs in the browser, not on a server.** The model is exported to ONNX
  and executed by onnxruntime-web; hosting is a static site on Vercel. Reason: free
  Python hosting sleeps when idle, and a recruiter clicking the link gets a cold start
  or a dead page — a static site is always up instantly. Cost: preprocessing had to be
  re-implemented in JS and verified against the Python pipeline (done, above).
- **Reordered the plan: fine-tuned ResNet before the from-scratch CNN.** A deployable
  model plus frontend must exist well before the deadline; the ResNet is the model that
  ships, and the published benchmark (~82% from scratch vs ~97% fine-tuned at this data
  size) says the CNN cannot close the gap anyway. The CNN stays in the project as the
  comparison piece, built after deployment works end to end.

**Learned / hit**
- MLP 0.48 → ResNet 0.80 test F1 on identical data, loop, and loss. The gap is
  features, not tuning — the strongest argument in the project for transfer learning.

**Learned / hit (fine-tune round)**
- Train loss ~0.05 vs val ~0.20 by epoch 10 — the model memorizes the 664 training
  images. A single horizontal flip barely slowed that (train loss 0.045 → 0.056) and
  moved F1 within noise.
- **The noise floor on this dataset is ±0.02 F1.** With 31 ragdolls in val or test,
  one cat changing sides moves recall by 1/31 ≈ 0.032. Differences smaller than that
  are not evidence of anything.

**Done (deployment)**
- Live at **https://cat-classifier-five.vercel.app/** — static site on Vercel,
  root directory `frontend/`, no build step. README written for the public repo.

**Next**
- From-scratch CNN comparison (the original stage 4), then the write-up.
- One deliberate data batch later: more ragdolls + phase-2 breeds, dedupe against
  the existing 949, rebuild the split, recompute the class weight, retrain
  everything for the final comparison table. Better ResNet weights swap into the
  live site by re-running the export — the frontend never changes.

---

## 2026-08-03

**Done**
- Stage 3 complete: MLP baseline trained on the cat images (`nn/model.py`) —
  `Flatten → Linear(150528, 128) → ReLU → Linear(128, 2)`, Adam, weighted
  cross-entropy, 10 epochs. Best val F1 **0.517** (epoch 6), settling around 0.48.

**Decided**
- **Class-weighted loss** `[1.0, 3.61]` (= 520/144, the training-split imbalance) —
  required to get the MLP off the majority-class answer at all. Details in
  project-notes; applies to every model trained on this split.

**Learned / hit**
- First run at lr 1e-3: epoch-1 train loss 12.1 — Adam's first steps are ~40% of the
  first layer's init scale at this width, blowing the weights apart before recovery.
  Dropped to 1e-4.
- **Unweighted loss collapsed to the majority class**: F1 0.000 for 10 straight epochs
  while val loss fell to 0.45 — *below* the 0.52 of always answering "not ragdoll."
  Loss alone cannot detect this failure; only the P/R/F1 print exposed it.
- Weighted run never stabilized: recall oscillated 0.13 ↔ 0.97 across epochs — the
  model thrashing its threshold rather than learning features. Expected from flattened
  pixels; this is the baseline the CNN and ResNet must beat.

**Next**
- Pretrained ResNet (stage reordered — see 2026-08-05).

---

## 2026-08-01

**Done**
- Cleaned the dataset: removed the HTML file and two duplicate images. 953 → **949**,
  ragdoll 206 (21.7%).
- Built `data_prep/prep.py` — walks the breed folders, assigns labels, splits 70/15/15
  stratified on the ragdoll label with a fixed seed, writes `data/splits.csv`
  (`id,image_path,breed,ragdoll,split`). Result: train 664 / val 142 / test 143, ragdoll
  at 21.7% / 21.8% / 21.7%.
- Built `data_prep/dataset.py` — `CatDataset` subclassing `torch.utils.data.Dataset`,
  plus separate train and eval transform pipelines (`Resize(256)` → `CenterCrop(224)` →
  `ToImage` → `ToDtype(float32, scale=True)`).
- Verified end to end: batch of 8 comes out `(8, 3, 224, 224)` float32 with labels
  `(8,)` int64, pixel range 0.008–0.98, and **0 label/folder mismatches across all 664
  training rows**.

**Decided**
- **Center crop over squash** — `Resize(256)` then `CenterCrop(224)`. 224 is what the
  pretrained ResNet expects at stage 5, so the geometry carries through unchanged.
- **Two files, not one.** `prep.py` is run once and writes the split; `dataset.py` is
  imported. Importing a module executes it top to bottom, so combining them would
  regenerate the split on every training run.
- **Normalization deferred.** It gives a modest convergence speedup and does not decide
  whether the MLP works. Left commented out so it can be turned on later as a measured
  before/after instead of an assumption. Required (non-optional) only at the ResNet stage.
- **Breed column carried in the split file** even though phase 1 ignores it, so phase 2
  does not require regenerating and re-randomizing the split.

**Learned / hit**
- **The Kaggle set was not duplicate-free.** Two images existed in both `ragdoll/` and
  `siamese/` with identical bytes — contradictory labels, on exactly the two breeds
  hardest to tell apart. The set had been deduplicated by filename hash *within* each
  breed folder but not across them. Left in, one copy would have landed in train and one
  in test, on the class being scored. A perceptual-hash sweep found no others.
- **PNGs do not all decode to 3 channels.** Measured 4 on one (alpha) and 1 on another
  (grayscale). Without `.convert('RGB')`, `DataLoader` cannot stack a batch.
- **Opening the csv inside the write loop with `mode='w'` truncates on every iteration**,
  leaving a 2-line file — header plus the last image. With `mode='a'` it instead appends
  a fresh 949 rows per run. Both fixed by opening once, outside the loop.
- **`train_test_split` returns the same dict objects, not copies.** Setting `r['split']`
  on the returned lists therefore mutates the rows in the original list, which is what
  lets the whole thing be written out in one pass, still in folder order. Only true for
  lists — the same call on a numpy array returns copies, because numpy fancy indexing
  copies.

**Next**
- Decide where the flatten lives (dataset vs. `nn.Flatten()` as the MLP's first layer)
  and what input resolution the MLP gets — 224x224 is 150,528 inputs against 664
  training images.
- Stage 3: MLP on cat images. Expect poor results; it is the baseline to beat.

---

## 2026-07-30

**Done**
- Project scaffolding: `.gitignore`, `docs/`, `requirements.txt`, git repository
  initialized.
- Development environment set up; dependencies pinned in `requirements.txt`.
- Dataset in place: 953 images across 5 breeds (bengal 177, domestic_shorthair 170,
  maine_coon 191, ragdoll 207, siamese 208), 548 MB, sourced from the Kaggle link in
  `data/dataset_link`.
- Audited the dataset. Found one failed download saved as HTML
  (`maine_coon/2003-4288-2848-dsc-8088-2e700.dsc-8088.htm`) that will throw on decode,
  mixed formats (944 jpg/jpeg, 7 png, 1 webp), and highly heterogeneous image sizes
  (~1024x768 up to 4032x3024, both orientations).

**Decided**
- **First task is binary: ragdoll vs. not-ragdoll.** Both classes come from the existing
  scrape, so there is no need to source a negative set — and no risk of the model
  separating classes on resolution or compression artifacts instead of on the animal.
  The negatives also contain the two hardest confusers: siamese (also colorpoint) and
  maine coon (also long-haired and large).
- **Target set at F1 ≥ 0.90 on the ragdoll class**, not raw accuracy. Ragdoll is 21.7%
  of the data, so always predicting "not ragdoll" already scores 78.3%, and a model at
  90% accuracy can still miss nearly half the ragdolls. Precision, recall, and a
  confusion matrix get reported alongside.
- **PIL over OpenCV** for loading — `cv2.imread` returns `None` rather than raising on
  undecodable files, and OpenCV's BGR channel order would silently misalign the
  per-channel normalization used by a pretrained model later.
- **Persist the train/val/test split to a manifest** rather than recomputing it per run.
  A fixed seed is not sufficient once files are added, and comparing models across
  different test sets would be meaningless.
- **Deployment target is likely the fine-tuned pretrained model, not the from-scratch
  CNN.** At ~2,000 images, the comparable published benchmark reaches ~82% from scratch
  and ~97% fine-tuned. The from-scratch CNN stays in the project as the comparison that
  demonstrates why.

**Next**
- Install dependencies and confirm GPU acceleration is available.
- Port the MNIST MLP to PyTorch as a correctness check against a known result (98.08%
  from the previous from-scratch NumPy implementation) before introducing new data.
