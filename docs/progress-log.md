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

## 2026-07-30

**Done**
- Project scaffolding: `.gitignore`, `docs/`, `requirements.txt`, git repository
  initialized.
- Python 3.12.10 virtualenv created.
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
- Install dependencies and confirm the MPS backend is available.
- Port the MNIST MLP to PyTorch as a correctness check against a known result (98.08%
  from the previous from-scratch NumPy implementation) before introducing new data.
