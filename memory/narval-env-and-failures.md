---
name: narval-env-and-failures
description: VignOCR Narval HPC environment facts + the recurring training-failure pattern
metadata:
  type: project
---

VignOCR trains RF-DETR on the Narval cluster (Digital Research Alliance). Allocation: account `def-khenni`, user `ydait`, project `6075472`. Real project path `/lustre06/project/6075472/ydait/pharmakon-ocr` (symlinked `~/projects/def-khenni/ydait/pharmakon-ocr`). venv lives on scratch `/lustre07/scratch/ydait/vignocr/venv`. Modules: StdEnv/2023, python/3.11, cuda/12.6. Compute nodes are air-gapped (no internet).

**The recurring failure pattern (as of 2026-06-05):** validation jobs pass, every training job (02 detection / 02a vignette) has failed — always a Python dependency/provisioning defect on the offline node, never GPU/CUDA/SLURM/architecture. Fixed reactively one ImportError at a time: RC-1 VIGNOCR_ACCOUNT not propagated → RC-2 EXIT-trap unbound `jlog` cascade-cancelled the afterok DAG → RC-3 `sympy` missing (torch._dynamo lazy dep) → RC-4 pretrained weights uncached → **RC-5 (live): `pytorch_lightning` missing because pyproject pins bare `rfdetr>=1.1` not `rfdetr[train,loggers]`** — rfdetr lazy-imports its Lightning training stack inside `model.train()`, so it crashes mid-training, and the setup verifier only checks the inference import chain (`rfdetr`), never `rfdetr.training`.

**Why:** offline wheelhouse + from-PyPI deps provisioned reactively. **How to apply:** declare `rfdetr[train,loggers]`, verify the *training* chain (`import rfdetr.training`), smoke-test one epoch in `salloc` before submitting the afterok DAG (one non-zero exit cancels all downstream stages). Also fix `train.py` kwargs: `lr_backbone`→`lr_encoder`, drop `clip_max_norm`/`num_workers` (not in rfdetr's `train()` API). See [[vignocr-dataset-gaps]].
