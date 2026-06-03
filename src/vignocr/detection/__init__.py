"""Detection stage — RF-DETR (medium) over the 15-class vignette schema.

Four entrypoints, all config-driven, seeded, and lazy about heavy ML libs
(torch / rfdetr / onnx / onnxruntime / pycocotools / albumentations are imported
*inside* functions, never here) so this package imports on a CPU-only core:

    train.run(cfg_path, run_dir, resume=None) -> Path   # best checkpoint
    eval.run(ckpt, root, split="valid")       -> dict   # mAP, per-class AP, loc-recall
    export.to_onnx(ckpt, out)                 -> Path   # + torch<->onnxruntime parity check
    infer.Detector(ckpt_or_onnx).detect(img)  -> list[Detection]

``infer`` (the ``Detection`` dataclass + ``Detector`` class) is import-safe on
CPU; constructing a ``Detector`` or running a train/eval/export is what triggers
the lazy ML import (raising a clear ``ImportError`` -> ``pip install -e .[ml]``).
"""

from __future__ import annotations

from vignocr.detection.infer import Detection, Detector

__all__ = ["Detection", "Detector"]
