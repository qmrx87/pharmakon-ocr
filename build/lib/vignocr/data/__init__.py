"""Dataset layer: synthetic fixture generation, COCO loading, validation, stats.

Pure-CPU and config-driven. Images are drawn with Pillow only (no ML libs), so
the whole data layer imports and runs without the ``[ml]`` extra. The 15-class
schema is never hardcoded here — it flows from ``configs/classes.yaml`` via
``vignocr.common.get_classes()``.

Public API (import from ``vignocr.data``):
    synthetic.generate, synthetic.load_ground_truth, synthetic.GROUND_TRUTH_FILENAME
    coco.load_split, coco.crops_for_image, coco.CocoSplit, coco.Crop
    validate.check_integrity, validate.IntegrityReport
    stats.summarize
"""

from vignocr.data import coco, stats, synthetic, validate
from vignocr.data.coco import CocoSplit, Crop, crops_for_image, load_split
from vignocr.data.stats import summarize
from vignocr.data.synthetic import GROUND_TRUTH_FILENAME, generate, load_ground_truth
from vignocr.data.validate import IntegrityReport, check_integrity

__all__ = [
    "synthetic",
    "coco",
    "validate",
    "stats",
    "generate",
    "load_ground_truth",
    "GROUND_TRUTH_FILENAME",
    "load_split",
    "crops_for_image",
    "CocoSplit",
    "Crop",
    "check_integrity",
    "IntegrityReport",
    "summarize",
]
