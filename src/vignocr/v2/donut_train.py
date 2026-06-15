"""Fine-tune Donut (v2a) on the VLM dataset: vignette image -> tagged JSON.

Same operational contract as the v1 trainers (config-driven, seeded, resumable,
offline-safe, run-dir snapshot, ``run(cfg_path, run_dir, resume) -> Path``).
Heavy libs (torch / transformers) are imported lazily inside :func:`run`.

OFFLINE: the base model loads from the HF cache only (lib.sh exports
``HF_HOME`` + ``HF_HUB_OFFLINE=1``); pre-fetch once on the login node with
``scripts/fetch_pretrained_v2.sh``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vignocr.common import get_logger, load_config, repo_root, seed_everything
from vignocr.v2.donut_format import json2token, special_tokens_for, token2json
from vignocr.v2.normalize import normalize_value

log = get_logger(__name__)

_V2_HINT = "Donut fine-tuning needs the ml+v2 extras. Run: pip install -e .[ml,v2]"


def _dataset_dir(cfg: dict[str, Any]) -> Path:
    """Dataset root: env override > config; relative paths resolve to the repo."""
    raw = os.environ.get("VIGNOCR_VLM_DATASET_DIR") or cfg.get("dataset", {}).get(
        "dir", "ocr_dataset_vlm"
    )
    p = Path(raw)
    return p if p.is_absolute() else repo_root() / p


def _ckpt_dir(cfg: dict[str, Any], run_dir: Path) -> Path:
    """Checkpoints to SCRATCH on HPC (mirrors detection/_resolve_checkpoint_dir)."""
    cfg_dir = cfg.get("train", {}).get("checkpoint", {}).get("dir", "scratch/v2/donut")
    scratch = os.environ.get("VIGNOCR_SCRATCH")
    base = Path(scratch) / cfg_dir if scratch else (
        Path(cfg_dir) if Path(cfg_dir).is_absolute() else repo_root() / cfg_dir
    )
    out = base / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(cfg_path: str, run_dir: Path | str, resume: Path | str | None = None) -> Path:
    """Fine-tune Donut per ``configs/<cfg_path>.yaml``; return the best ckpt dir.

    ``resume``: a previously-saved checkpoint DIRECTORY (HF ``save_pretrained``
    layout) to continue from — model+processor are loaded from it instead of the
    base; the optimizer restarts (epoch-level resume, deliberately simple).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(cfg_path)
    tcfg = cfg.get("train", {})
    mcfg = cfg.get("model", {})
    fields: list[str] = list(cfg.get("fields", []))
    seed = int(tcfg.get("seed", 1337))
    seed_everything(seed)

    data_dir = _dataset_dir(cfg)
    train_rows = _read_jsonl(data_dir / "train" / "metadata.jsonl")
    val_rows = _read_jsonl(data_dir / "valid" / "metadata.jsonl")
    if not train_rows:
        raise FileNotFoundError(
            f"no training rows under {data_dir}/train — run scripts/build_vlm_dataset.py first"
        )
    log.info("donut.dataset", dir=str(data_dir), train=len(train_rows), val=len(val_rows))

    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            DonutProcessor,
            VisionEncoderDecoderModel,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(_V2_HINT) from exc

    base = str(resume) if resume else str(mcfg.get("base", "naver-clova-ix/donut-base"))
    task_token = str(mcfg.get("task_start_token", "<s_vignocr>"))
    image_size = list(mcfg.get("image_size", [640, 960]))  # H, W
    max_length = int(mcfg.get("max_length", 192))

    processor = DonutProcessor.from_pretrained(base)
    model = VisionEncoderDecoderModel.from_pretrained(base)

    # Shrink the canvas to vignette scale (Donut-base pretrains at 2560x1920 —
    # 10x more pixels than a sticker needs; this is the main latency lever).
    processor.image_processor.size = {"height": image_size[0], "width": image_size[1]}
    model.config.encoder.image_size = image_size

    # Register the field tags + task token as single-id special tokens.
    new_tokens = special_tokens_for(fields, task_token)
    n_added = processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    if n_added:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
    task_token_id = processor.tokenizer.convert_tokens_to_ids(task_token)
    model.config.decoder_start_token_id = task_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder.max_length = max_length

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = str(tcfg.get("precision", "bf16")) == "bf16" and device == "cuda"
    model.to(device)

    field_order = fields

    class _VlmDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]], split_dir: Path) -> None:
            self.rows = rows
            self.split_dir = split_dir

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int) -> dict[str, Any]:
            row = self.rows[i]
            img = Image.open(self.split_dir / row["file_name"]).convert("RGB")
            pixel = processor(img, return_tensors="pt").pixel_values[0]
            target = task_token + json2token(row["ground_truth"]["gt_parse"], field_order)
            ids = processor.tokenizer(
                target + processor.tokenizer.eos_token,
                max_length=max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            ).input_ids[0]
            labels = ids.clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel, "labels": labels,
                    "gt": row["ground_truth"]["gt_parse"]}

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "gt": [b["gt"] for b in batch],
        }

    bs = int(tcfg.get("batch_size", 4))
    accum = int(tcfg.get("grad_accum_steps", 4))
    epochs = int(tcfg.get("epochs", 30))
    workers = int(tcfg.get("num_workers", 4))
    train_dl = DataLoader(_VlmDataset(train_rows, data_dir / "train"), batch_size=bs,
                          shuffle=True, num_workers=workers, collate_fn=_collate)
    val_dl = DataLoader(_VlmDataset(val_rows, data_dir / "valid"), batch_size=bs,
                        shuffle=False, num_workers=workers, collate_fn=_collate)

    optim = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 3e-5)),
                              weight_decay=float(tcfg.get("weight_decay", 0.01)))
    total_steps = max(1, (len(train_dl) // max(1, accum)) * epochs)
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * float(tcfg.get("warmup_ratio", 0.05))), total_steps
    )

    ckpt_dir = _ckpt_dir(cfg, run_dir)
    (run_dir / "run_meta.json").write_text(json.dumps({
        "cfg_path": cfg_path, "base": base, "fields": fields, "image_size": image_size,
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "checkpoint_dir": str(ckpt_dir), "resume": str(resume) if resume else None,
    }, indent=2), encoding="utf-8")

    early = tcfg.get("early_stop", {})
    patience = int(early.get("patience", 6))
    min_delta = float(early.get("min_delta", 0.002))
    best_em, bad_epochs = -1.0, 0
    best_dir = ckpt_dir / "best"

    @torch.no_grad()
    def _validate(cap: int = 96) -> float:
        """Mean per-field exact-match (normalized) over <=cap val images."""
        model.eval()
        hits = total = 0
        seen = 0
        for batch in val_dl:
            pv = batch["pixel_values"].to(device)
            prompt = torch.full((pv.shape[0], 1), task_token_id, device=device)
            out = model.generate(
                pv, decoder_input_ids=prompt, max_length=max_length,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True, num_beams=1,
            )
            for seq, gt in zip(out, batch["gt"], strict=False):
                pred = token2json(processor.tokenizer.decode(seq, skip_special_tokens=False))
                for f, gt_v in gt.items():
                    total += 1
                    if normalize_value(f, pred.get(f, "")) == normalize_value(f, gt_v):
                        hits += 1
            seen += pv.shape[0]
            if seen >= cap:
                break
        model.train()
        return (hits / total) if total else 0.0

    log.info("donut.train.start", base=base, device=device, bf16=use_bf16,
             epochs=epochs, batch_size=bs, accum=accum, steps=total_steps)
    model.train()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else None
    step = 0
    for epoch in range(epochs):
        running = 0.0
        for i, batch in enumerate(train_dl):
            pv = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            if autocast:
                with autocast:
                    loss = model(pixel_values=pv, labels=labels).loss / accum
            else:
                loss = model(pixel_values=pv, labels=labels).loss / accum
            loss.backward()
            running += float(loss.item()) * accum
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
        em = _validate()
        log.info("donut.epoch", epoch=epoch, loss=round(running / max(1, len(train_dl)), 4),
                 val_field_em=round(em, 4), best=round(best_em, 4))
        if em > best_em + min_delta:
            best_em, bad_epochs = em, 0
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            (best_dir / "vignocr_donut.json").write_text(json.dumps({
                "fields": fields, "task_start_token": task_token,
                "image_size": image_size, "max_length": max_length,
                "val_field_em": em, "epoch": epoch,
            }, indent=2), encoding="utf-8")
        else:
            bad_epochs += 1
            if early.get("enabled", True) and bad_epochs >= patience:
                log.info("donut.early_stop", epoch=epoch, best_em=round(best_em, 4))
                break

    if not best_dir.exists():  # never validated above floor — save last anyway
        model.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)
    log.info("donut.train.done", best_em=round(best_em, 4), ckpt=str(best_dir))
    return best_dir
