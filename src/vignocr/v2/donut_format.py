"""Donut target-sequence format: flat field dict <-> tagged token sequence.

Donut decodes a token sequence like::

    <s_vignocr><s_num_lot>B1234</s_num_lot><s_date_exp>05/2027</s_date_exp></s>

This module owns the (pure-Python, CPU-testable) mapping between that sequence
and the flat ``{field: value}`` dict the rest of VignOCR speaks. Field tags are
registered as *special tokens* at fine-tune time (see :func:`special_tokens_for`)
so each tag is a single token id — shorter sequences, unambiguous parsing.

Only FLAT string fields are supported on purpose: the vignette schema is flat,
and a flat grammar keeps ``token2json`` trivially robust to partial/yanked
generations (every well-formed prefix parses).
"""

from __future__ import annotations

import re

__all__ = ["json2token", "token2json", "special_tokens_for"]


def _open_tag(field: str) -> str:
    return f"<s_{field}>"


def _close_tag(field: str) -> str:
    return f"</s_{field}>"


def json2token(values: dict[str, str], field_order: list[str] | None = None) -> str:
    """Serialize ``{field: value}`` to Donut's tagged sequence.

    Args:
        values: flat field -> value strings (non-string values are str()-ed;
            ``None``/empty values are skipped — absent field == not on vignette).
        field_order: emission order (stable decoding helps the decoder learn);
            fields not listed are appended alphabetically.

    Returns:
        The tagged body, WITHOUT the task-start/end tokens (the trainer adds
        ``<s_vignocr>`` ... ``</s>`` around it).
    """
    order = list(field_order or [])
    rest = sorted(k for k in values if k not in order)
    out: list[str] = []
    for field in [*order, *rest]:
        v = values.get(field)
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        out.append(f"{_open_tag(field)}{s}{_close_tag(field)}")
    return "".join(out)


_TAG_RE = re.compile(r"<s_([a-zA-Z0-9_]+)>(.*?)</s_\1>", re.DOTALL)


def token2json(sequence: str) -> dict[str, str]:
    """Parse a generated sequence back to ``{field: value}``.

    Tolerant by construction: ignores the task token, ``</s>``/pad debris, and
    any malformed fragment — every well-formed ``<s_f>v</s_f>`` pair found is
    returned, everything else is dropped (the abstention layer downstream treats
    a missing field as "not read", never as a guess).
    """
    return {m.group(1): m.group(2).strip() for m in _TAG_RE.finditer(sequence or "")}


def special_tokens_for(fields: list[str], task_start_token: str) -> list[str]:
    """The special tokens to register on the tokenizer for ``fields``."""
    toks = [task_start_token]
    for f in fields:
        toks.extend((_open_tag(f), _close_tag(f)))
    return toks
