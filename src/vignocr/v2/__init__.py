"""VignOCR v2 — challenger variants trained/compared against the v1 cascade.

v2a (``donut_*``)  : OCR-free VLM — vignette image -> JSON, one fine-tuned model.
v2b (``fullpage``) : pretrained docTR detection + PARSeq recognition over the
                     whole vignette + a deterministic layout parser.

Both variants share v1's deterministic core (checksum, nomenclature correction,
abstention) — they only replace the detect+read stages. Heavy ML libraries are
imported lazily inside the modules that use them; importing this package is
CPU-safe.
"""
