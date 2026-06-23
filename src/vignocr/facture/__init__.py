"""FactureOCR — supplier-invoice reading for the pharmacy STOCK-INTAKE workflow.

A pharmacy receives a supplier invoice (``facture``) listing many drugs at once
with, per line: designation, DCI, quantity, lot, expiry, unit price (P.U.HT),
PPA, SHP, and the line total. Reading the invoice in one shot pre-fills a whole
stock-intake (réception) instead of scanning each box's vignette — the vignette
path (``vignocr.v2.claude_extract``) stays for per-unit / sales validation.

Two pieces, mirroring the vignette design:
  * :class:`~vignocr.facture.claude_extract.FactureExtractor` — Claude vision
    reads the invoice image into a structured ``{header, lines, totals}`` dict
    (handles the per-supplier layout variation a rigid template parser can't).
  * :mod:`vignocr.facture.verify` — a deterministic ARITHMETIC verifier
    (qty × unit_price ≈ line_total; Σ lines ≈ net) that flags OCR digit errors
    on money the way the vignette checksum guards prices.

Heavy/network deps (the anthropic SDK) are imported lazily; importing this
package is CPU/offline-safe.
"""
