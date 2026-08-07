"""
exp167_schema_dnorm_table.py — the all-schemas x d_norm specificity
table (handoff lead #5), prompted by Niamh's question: is there an
UP-DOWN/magnitude grounding (MORE IS UP -> residual norm)?

DESCRIPTIVE / EXPLORATORY — no verdict gate. Expectations stated in
chat + notebook before running: Pythia BALANCE top (~0.7), UP-DOWN and
LIGHT-DARK moderate (0.3-0.5); GPT-2/Llama all near zero.

Measures, per model per layer (word residuals only, no prompts):
  (a) cos(schema_dir, d_norm_ho)  — held-out d_norm, exp154c protocol
  (b) out-of-sample scalar check: corr over TEST words of
      (unit-residual @ schema_dir) vs z-scored residual norm —
      the carrier correlation restricted to each schema's direction.
Both reported for all 8 schemas; layers = exp161's five per model.
"""

import gc
import os

import numpy as np
import torch
from huggingface_hub import get_token

os.environ["HF_TOKEN"] = get_token() or ""

from markedness_norm_protocol import (SCHEMA_NAMES, corrf,
                                      build_word_lists, collect_residuals)
from attn_entropy_lib import MODELS
from exp161_balance_entropy_prereg import build_layer_dirs


def main():
    from transformer_lens import HookedTransformer
    all_words, est_words, test_words = build_word_lists()
    print("exp167 — all-schemas x d_norm specificity table (descriptive)")
    print(f"vocab {len(all_words)} (est {len(est_words)} / "
          f"test {len(test_words)})")

    for tag, cfg in MODELS.items():
        print(f"\n{'='*72}\n{tag}\n{'='*72}")
        model = HookedTransformer.from_pretrained(cfg["repo"], device="mps")
        model.eval()
        LAYERS = cfg["layers"]
        residuals = collect_residuals(model, LAYERS, all_words, log_every=0)

        for L in LAYERS:
            d = build_layer_dirs(residuals, L, all_words, est_words,
                                 test_words)
            norms = {w: float(np.linalg.norm(residuals[w][L]))
                     for w in test_words}
            zn = np.array(list(norms.values()))
            zn = (zn - zn.mean()) / zn.std()
            units = {w: residuals[w][L] / np.linalg.norm(residuals[w][L])
                     for w in test_words}
            rows = []
            for sn in SCHEMA_NAMES:
                cos_dn = float(d["schema"][sn] @ d["d_norm_ho"])
                proj = np.array([float(units[w] @ d["schema"][sn])
                                 for w in test_words])
                scal = corrf(proj, zn)
                rows.append((sn, cos_dn, scal))
            rows.sort(key=lambda r: -abs(r[1]))
            line = "  ".join(f"{sn.split('-')[0][:5]} {c:+.2f}/{s:+.2f}"
                             for sn, c, s in rows)
            print(f"  L{L:>2} (carrier {d['carrier_out']:+.2f})  "
                  f"cos(dir,d_norm)/scalar-test: {line}")

        del model, residuals
        gc.collect(); torch.mps.empty_cache()


if __name__ == "__main__":
    main()
