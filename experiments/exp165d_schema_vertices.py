"""exp165d_schema_vertices.py — DESCRIPTIVE extraction (not a new
hypothesis test). exp165c found folding is schema-general (all schemas
bowl, not just BALANCE) and speculated all bowls floor near c≈0 (the
natural state) -> the model rests at a JOINT stability equilibrium across
schema coordinates. This pulls the VERTEX (c*) + curvature of all 7
schema bowls + BALANCE from the SAME deterministic (SEED=165) all-layer
config exp165c used, to ask: do the optima cluster near zero, or scatter?

Faithful re-extraction (same seed/prompts/directions => identical curves
to 165c), computing the vertices 165c didn't print. Reuses 165b steering
machinery + 165c curve statistics wholesale.

Run: ./lakoff/bin/python3 exp165d_schema_vertices.py
"""
import os, gc
import numpy as np
from huggingface_hub import get_token
os.environ["HF_TOKEN"] = get_token() or ""

from markedness_norm_protocol import build_word_lists, SCHEMA_NAMES
from exp165b_all_layer_steer import _build_all_layer_dirs, _median_norms, _measure_all_layer
from exp165c_goldilocks_fold import DOSE_C, quad_stats, asymmetry, SEED, DEVICE


def _run_curve(model, dir_by_layer, m):
    return {c: _measure_all_layer(model, dir_by_layer, m, c)[0] for c in DOSE_C}


def main():
    import torch
    from transformer_lens import HookedTransformer
    print("exp165d — schema VERTEX map (descriptive; SEED=165, faithful to 165c)")
    all_words, est_words, test_words = build_word_lists()
    model = HookedTransformer.from_pretrained("gpt2-medium", device=DEVICE)
    model.eval()
    rng = np.random.default_rng(SEED)
    bal, schema, _ = _build_all_layer_dirs(model, all_words, est_words, test_words, rng)
    m = _median_norms(model)

    rows = []  # (name, k, cstar, argmin, asym)
    print("steering BALANCE + 7 schemas over 13-pt grid (all layers)...")
    by = _run_curve(model, bal, m)
    k, b, a, cstar, argmin, _ = quad_stats(by)
    rows.append(("BALANCE", k, cstar, argmin, asymmetry(by)))
    for sn, sd in schema.items():
        by = _run_curve(model, sd, m)
        k, b, a, cstar, argmin, _ = quad_stats(by)
        rows.append((sn, k, cstar, argmin, asymmetry(by)))

    print(f"\n{'='*72}")
    print(f"  {'direction':<16} {'curv k':>8} {'vertex c*':>10} {'argmin':>8} {'asym':>8}")
    print(f"  {'-'*16} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    cstars = []
    for name, k, cstar, argmin, asym in rows:
        cstars.append(cstar)
        print(f"  {name:<16} {k:>+8.3f} {cstar:>+10.3f} {argmin:>+8.2f} {asym:>+8.3f}")

    cstars = np.array(cstars)
    print(f"\n  vertex c* summary (all 8 meaningful dirs):")
    print(f"    mean {cstars.mean():+.3f}  sd {cstars.std(ddof=1):.3f}  "
          f"range [{cstars.min():+.3f}, {cstars.max():+.3f}]")
    near0 = int(np.sum(np.abs(cstars) <= 0.10))
    pos = int(np.sum(cstars > 0))
    print(f"    {near0}/8 floor within |c*|<=0.10 of the natural state;  "
          f"{pos}/8 floor on the positive side")
    print(f"\n  READ: if vertices cluster near 0 -> the natural state IS the joint")
    print(f"  stability optimum across schema coords (Niamh's 'balance is the ground'")
    print(f"  as resting-state equilibrium). If scattered -> each schema has its own")
    print(f"  distinct optimum, no shared balance point.")
    del model; gc.collect(); torch.mps.empty_cache()


if __name__ == "__main__":
    main()
