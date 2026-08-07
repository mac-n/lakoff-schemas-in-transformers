"""exp165e_random_vertices.py — nails the exp165d caveat. exp165d found
all 8 MEANINGFUL directions floor near c=0. But floor-at-zero may be
GENERIC (any perturbation -> OOD -> less stable). This extracts the 16
RANDOM directions' vertices from the SAME (SEED=165) config and asks:
do random bowls ALSO floor at zero (floor-at-zero is generic) or SCATTER
(floor-at-zero is meaning-specific -> the 'joint balance point' is real)?

Run: ./lakoff/bin/python3 exp165e_random_vertices.py
"""
import os, gc
import numpy as np
from huggingface_hub import get_token
os.environ["HF_TOKEN"] = get_token() or ""

from markedness_norm_protocol import build_word_lists
from exp165b_all_layer_steer import _build_all_layer_dirs, _median_norms, _measure_all_layer
from exp165c_goldilocks_fold import DOSE_C, quad_stats, SEED, DEVICE

# exp165d's 8 meaningful vertices, for direct comparison
MEANINGFUL_CSTARS = [0.053, 0.021, 0.067, -0.050, -0.027, -0.035, -0.019, -0.073]


def _run_curve(model, dir_by_layer, m):
    return {c: _measure_all_layer(model, dir_by_layer, m, c)[0] for c in DOSE_C}


def main():
    import torch
    from transformer_lens import HookedTransformer
    print("exp165e — RANDOM vertices (caveat test; SEED=165, faithful to 165c)")
    all_words, est_words, test_words = build_word_lists()
    model = HookedTransformer.from_pretrained("gpt2-medium", device=DEVICE)
    model.eval()
    rng = np.random.default_rng(SEED)
    _, _, randoms = _build_all_layer_dirs(model, all_words, est_words, test_words, rng)
    m = _median_norms(model)

    print(f"steering {len(randoms)} random all-layer directions over 13-pt grid...")
    rng_ks, rng_cstars = [], []
    for i, rd in enumerate(randoms):
        by = _run_curve(model, rd, m)
        k, b, a, cstar, argmin, _ = quad_stats(by)
        rng_ks.append(k); rng_cstars.append(cstar)
        print(f"  random[{i:>2}]  k={k:>+7.3f}  c*={cstar:>+7.3f}")

    rc = np.array(rng_cstars); mc = np.array(MEANINGFUL_CSTARS)
    print(f"\n{'='*72}")
    print(f"  RANDOM vertices:     mean {rc.mean():+.3f}  sd {rc.std(ddof=1):.3f}  "
          f"range [{rc.min():+.3f},{rc.max():+.3f}]  within|0.10|: {int(np.sum(np.abs(rc)<=0.10))}/16")
    print(f"  MEANINGFUL vertices: mean {mc.mean():+.3f}  sd {mc.std(ddof=1):.3f}  "
          f"range [{mc.min():+.3f},{mc.max():+.3f}]  within|0.10|: {int(np.sum(np.abs(mc)<=0.10))}/8")
    print(f"  RANDOM curvatures:   mean {np.mean(rng_ks):+.3f}  sd {np.std(rng_ks,ddof=1):.3f}")
    print(f"\n  READ:")
    print(f"  - random vertices ALSO cluster at 0 -> floor-at-zero is GENERIC (OOD);")
    print(f"    keep only the bowl-DEPTH finding (meaning 3x more sensitive).")
    print(f"  - random vertices SCATTER while meaning floors at 0 -> the joint")
    print(f"    balance point IS meaning-specific (Niamh's 'ground' survives strong).")
    del model; gc.collect(); torch.mps.empty_cache()


if __name__ == "__main__":
    main()
