"""
exp132_null_signature.py — does the random pseudo-suffix null also show
a consistent per-layer signature on schema directions?

If yes: the "layer-dependent signature" we attributed to morphology is
just generic geometric structure of any averaged-word-difference vector.

If no: the signature is morphology-specific.

Loads exp130's saved null matrices and compares real-suffix mean
projection per layer per schema vs:
  - null mean (should be ~0 if directions are random)
  - null std (variance of individual null directions)
  - null 5th/95th percentile (range of typical individual nulls)
"""

import numpy as np

d = np.load("/Users/macn/Documents/embeddingexp/exp130_results.npz")
layers = d["layers"]
schema_names = d["schema_names"]
suffix_names = d["suffix_names"]

print("\nComparison of real-suffix mean projection vs null distribution per schema per layer")
print("=" * 90)

for L in layers:
    real = d[f"matrix_L{L}"]   # [N_suffixes, N_schemas]
    null = d[f"null_L{L}"]     # [K_NULL, N_schemas]

    print(f"\nLayer {L}")
    print(f"  {'schema':<22}  {'real mean':>10}  {'null mean':>10}  "
          f"{'null std':>10}  {'null 5%':>10}  {'null 95%':>10}  {'z-score':>8}")
    for j, sn in enumerate(schema_names):
        real_mean = float(real[:, j].mean())
        null_mean = float(null[:, j].mean())
        null_std = float(null[:, j].std())
        null_p5 = float(np.percentile(null[:, j], 5))
        null_p95 = float(np.percentile(null[:, j], 95))
        if null_std > 1e-9:
            z = (real_mean - null_mean) / null_std
        else:
            z = float('nan')
        print(f"  {sn:<22}  {real_mean:>+10.3f}  {null_mean:>+10.3f}  "
              f"{null_std:>10.3f}  {null_p5:>+10.3f}  {null_p95:>+10.3f}  {z:>+8.2f}")


# Also: is the null DIRECTION-OF-LAYER signature consistent across trials?
# i.e., does null trial k1 at L4 correlate with null trial k1 at L20?
# This would show whether the layer-signature is in the null too.
print("\n" + "=" * 90)
print("Cross-layer consistency of NULL signatures (do random pseudo-suffixes preserve")
print("their schema-projection pattern across layers, like real suffixes do?)")
print("=" * 90)

# For each null trial k, get its full 8-schema projection vector at each layer
# Then compute cross-layer correlation across trials
print(f"\n  {'layer_pair':<14}  {'null cross-layer cos':>22}  {'real suffixes mean':>22}")
for i, L1 in enumerate(layers):
    for L2 in layers[i+1:]:
        null_L1 = d[f"null_L{L1}"]   # [K, 8]
        null_L2 = d[f"null_L{L2}"]   # [K, 8]
        # Per trial: cos between its schema vector at L1 and at L2
        # Normalize each row
        n1 = null_L1 / (np.linalg.norm(null_L1, axis=1, keepdims=True) + 1e-9)
        n2 = null_L2 / (np.linalg.norm(null_L2, axis=1, keepdims=True) + 1e-9)
        cross = (n1 * n2).sum(axis=1)  # per-trial cos
        # Same for real suffixes
        real_L1 = d[f"matrix_L{L1}"]
        real_L2 = d[f"matrix_L{L2}"]
        r1 = real_L1 / (np.linalg.norm(real_L1, axis=1, keepdims=True) + 1e-9)
        r2 = real_L2 / (np.linalg.norm(real_L2, axis=1, keepdims=True) + 1e-9)
        real_cross = (r1 * r2).sum(axis=1)
        print(f"  L{L1}↔L{L2}      "
              f"  mean={cross.mean():+.3f} std={cross.std():.3f}   "
              f"     mean={real_cross.mean():+.3f}")


# Also: do null SIGNATURES (averaged across all trials) reproduce the per-layer pattern we
# see in the real suffix matrix?
print("\n" + "=" * 90)
print("NULL MEAN SIGNATURE per schema per layer  vs  REAL MEAN SIGNATURE")
print("=" * 90)

for L in layers:
    real = d[f"matrix_L{L}"]
    null = d[f"null_L{L}"]
    real_sig = real.mean(axis=0)   # [N_schemas]
    null_sig = null.mean(axis=0)   # [N_schemas]
    # cos between real signature and null signature
    if np.linalg.norm(null_sig) > 1e-9 and np.linalg.norm(real_sig) > 1e-9:
        sig_cos = float(real_sig @ null_sig / (np.linalg.norm(real_sig) * np.linalg.norm(null_sig)))
    else:
        sig_cos = float('nan')
    print(f"\n  Layer {L}:")
    print(f"    real signature: {[f'{v:+.2f}' for v in real_sig]}")
    print(f"    null signature: {[f'{v:+.2f}' for v in null_sig]}")
    print(f"    cos(real_sig, null_sig) = {sig_cos:+.3f}")
