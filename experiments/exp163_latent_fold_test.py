"""
exp163_latent_fold_test.py — the test exp162 couldn't do: is the schema
system a CURVED low-dimensional latent (one signed t + its fold t²/|t|)
even though the schema DIRECTIONS span ~6 linear dims?

(Niamh's question, 2026-06-11: "what would a collapse onto one linear and
one nonlinear (quadratic) dimension look like?" Answer: NOT like low
PR in direction-PCA — a curved 1D manifold has HIGH linear dimensionality
[horseshoe effect]. exp162 rejected the fold-as-explicit-direction
version only. This tests the latent-curve version.)

Method (per model, mid layer): project prompt tokens (exp160's 40 generic
prompts; unit residuals, aniso/freq-stripped dirs) onto all 8 schema
axes -> point cloud P [n_tokens x 8], column-centred. Fit and compare
variance explained (R^2):
  M1   rank-1 SVD (one linear latent)               — floor
  M2   rank-2 SVD (any two linear latents)          — ceiling for 2-factor
  M2q  CONSTRAINED: P ~ t a' + t^2 c'  (ONE latent + its square;
       alternating LS, per-token t via cubic root)  — Niamh's hypothesis
  M2a  CONSTRAINED: P ~ t a' + |t| c'  (fold as abs) — ReLU variant
  NULL column-shuffled rank-2 (20 shuffles)         — generic-structure null
Horseshoe diagnostic: R^2 of quadratic fit PC2 ~ PC1 (+ saved scatter).

READ: M2q ~= M2 >> M1, with M2 well above null  => latent fold REAL
      (two-primitive conjecture survives in curved form; exp162's verdict
      was about the wrong geometry).
      M2q ~= M1                                   => no fold; t^2 adds nothing.
      M2 itself low (~null)                       => schema projections not
      even 2-latent-explainable; high intrinsic dim; conjecture dead in
      BOTH forms.

EXPLORATORY EXPECTATIONS (2026-06-11, committed before running):
  E1 horseshoe R^2(PC2~PC1^2) < 0.3 (no strong visible fold)
  E2 M2 (rank-2 ceiling) explains 45-60% — above null (~30-40%?) but far
     from a 2-latent system
  E3 M2q closes <half the M1->M2 gap (fold adds little)
  i.e. I expect the curved version to fail too, but less confidently than
  I expected exp162's outcome — Scenario B is genuinely untested.
"""

import gc
import os

import numpy as np
import torch
from huggingface_hub import get_token

os.environ["HF_TOKEN"] = get_token() or ""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML
from markedness_norm_protocol import SCHEMA_NAMES, COMMON, RARE
from attn_entropy_lib import PROMPTS, schema_words

MODELS = {
    "pythia-410m":  dict(repo="pythia-410m",            layer=12),
    "gpt2-medium":  dict(repo="gpt2-medium",            layer=12),
    "Llama-3.2-1B": dict(repo="meta-llama/Llama-3.2-1B", layer=8),
}

rng = np.random.default_rng(17)


def r2(P, Phat):
    return 1.0 - np.sum((P - Phat) ** 2) / np.sum(P ** 2)


def fit_quad_latent(P, fold, iters=80, seed_t=None):
    """P [n x 8] centred. Fit P ~ t a' + fold(t) c', fold in {square, abs}."""
    n = P.shape[0]
    t = seed_t.copy() if seed_t is not None else np.linalg.svd(P)[0][:, 0]
    t = t / np.std(t)
    for _ in range(iters):
        F = np.vstack([t, t ** 2 if fold == "sq" else np.abs(t)]).T  # n x 2
        coef, *_ = np.linalg.lstsq(F, P, rcond=None)                  # 2 x 8
        a, c = coef[0], coef[1]
        if fold == "sq":
            # per-token minimize ||p - a t - c t^2||^2 ; d/dt = cubic
            aa, ac, cc = a @ a, a @ c, c @ c
            for i in range(n):
                pa, pc = P[i] @ a, P[i] @ c
                # 2cc t^3 + 3ac t^2 + (aa - 2pc... ) careful:
                # L'(t)/2 = cc*2 t^3 + 3 ac t^2 + (aa - 2 pc) t - pa  [derived]
                roots = np.roots([2 * cc, 3 * ac, (aa - 2 * pc), -pa])
                roots = roots[np.isreal(roots)].real
                if len(roots) == 0:
                    continue
                losses = [np.sum((P[i] - a * r - c * r ** 2) ** 2) for r in roots]
                t[i] = roots[int(np.argmin(losses))]
        else:
            # |t|: two linear branches
            for i in range(n):
                best = (np.inf, t[i])
                for s in (+1, -1):
                    v = a + s * c
                    ti = (P[i] @ v) / (v @ v)
                    if s * ti < 0:           # branch-inconsistent -> clamp to 0
                        ti = 0.0
                    loss = np.sum((P[i] - a * ti - c * abs(ti)) ** 2)
                    if loss < best[0]:
                        best = (loss, ti)
                t[i] = best[1]
        sd = np.std(t)
        if sd > 0:
            t = t / sd
    F = np.vstack([t, t ** 2 if fold == "sq" else np.abs(t)]).T
    coef, *_ = np.linalg.lstsq(F, P, rcond=None)
    return r2(P, F @ coef), t


fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (tag, cfg) in zip(axes, MODELS.items()):
    print(f"\n{'='*72}\n{tag}  (layer {cfg['layer']})\n{'='*72}")
    model = HookedTransformer.from_pretrained(cfg["repo"], device="mps")
    model.eval()
    L = cfg["layer"]
    rhook = f"blocks.{L}.hook_resid_post"

    vocab = schema_words()
    wres = {}
    for w in vocab:
        with torch.no_grad():
            _, c = model.run_with_cache(model.to_tokens(w), names_filter=[rhook])
        wres[w] = c[rhook][0, -1, :].float().cpu().numpy()
    aniso = np.stack(list(wres.values())).mean(0); aniso /= np.linalg.norm(aniso)
    fr = (np.mean([wres[w] for w in COMMON], 0) - np.mean([wres[w] for w in RARE], 0))
    fr /= np.linalg.norm(fr); fro = fr - (fr @ aniso) * aniso; fro /= np.linalg.norm(fro)

    def strip(d):
        d = d - (d @ aniso) * aniso; d = d - (d @ fro) * fro
        return d / np.linalg.norm(d)

    dirs = []
    for sn in SCHEMA_NAMES:
        pairs = LAKOFF_SCHEMAS_MML[sn]
        pos = sorted(set(p[0] for p in pairs)); neg = sorted(set(p[1] for p in pairs))
        dirs.append(strip(np.mean([wres[w] for w in pos], 0)
                          - np.mean([wres[w] for w in neg], 0)))
    D = np.stack(dirs)  # 8 x d

    # prompt-token projections
    rows = []
    for prompt in PROMPTS:
        toks = model.to_tokens(prompt)
        with torch.no_grad():
            _, c = model.run_with_cache(toks, names_filter=[rhook])
        resid = c[rhook][0].float().cpu().numpy()
        for q in range(1, resid.shape[0]):
            u = resid[q] / np.linalg.norm(resid[q])
            rows.append(D @ u)
    P = np.array(rows)
    P = P - P.mean(0)
    n = P.shape[0]

    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    lam = S ** 2
    r2_1 = float(lam[0] / lam.sum())
    r2_2 = float(lam[:2].sum() / lam.sum())
    pc1, pc2 = U[:, 0] * S[0], U[:, 1] * S[1]

    # horseshoe: PC2 ~ quadratic(PC1)
    A = np.vstack([pc1 ** 2, pc1, np.ones(n)]).T
    co, *_ = np.linalg.lstsq(A, pc2, rcond=None)
    hs = 1.0 - np.sum((pc2 - A @ co) ** 2) / np.sum((pc2 - pc2.mean()) ** 2)

    r2_q, t_q = fit_quad_latent(P, "sq", seed_t=U[:, 0].copy())
    r2_a, _ = fit_quad_latent(P, "abs", seed_t=U[:, 0].copy())

    nulls = []
    for _ in range(20):
        Psh = np.column_stack([rng.permutation(P[:, j]) for j in range(8)])
        Psh = Psh - Psh.mean(0)
        lsh = np.linalg.svd(Psh, compute_uv=False) ** 2
        nulls.append(float(lsh[:2].sum() / lsh.sum()))
    nulls = np.array(nulls)

    print(f"  n_tokens = {n}")
    print(f"  M1 rank-1 linear            R² = {100*r2_1:.0f}%")
    print(f"  M2 rank-2 linear (ceiling)  R² = {100*r2_2:.0f}%   "
          f"| null rank-2: {100*nulls.mean():.0f}% ± {100*nulls.std():.0f}%")
    print(f"  M2q one latent + t²         R² = {100*r2_q:.0f}%")
    print(f"  M2a one latent + |t|        R² = {100*r2_a:.0f}%")
    gap = r2_2 - r2_1
    closed = (max(r2_q, r2_a) - r2_1) / gap if gap > 1e-9 else float("nan")
    print(f"  fold closes {100*closed:.0f}% of the rank-1 -> rank-2 gap")
    print(f"  horseshoe R²(PC2 ~ quad(PC1)) = {hs:.2f}")

    ax.scatter(pc1, pc2, s=6, alpha=0.4)
    xs = np.linspace(pc1.min(), pc1.max(), 100)
    ax.plot(xs, co[0] * xs ** 2 + co[1] * xs + co[2], "r-", lw=1.5,
            label=f"quad fit R²={hs:.2f}")
    ax.set_title(f"{tag} L{L}"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=8)

    del model, wres
    gc.collect(); torch.mps.empty_cache()

plt.suptitle("exp163 — schema-projection cloud: horseshoe test (fold = curved latent?)")
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp163_horseshoe.png", dpi=120)
print("\nSaved exp163_horseshoe.png")
print("\nREAD: M2q≈M2>>M1 & M2>>null  => latent fold real (conjecture lives, curved form)")
print("      M2q≈M1                  => no fold")
print("      M2≈null                 => not even 2-latent; conjecture dead in both forms")
