"""
exp162_schema_dimensionality.py — does the 8-schema space collapse to a
small number of primitives, and do they look like valence + arousal?
(Niamh's conjecture, 2026-06-11: "does it all boil down to one nonlinear
and one linear primitive?" — refined: how many primitives, what are they;
embodiment of each is a LATER question.)

THE control that decides whether the two-primitive story is real or an
artifact of our bipolar vocab construction: PCA/SVD of the 8 stripped
schema directions, vs a NULL of random-partition pseudo-schemas built
over the SAME vocabulary with the SAME group sizes. If random bipolar
contrasts collapse just as hard, "two primitives" is manufactured by how
we built the schemas, not a fact about the model.

Metrics (per model, per layer):
  - participation ratio PR = (Σλ)²/Σλ²  of the schema-direction Gram
    spectrum (uncentered SVD; shared component is itself a candidate
    primitive). PR ≈ effective number of dimensions. PR≈2 ⇒ two
    primitives; PR≈4 ⇒ four.
  - top-2 cumulative variance fraction.
  - NULL: 200 random-partition pseudo-schema sets → null PR + top-2
    distribution. Real must be MORE concentrated (lower PR / higher
    top-2) than null to claim genuine low-dim structure.
  - Identification: build valence (good/bad) + arousal (calm/agitated,
    valence-crossed) directions. Report cos(valence,arousal);
    fraction of each captured in the top-2 schema subspace; and each
    schema's cos with valence and (valence-orthogonalised) arousal.

PRE-REGISTRATION (2026-06-11, before running; this Claude):
  Committed prediction (NOT reflexively deflationary — trying to predict
  what I actually believe): genuine low-dim structure well above null,
  but NOT a clean 2 — I expect PR ≈ 2.5-3.5 (one dominant shared axis +
  arousal + some residual), so "one linear + one nonlinear" is ~one
  primitive too few. Identification:
    P1 PR(real) < PR(null_5th_pctile) at decision layers (real structure).
    P2 PR(real) in [2.5, 3.5] (low-dim but not 2).
    P3 valence is the dominant shared axis; BALANCE has the highest
       cos with the arousal axis among the 8 schemas (signed balance:
       balanced≈calm, unbalanced≈agitated).
  Decision rule (verdict162, synthetic-tested):
    V_TWO  PR(real) <= 2.3 AND below null AND valence+arousal capture
           >= 70% of the top-2 subspace: TWO-PRIMITIVE conjecture
           SUPPORTED. Niamh right; proceed to folding + coupling tests.
    V_NULL PR(real) within null band: no special structure; the schema
           intercorrelations were generic. Conjecture rejected.
    V_LOWD PR(real) below null but > 2.3: low-dimensional but >2
           primitives; report effective dim; refine (my P2 lands here).
"""

import gc
import os

import numpy as np
import torch
from huggingface_hub import get_token

os.environ["HF_TOKEN"] = get_token() or ""

from transformer_lens import HookedTransformer

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML
from markedness_norm_protocol import SCHEMA_NAMES, COMMON, RARE

MODELS = {
    "pythia-410m":  dict(repo="pythia-410m",             layers=[4, 12, 20]),
    "gpt2-medium":  dict(repo="gpt2-medium",             layers=[4, 12, 20]),
    "Llama-3.2-1B": dict(repo="meta-llama/Llama-3.2-1B",  layers=[3, 8, 13]),
}

VALENCE_PAIRS = [("good","bad"),("pleasant","unpleasant"),("wonderful","awful"),
    ("lovely","nasty"),("delightful","dreadful"),("positive","negative"),
    ("nice","horrible"),("fine","lousy")]
# arousal: high vs low ACTIVATION, valence deliberately crossed within each pole
AROUSAL_PAIRS = [("excited","calm"),("frantic","relaxed"),("agitated","serene"),
    ("energetic","sluggish"),("intense","mild"),("alert","drowsy"),
    ("restless","settled"),("tense","placid")]


def participation_ratio(sv):
    lam = sv ** 2
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def top2_frac(sv):
    lam = sv ** 2
    return float(lam[:2].sum() / lam.sum())


# ---------------- verdict (pure, selftest-able) ----------------

def verdict162(pr_real, pr_null_5th, va_capture):
    if pr_real <= 2.3 and pr_real < pr_null_5th and va_capture >= 0.70:
        return "V_TWO"
    if pr_real >= pr_null_5th:
        return "V_NULL"
    return "V_LOWD"


def selftest_verdict162():
    assert verdict162(2.1, 5.0, 0.80) == "V_TWO"
    assert verdict162(6.0, 5.0, 0.80) == "V_NULL"
    assert verdict162(3.0, 5.0, 0.80) == "V_LOWD"
    assert verdict162(2.1, 5.0, 0.50) == "V_LOWD"   # low PR but axes not valence/arousal
    print("selftest_verdict162: all branches fire correctly.")


print("Running verdict-logic selftest first (synthetic-test convention)...")
selftest_verdict162()

# words to collect
schema_words = set(COMMON + RARE)
for sn in SCHEMA_NAMES:
    for p, n in LAKOFF_SCHEMAS_MML[sn]:
        schema_words.add(p); schema_words.add(n)
aff_words = set()
for a, b in VALENCE_PAIRS + AROUSAL_PAIRS:
    aff_words.add(a); aff_words.add(b)
all_w = sorted(schema_words | aff_words)
schema_pool = sorted(schema_words - set(COMMON) - set(RARE))

rng = np.random.default_rng(13)

for tag, cfg in MODELS.items():
    print(f"\n{'='*72}\n{tag}\n{'='*72}")
    model = HookedTransformer.from_pretrained(cfg["repo"], device="mps")
    model.eval()
    LAYERS = cfg["layers"]
    rhooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    res = {}
    for w in all_w:
        with torch.no_grad():
            _, c = model.run_with_cache(model.to_tokens(w), names_filter=rhooks)
        res[w] = {L: c[f"blocks.{L}.hook_resid_post"][0, -1, :].float().cpu().numpy()
                  for L in LAYERS}

    for L in LAYERS:
        arr = np.stack([res[w][L] for w in all_w]); aniso = arr.mean(0); aniso /= np.linalg.norm(aniso)
        fr = (np.mean([res[w][L] for w in COMMON], 0) - np.mean([res[w][L] for w in RARE], 0))
        fr /= np.linalg.norm(fr); fro = fr - (fr @ aniso) * aniso; fro /= np.linalg.norm(fro)

        def strip(d):
            d = d - (d @ aniso) * aniso; d = d - (d @ fro) * fro
            return d / np.linalg.norm(d)

        def diff_dir(pairs):
            pos = sorted(set(p[0] for p in pairs)); neg = sorted(set(p[1] for p in pairs))
            return strip(np.mean([res[w][L] for w in pos], 0) - np.mean([res[w][L] for w in neg], 0))

        S = np.stack([diff_dir(LAKOFF_SCHEMAS_MML[sn]) for sn in SCHEMA_NAMES])  # 8 x d
        sv = np.linalg.svd(S, compute_uv=False)
        pr_real = participation_ratio(sv); t2_real = top2_frac(sv)
        Vt = np.linalg.svd(S, full_matrices=False)[2]
        top2 = Vt[:2]

        # null: random-partition pseudo-schemas, matched sizes
        sizes = [(len(set(p[0] for p in LAKOFF_SCHEMAS_MML[sn])),
                  len(set(p[1] for p in LAKOFF_SCHEMAS_MML[sn]))) for sn in SCHEMA_NAMES]
        prs = []
        for _ in range(200):
            rows = []
            for (npos, nneg) in sizes:
                pick = rng.choice(schema_pool, size=npos + nneg, replace=False)
                d = np.mean([res[w][L] for w in pick[:npos]], 0) - np.mean([res[w][L] for w in pick[npos:]], 0)
                rows.append(strip(d))
            prs.append(participation_ratio(np.linalg.svd(np.stack(rows), compute_uv=False)))
        prs = np.array(prs); pr_null_5 = float(np.percentile(prs, 5))

        # identification
        val = diff_dir(VALENCE_PAIRS); aro_raw = diff_dir(AROUSAL_PAIRS)
        aro = aro_raw - (aro_raw @ val) * val; aro /= np.linalg.norm(aro)  # valence-orthogonalised arousal
        cap = lambda v: float(np.sum((top2 @ v) ** 2))   # frac of v in top-2 subspace
        va_capture = 0.5 * (cap(val) + cap(aro))
        cos_va = float(val @ aro_raw)
        bal_aro = float(diff_dir(LAKOFF_SCHEMAS_MML["BALANCE"]) @ aro)
        # which schema closest to arousal / valence
        sc = {sn: diff_dir(LAKOFF_SCHEMAS_MML[sn]) for sn in SCHEMA_NAMES}
        top_aro = max(SCHEMA_NAMES, key=lambda s: abs(sc[s] @ aro))
        top_val = max(SCHEMA_NAMES, key=lambda s: abs(sc[s] @ val))

        code = verdict162(pr_real, pr_null_5, va_capture)
        print(f"\n--- Layer {L} ---")
        print(f"  PR(real) = {pr_real:.2f}   top-2 var = {100*t2_real:.0f}%   "
              f"| null PR mean {prs.mean():.2f}, 5th pctile {pr_null_5:.2f}")
        print(f"  valence/arousal capture of top-2 subspace = {100*va_capture:.0f}%  "
              f"(val {100*cap(val):.0f}%, aro {100*cap(aro):.0f}%); cos(val,aro_raw)={cos_va:+.2f}")
        print(f"  cos(BALANCE, arousal⊥val) = {bal_aro:+.2f}   "
              f"top schema~arousal: {top_aro.split('-')[0]}   top schema~valence: {top_val.split('-')[0]}")
        print(f"  -> {code}")
    del model, res
    gc.collect(); torch.mps.empty_cache()

print("\nV_TWO=two-primitive supported  V_LOWD=low-dim but >2  V_NULL=no structure")
