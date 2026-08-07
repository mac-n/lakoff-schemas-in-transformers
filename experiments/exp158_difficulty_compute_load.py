"""
exp158_difficulty_compute_load.py — the SECOND-EMBODIMENT test: does
DIFFICULTY-BURDEN recruit MLP compute load the way BALANCE recruits
residual norm? (Niamh's bar, 2026-06-11: one grounded concept is a
curiosity; two is a phenomenon.)

Physiology: per-word MLP activation mass at layer L = L1 norm of
mlp.hook_post at the last token ("compute load" / metabolic effort).
NB in pre-LN models the MLP input is LayerNorm'd, so load is not a
trivial restatement of residual norm — it reads the residual DIRECTION.
We verify independence explicitly via cos(d_load, d_norm).

Concept: DIFFICULTY-BURDEN schema axis (vocab already in
lakoff_canonical_vocabulary; POSITIVE pole = burden side: heavy,
burdensome, stuck, hard — so recruitment predicts POSITIVE cos).

Lessons from this week baked in:
  - d_load built from HELD-OUT estimation words only (exp154c/d).
  - Token count regressed out of the load metric BEFORE z-scoring
    (exp157b: last-token measures inherit tokenization structure).
  - Specificity table: ALL 8 schema axes vs d_load, plus the
    cross-couplings cos(DIFFICULTY, d_norm) and cos(BALANCE, d_load).
    A real recruitment is concept-specific; "everything couples to
    everything" means a residual confound direction, not embodiment.
  - Verdict logic is a pure function, synthetic-tested before the run.

PRE-REGISTRATION (2026-06-11, before running; this Claude):
  Committed prediction — and this one is genuinely adversarial to the
  hypothesis we WANT true: NO recruitment. |cos(DIFFICULTY, d_load)|
  < 0.3 at all decision layers. Reasoning: BALANCE→norm had a concrete
  carrier mechanism (markedness displaces norm; the axis reads it).
  For DIFFICULTY→load I can name no analogous property that
  burden-vocabulary actually HAS — the model's effort on a word is not
  related to the word's meaning about effort. My deflationary prior has
  been right four times this week. If the coupling shows up ANYWAY,
  against this committed prior, that is strong evidence — which is
  exactly why the prediction is worth committing.
  Point predictions:
    P1 Δload (burden-pole minus ease-pole, token-count-controlled) is
       small: |Δ| < 0.5 sd at decision layers.
    P2 |cos(DIFFICULTY, d_load)| < 0.3 at all decision layers [8,12,16,20].
    P3 independence holds: |cos(d_load, d_norm)| < 0.5 (the two
       physiologies are distinguishable axes).
  Decision rule:
    D1 cos(DIFFICULTY, d_load) >= +0.5 at all decision layers AND
       specific (DIFFICULTY ranks top among 8 schemas on |cos vs d_load|
       at a majority of decision layers, and beats its own d_norm
       coupling) AND independent (P3 holds): SECOND EMBODIMENT INSTANCE.
       My prior falsified; run GPT-2 + Llama for the contingency check;
       Part 2 becomes "concepts recruit afforded physiologies".
    D2 |cos(DIFFICULTY, d_load)| < 0.3 at all decision layers: no
       recruitment. The embodiment story stays single-instance; next
       candidate is LIGHT-DARK→attention-entropy (BURROWS).
    D3 otherwise (intermediate, or non-specific, or P3 fails): suspect
       a confound direction before interpreting — especially if ALL
       schemas couple to d_load, or d_load ≈ d_norm.
"""

import numpy as np
import torch
from transformer_lens import HookedTransformer

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML
from markedness_norm_protocol import (
    SCHEMA_NAMES, COMMON, RARE, build_word_lists, corrf,
)

LAYERS = [4, 8, 12, 16, 20]
DECISION_LAYERS = [8, 12, 16, 20]


# ---------------- verdict logic (pure, selftest-able) ----------------

def verdict158(summary, decision_layers, couple_hi=0.5, couple_lo=0.3,
               indep=0.5):
    """summary[L]: dict with cos_diff_dload, cos_diff_dnorm,
    cos_dload_dnorm, schema_rank (1 = DIFFICULTY top among 8 on |cos|)."""
    cds = [summary[L]["cos_diff_dload"] for L in decision_layers]
    q_couple = all(c >= couple_hi for c in cds)
    q_null = all(abs(c) < couple_lo for c in cds)
    q_spec = (sum(1 for L in decision_layers if summary[L]["schema_rank"] == 1)
              > len(decision_layers) / 2
              and all(summary[L]["cos_diff_dload"] > summary[L]["cos_diff_dnorm"]
                      for L in decision_layers))
    q_indep = all(abs(summary[L]["cos_dload_dnorm"]) < indep
                  for L in decision_layers)
    if q_couple and q_spec and q_indep:
        return "D1"
    if q_null:
        return "D2"
    return "D3"


def selftest_verdict158(dl=(8, 12, 16, 20)):
    def fake(cdl, cdn, cdd, rank):
        return {L: {"cos_diff_dload": cdl, "cos_diff_dnorm": cdn,
                    "cos_dload_dnorm": cdd, "schema_rank": rank} for L in dl}
    assert verdict158(fake(+0.65, +0.10, +0.20, 1), dl) == "D1"
    assert verdict158(fake(+0.05, +0.10, +0.20, 4), dl) == "D2"
    assert verdict158(fake(-0.25, +0.10, +0.20, 2), dl) == "D2"
    assert verdict158(fake(+0.65, +0.10, +0.80, 1), dl) == "D3"  # not indep
    assert verdict158(fake(+0.65, +0.70, +0.20, 1), dl) == "D3"  # not specific
    assert verdict158(fake(+0.40, +0.10, +0.20, 1), dl) == "D3"  # intermediate
    print("selftest_verdict158: all branches fire correctly.")


print("Running verdict-logic selftest first (synthetic-test convention)...")
selftest_verdict158()

# ---------------- collection ----------------

print("Loading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device="mps")
model.eval()

all_words, est_words, test_words = build_word_lists()
# DIFFICULTY vocab joins the held-out test set for d_load estimation
diff_vocab = {w for p in LAKOFF_SCHEMAS_MML["DIFFICULTY-BURDEN"] for w in p}
est_words_load = [w for w in est_words if w not in diff_vocab]
print(f"Words: {len(all_words)} total; d_load estimation set "
      f"{len(est_words_load)} (held out: suffix pairs, BALANCE, DIFFICULTY)")

resid_hooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
mlp_hooks = [f"blocks.{L}.mlp.hook_post" for L in LAYERS]
print(f"Collecting residuals + MLP load for {len(all_words)} words...")
residuals, load, tokcount = {}, {L: {} for L in LAYERS}, {}
for k, w in enumerate(all_words):
    toks = model.to_tokens(w)
    tokcount[w] = toks.shape[1]
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=resid_hooks + mlp_hooks)
    residuals[w] = {L: cache[f"blocks.{L}.hook_resid_post"][0, -1, :]
                    .float().cpu().numpy() for L in LAYERS}
    for L in LAYERS:
        load[L][w] = float(cache[f"blocks.{L}.mlp.hook_post"][0, -1, :]
                           .float().abs().sum())
    if (k + 1) % 100 == 0:
        print(f"  {k+1}/{len(all_words)}")

print("\n" + "=" * 78)
print("exp158 — DIFFICULTY-BURDEN vs MLP compute load (Pythia 410M)")
print("=" * 78)

summary = {}
for L in LAYERS:
    arr = np.stack([residuals[w][L] for w in all_words])
    aniso = arr.mean(axis=0); aniso /= np.linalg.norm(aniso)

    def mean_acts(words):
        return np.mean([residuals[w][L] for w in words], axis=0)

    freq = mean_acts(COMMON) - mean_acts(RARE); freq /= np.linalg.norm(freq)
    freq_o = freq - (freq @ aniso) * aniso; freq_o /= np.linalg.norm(freq_o)

    def strip(d):
        d = d - (d @ aniso) * aniso
        d = d - (d @ freq_o) * freq_o
        return d / np.linalg.norm(d)

    norms = {w: float(np.linalg.norm(residuals[w][L])) for w in all_words}
    units = {w: residuals[w][L] / norms[w] for w in all_words}

    # token-count-controlled load (exp157b lesson): regress tc out FIRST
    tc = np.array([tokcount[w] for w in all_words], float)
    lv = np.array([load[L][w] for w in all_words])
    A = np.vstack([tc, np.ones_like(tc)]).T
    coef, *_ = np.linalg.lstsq(A, lv, rcond=None)
    lres = dict(zip(all_words, lv - A @ coef))

    def cov_dir(words, scalar):
        sv = np.array([scalar[w] for w in words])
        zz = dict(zip(words, (sv - sv.mean()) / sv.std()))
        d = np.sum([zz[w] * units[w] for w in words], axis=0)
        return strip(d / np.linalg.norm(d)), zz

    d_load, _ = cov_dir(est_words_load, lres)
    # d_norm built the same held-out way, same estimation set, for the
    # cross-coupling and independence rows (token-count-controlled too)
    nres_v = np.array([norms[w] for w in all_words])
    coefn, *_ = np.linalg.lstsq(A, nres_v, rcond=None)
    nres = dict(zip(all_words, nres_v - A @ coefn))
    d_norm, _ = cov_dir(est_words_load, nres)

    # carrier sanity (in/out of estimation sample)
    r_in = corrf([units[w] @ d_load for w in est_words_load],
                 [lres[w] for w in est_words_load])
    held = [w for w in all_words if w not in est_words_load]
    r_out = corrf([units[w] @ d_load for w in held], [lres[w] for w in held])

    def schema_dir(sn):
        pairs = LAKOFF_SCHEMAS_MML[sn]
        pos = sorted(set(p[0] for p in pairs)); neg = sorted(set(p[1] for p in pairs))
        raw = mean_acts(pos) - mean_acts(neg)
        return strip(raw / np.linalg.norm(raw))

    schema_cos = {sn: float(schema_dir(sn) @ d_load) for sn in SCHEMA_NAMES}
    rank = 1 + sum(1 for sn in SCHEMA_NAMES if sn != "DIFFICULTY-BURDEN"
                   and abs(schema_cos[sn]) > abs(schema_cos["DIFFICULTY-BURDEN"]))

    # raw physiological check: Δload between poles (token-count-controlled, sd units)
    pos_w = sorted(set(p[0] for p in LAKOFF_SCHEMAS_MML["DIFFICULTY-BURDEN"]))
    neg_w = sorted(set(p[1] for p in LAKOFF_SCHEMAS_MML["DIFFICULTY-BURDEN"]))
    sd = np.std([lres[w] for w in all_words])
    dload_poles = (np.mean([lres[w] for w in pos_w])
                   - np.mean([lres[w] for w in neg_w])) / sd

    summary[L] = {
        "cos_diff_dload": schema_cos["DIFFICULTY-BURDEN"],
        "cos_diff_dnorm": float(schema_dir("DIFFICULTY-BURDEN") @ d_norm),
        "cos_bal_dload": schema_cos["BALANCE"],
        "cos_bal_dnorm": float(schema_dir("BALANCE") @ d_norm),
        "cos_dload_dnorm": float(d_load @ d_norm),
        "schema_rank": rank, "schema_cos": schema_cos,
        "dload_poles_sd": float(dload_poles),
        "r_in": r_in, "r_out": r_out,
    }
    s = summary[L]
    print(f"\n--- Layer {L} ---")
    print(f"  carrier: in-sample {s['r_in']:+.3f}, out-of-sample {s['r_out']:+.3f}")
    print(f"  Δload burden−ease poles (tc-controlled, sd units): {s['dload_poles_sd']:+.2f}")
    print(f"  cos(DIFFICULTY, d_load) = {s['cos_diff_dload']:+.3f}   "
          f"[rank {s['schema_rank']}/8 among schemas]")
    print(f"  cross-couplings: cos(DIFF, d_norm) = {s['cos_diff_dnorm']:+.3f}   "
          f"cos(BAL, d_load) = {s['cos_bal_dload']:+.3f}   "
          f"cos(BAL, d_norm) = {s['cos_bal_dnorm']:+.3f}")
    print(f"  independence: cos(d_load, d_norm) = {s['cos_dload_dnorm']:+.3f}")
    print("  all schemas vs d_load: " +
          ", ".join(f"{sn.split('-')[0][:5]} {schema_cos[sn]:+.2f}"
                    for sn in SCHEMA_NAMES))

print("\n" + "=" * 78)
print("VERDICT vs pre-registered decision rule")
print("=" * 78)
code = verdict158(summary, DECISION_LAYERS)
cds = [summary[L]["cos_diff_dload"] for L in DECISION_LAYERS]
print(f"  cos(DIFFICULTY, d_load) at decision layers: "
      + ", ".join(f"L{L}:{c:+.2f}" for L, c in zip(DECISION_LAYERS, cds)))
print(f"  independence |cos(d_load,d_norm)|: "
      + ", ".join(f"L{L}:{abs(summary[L]['cos_dload_dnorm']):.2f}"
                  for L in DECISION_LAYERS))
print(f"  -> {code}: " + {
    "D1": "SECOND EMBODIMENT INSTANCE. Prior falsified; run GPT-2+Llama "
          "contingency check; Part 2 = 'concepts recruit afforded physiologies'.",
    "D2": "no recruitment — embodiment stays single-instance; next candidate "
          "LIGHT-DARK→attention-entropy (BURROWS).",
    "D3": "intermediate/non-specific/entangled — suspect a confound direction "
          "before interpreting anything.",
}[code])
