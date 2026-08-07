"""exp165b_all_layer_steer.py — ALL-LAYER causal steer (PREREG_exp165b.md,
frozen before this script). Inject the norm-independent BALANCE direction
at EVERY layer of GPT-2-medium simultaneously and measure whether global
attention entropy moves, vs 16 matched all-layer random directions.

Niamh's design: within-layer correlation ≠ downstream causal channel; a
single shallow nudge (exp165) is too weak; all-layer accumulation is the
principled strong test, and per-layer response answers WHERE the lever is.

Reuses: build_layer_dirs/proj_out (exp161), attn_entropy_per_query (lib),
ols_slope/spearman (exp165), frozen prompts (exp166).

Run:               ./lakoff/bin/python3 exp165b_all_layer_steer.py
Validate harness:  ./lakoff/bin/python3 -c "import exp165b_all_layer_steer as e; e.validate_harness()"
"""
import os, gc, hashlib
import numpy as np
from huggingface_hub import get_token
os.environ["HF_TOKEN"] = get_token() or ""

from attn_entropy_lib import attn_entropy_per_query
from markedness_norm_protocol import build_word_lists, collect_residuals, SCHEMA_NAMES
from exp161_balance_entropy_prereg import build_layer_dirs, proj_out, AXIS_WORDS
from exp166_prompt_verify import CANDIDATE as FRESH_PROMPTS
from exp165_balance_entropy_steer import ols_slope, spearman

# ---- frozen constants (PREREG_exp165b.md) ----
N_LAYERS = 24
DOSE = [-0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5]
N_RANDOM = 16
N_BOOT = 1000
SEED = 165
DEVICE = "mps"
# RULE PARAMETERS — must equal PREREG_exp165b.md (asserted below)
Z_THRESH = -1.64
SCHEMA_MAJORITY = 4            # strictly > 7/2
PROMPT_CHECKSUM = "4d54ff4297bd7e2c"


def _assert_rule_matches_prereg():
    """Checksum-style guard against the exp165 rule-drift recurring."""
    import re
    with open(os.path.join(os.path.dirname(__file__) or ".",
                           "PREREG_exp165b.md")) as f:
        txt = f.read()
    assert "Z_THRESH = −1.64" in txt, "prereg Z_THRESH text missing/changed"
    assert "≥4" in txt and "SCHEMA_MAJORITY" in txt, "prereg schema-majority changed"
    assert abs(Z_THRESH - (-1.64)) < 1e-9 and SCHEMA_MAJORITY == 4, "code != prereg rule"


# ====================================================================
# PURE classifier (corrected z-outlier rule) — selftest-able
# ====================================================================
def classify_causal_v2(slope_bal, ci_slope, ci_diff_vs_rand,
                       random_slopes, schema_slopes, spearman_bal):
    real = not (ci_slope[0] <= 0 <= ci_slope[1])
    negative = slope_bal < 0
    monotone = spearman_bal < 0
    if not (real and negative and monotone):
        return "NULL"
    mean_r = float(np.mean(random_slopes))
    sd_r = float(np.std(random_slopes, ddof=1))
    z = (slope_bal - mean_r) / sd_r if sd_r > 0 else 0.0
    beats_random = (z < Z_THRESH) and (ci_diff_vs_rand[1] < 0)
    beats_schema = sum(slope_bal < s for s in schema_slopes) >= SCHEMA_MAJORITY
    if beats_random and beats_schema:
        return "CAUSAL_DIRECTION_CONFIRMED"
    return "GENERIC_MAGNITUDE"


def selftest_classifier():
    rs = [-0.10, -0.08, -0.12, -0.09, -0.11, -0.07, -0.13, -0.10,
          -0.09, -0.11, -0.08, -0.10, -0.09, -0.12, -0.10, -0.11]  # ~ -0.10, sd~0.016
    sch = [-0.12, -0.09, -0.15, -0.08, -0.11, -0.10, -0.13]
    # CONFIRMED: clear outlier below cloud, beats >=4 schema, CI excl 0
    assert classify_causal_v2(-0.40, (-0.50, -0.30), (-0.31, -0.18),
                              rs, sch, -0.95) == "CAUSAL_DIRECTION_CONFIRMED"
    # GENERIC: negative/monotone/real but only ~2sigma is NOT enough here if within cloud:
    #   slope -0.135 vs mean -0.10 sd~0.016 -> z ~ -2.2 (outlier) -> would CONFIRM;
    #   pick -0.118 -> z ~ -1.1 -> NOT outlier -> GENERIC
    assert classify_causal_v2(-0.118, (-0.17, -0.06), (-0.03, +0.01),
                              rs, sch, -0.90) == "GENERIC_MAGNITUDE"
    # NULL: CI includes 0
    assert classify_causal_v2(-0.03, (-0.09, +0.05), (-0.02, +0.06),
                              rs, sch, -0.3) == "NULL"
    # NULL: wrong sign
    assert classify_causal_v2(+0.30, (+0.2, +0.4), (+0.1, +0.3),
                              rs, sch, +0.8) == "NULL"
    # outlier but fails schema majority -> GENERIC
    sch_strong = [-0.50, -0.48, -0.45, -0.47, -0.44, -0.49, -0.46]
    assert classify_causal_v2(-0.40, (-0.50, -0.30), (-0.31, -0.18),
                              rs, sch_strong, -0.95) == "GENERIC_MAGNITUDE"
    print("  selftest_classifier: PASS (z-outlier CONFIRMED / within-cloud GENERIC / "
          "NULL / wrong-sign / fails-schema).")


def synthetic_steering_test(seed=SEED):
    """Directional planted effect -> CONFIRMED; pure-magnitude -> not."""
    rng = np.random.default_rng(seed)
    D = 64
    t = rng.standard_normal(D); t /= np.linalg.norm(t)
    x0 = rng.standard_normal((300, D))
    m = float(np.median(np.linalg.norm(x0, axis=1)))
    rands = [(lambda v: v / np.linalg.norm(v))(rng.standard_normal(D)) for _ in range(N_RANDOM)]
    schemas = [(lambda v: v / np.linalg.norm(v))(rng.standard_normal(D)) for _ in range(7)]

    def slope_for(direction, Hfn):
        return ols_slope(DOSE, [Hfn(x0 + c * m * direction).mean() for c in DOSE])

    H_dir = lambda x: 2.0 - 0.5 * (x @ t) + 0.05 * np.linalg.norm(x, axis=1)
    sB = slope_for(t, H_dir); sR = [slope_for(r, H_dir) for r in rands]
    sS = [slope_for(s, H_dir) for s in schemas]
    spr = spearman(DOSE, [H_dir(x0 + c * m * t).mean() for c in DOSE])
    diff = sB - float(np.mean(sR))
    vA = classify_causal_v2(sB, (sB - 0.02, sB + 0.02), (diff - 0.02, diff + 0.02),
                            sR, sS, spr)
    assert vA == "CAUSAL_DIRECTION_CONFIRMED", f"(A) -> {vA}"
    H_mag = lambda x: 2.0 + 0.10 * np.linalg.norm(x, axis=1)
    sB2 = slope_for(t, H_mag); sR2 = [slope_for(r, H_mag) for r in rands]
    sS2 = [slope_for(s, H_mag) for s in schemas]
    spr2 = spearman(DOSE, [H_mag(x0 + c * m * t).mean() for c in DOSE])
    diff2 = sB2 - float(np.mean(sR2))
    vB = classify_causal_v2(sB2, (sB2 - 0.01, sB2 + 0.01), (diff2 - 0.01, diff2 + 0.01),
                            sR2, sS2, spr2)
    assert vB in ("GENERIC_MAGNITUDE", "NULL"), f"(B) -> {vB}"
    print(f"  synthetic_steering_test: PASS (directional -> CONFIRMED; magnitude -> {vB}).")


def validate_harness():
    print("exp165b harness validation (no model load)")
    print("=" * 60)
    _assert_rule_matches_prereg()
    print(f"  rule asserted vs prereg: Z_THRESH={Z_THRESH}, SCHEMA_MAJORITY={SCHEMA_MAJORITY}")
    selftest_classifier()
    synthetic_steering_test()
    h = hashlib.sha256("\n".join(FRESH_PROMPTS).encode()).hexdigest()[:16]
    assert h == PROMPT_CHECKSUM, f"prompt drift {h}"
    print(f"  prompts: exp166 set, checksum {h} OK; dose {DOSE}")
    print("HARNESS OK — safe to execute the model run.")


# ====================================================================
# Model run
# ====================================================================
def _build_all_layer_dirs(model, all_words, est_words, test_words, rng):
    layers = list(range(N_LAYERS))
    residuals = collect_residuals(model, layers, all_words, log_every=0)
    bal, schema, dnorm = {}, {sn: {} for sn in SCHEMA_NAMES if sn != "BALANCE"}, {}
    for L in layers:
        d = build_layer_dirs(residuals, L, all_words, est_words, test_words)
        bal[L] = d["bal_pn"]; dnorm[L] = d["d_norm_ho"]
        for sn in schema:
            schema[sn][L] = proj_out(d["schema"][sn], d["d_norm_ho"])
    Dd = bal[0].shape[0]
    randoms = []
    for _ in range(N_RANDOM):
        rd = {}
        for L in layers:
            r = rng.standard_normal(Dd)
            rd[L] = proj_out(r / np.linalg.norm(r), dnorm[L])
        randoms.append(rd)
    return bal, schema, randoms


def _median_norms(model):
    import torch
    hooks = [f"blocks.{L}.hook_resid_post" for L in range(N_LAYERS)]
    acc = {L: [] for L in range(N_LAYERS)}
    for prompt in FRESH_PROMPTS:
        toks = model.to_tokens(prompt)
        with torch.no_grad():
            _, c = model.run_with_cache(toks, names_filter=hooks)
        for L in range(N_LAYERS):
            r = c[f"blocks.{L}.hook_resid_post"][0].float().cpu().numpy()
            acc[L].extend(np.linalg.norm(r[1:], axis=1).tolist())
    return {L: float(np.median(v)) for L, v in acc.items()}


def _measure_all_layer(model, dir_by_layer, m_by_layer, c):
    """All-layer ActAdd at coefficient c. Returns (global_per_prompt[40],
    per_layer_per_prompt {L: [40]}) of mean attention entropy."""
    import torch
    addvecs = {L: torch.tensor((c * m_by_layer[L]) * dir_by_layer[L],
                               dtype=torch.float32, device=DEVICE)
               for L in range(N_LAYERS)}

    def make(L):
        v = addvecs[L]
        def hook(resid, hook):
            return resid + v.to(resid.dtype)
        return hook

    fwd = [(f"blocks.{L}.hook_resid_post", make(L)) for L in range(N_LAYERS)]
    pnames = [f"blocks.{L}.attn.hook_pattern" for L in range(N_LAYERS)]
    per_layer = {L: [] for L in range(N_LAYERS)}
    glob = []
    with model.hooks(fwd_hooks=fwd):
        for prompt in FRESH_PROMPTS:
            toks = model.to_tokens(prompt)
            strs = model.to_str_tokens(prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(toks, names_filter=pnames)
            keep = [q for q in range(1, len(strs))
                    if strs[q].strip().lower().strip(".,!?'\"") not in AXIS_WORDS]
            allvals = []
            for L in range(N_LAYERS):
                pat = cache[f"blocks.{L}.attn.hook_pattern"][0].float().cpu().numpy()
                ent = attn_entropy_per_query(pat)
                vals = [ent[q] for q in keep if not np.isnan(ent[q])]
                per_layer[L].append(np.mean(vals) if vals else np.nan)
                allvals.extend(vals)
            glob.append(np.mean(allvals) if allvals else np.nan)
            del cache
    return np.array(glob, float), {L: np.array(v, float) for L, v in per_layer.items()}


def _run_direction(model, dir_by_layer, m_by_layer):
    glob_by_c, perlayer_by_c = {}, {}
    for c in DOSE:
        g, pl = _measure_all_layer(model, dir_by_layer, m_by_layer, c)
        glob_by_c[c] = g; perlayer_by_c[c] = pl
    return glob_by_c, perlayer_by_c


def _slope_ci(by_c, rng):
    cs = sorted(by_c)
    means = [np.nanmean(by_c[c]) for c in cs]
    slope = ols_slope(cs, means)
    n = len(by_c[cs[0]])
    boots = [ols_slope(cs, [np.nanmean(by_c[c][rng.choice(n, n, True)]) for c in cs])
             for _ in range(N_BOOT)]
    return slope, (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def main():
    import torch
    from transformer_lens import HookedTransformer
    print("exp165b — ALL-LAYER causal BALANCE-steer (PREREG_exp165b.md)")
    validate_harness()
    rng = np.random.default_rng(SEED)
    all_words, est_words, test_words = build_word_lists()
    print(f"\nvocab: {len(all_words)}")
    model = HookedTransformer.from_pretrained("gpt2-medium", device=DEVICE)
    model.eval()

    print("\nbuilding all-layer directions + median norms...")
    bal, schema, randoms = _build_all_layer_dirs(model, all_words, est_words, test_words, rng)
    m = _median_norms(model)
    print(f"  median resid norms: L0 {m[0]:.1f} ... L12 {m[12]:.1f} ... L23 {m[23]:.1f}")

    print("\nsteering BALANCE (all layers) across dose grid...")
    bal_g, bal_pl = _run_direction(model, bal, m)
    print("steering 16 random all-layer directions...")
    rand_g = [_run_direction(model, rd, m)[0] for rd in randoms]
    print("steering 7 schema all-layer directions...")
    sch_g = {sn: _run_direction(model, sd, m)[0] for sn, sd in schema.items()}

    cs = sorted(DOSE)
    rng2 = np.random.default_rng(SEED + 1)
    s_bal, ci_bal = _slope_ci(bal_g, rng2)
    spr = spearman(cs, [np.nanmean(bal_g[c]) for c in cs])
    rand_sl = [_slope_ci(g, np.random.default_rng(SEED + 10 + i))[0] for i, g in enumerate(rand_g)]
    sch_sl = {sn: _slope_ci(g, np.random.default_rng(SEED + 50))[0] for sn, g in sch_g.items()}
    mean_r, sd_r = float(np.mean(rand_sl)), float(np.std(rand_sl, ddof=1))

    nP = len(bal_g[cs[0]]); rngd = np.random.default_rng(SEED + 99); diffs = []
    for _ in range(N_BOOT):
        idx = rngd.choice(nP, nP, True)
        sb = ols_slope(cs, [np.nanmean(bal_g[c][idx]) for c in cs])
        sr = [ols_slope(cs, [np.nanmean(g[c][idx]) for c in cs]) for g in rand_g]
        diffs.append(sb - np.mean(sr))
    ci_diff = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    z = (s_bal - mean_r) / sd_r if sd_r > 0 else 0.0

    print(f"\n{'='*72}\nGLOBAL DV (mean attn entropy over ALL layers)\n{'='*72}")
    print("  c:        " + "  ".join(f"{c:+.2f}" for c in cs))
    print("  BAL ent:  " + "  ".join(f"{np.nanmean(bal_g[c]):.3f}" for c in cs))
    print(f"\n  slope_BAL = {s_bal:+.4f}  CI[{ci_bal[0]:+.4f},{ci_bal[1]:+.4f}]  Spearman={spr:+.2f}")
    print(f"  random null: mean {mean_r:+.4f}  sd {sd_r:.4f} (n={len(rand_sl)})  "
          f"min {min(rand_sl):+.4f} max {max(rand_sl):+.4f}")
    print(f"  z(BAL vs random cloud) = {z:+.2f}   (CONFIRM needs z < {Z_THRESH})")
    print(f"  slope_BAL - mean_random: CI[{ci_diff[0]:+.4f},{ci_diff[1]:+.4f}]")
    n_beat = sum(s_bal < sv for sv in sch_sl.values())
    print(f"  schema slopes: " + ", ".join(f"{sn.split('-')[0][:4]} {sv:+.3f}"
                                            for sn, sv in sch_sl.items()))
    print(f"  BALANCE steeper than {n_beat}/7 schemas (CONFIRM needs >= {SCHEMA_MAJORITY})")

    verdict = classify_causal_v2(s_bal, ci_bal, ci_diff, rand_sl,
                                 list(sch_sl.values()), spr)

    print(f"\n  PER-LAYER BALANCE slope (where does entropy respond?):")
    for L in range(N_LAYERS):
        sl, _ = _slope_ci(bal_pl_at(bal_pl, L), np.random.default_rng(SEED + 500 + L))
        bar = "#" * int(min(40, abs(sl) * 200))
        print(f"    L{L:>2}: {sl:+.4f} {bar}")

    print(f"\n{'='*72}\nVERDICT vs PREREG_exp165b.md:  >>> {verdict} <<<\n{'='*72}")
    print("  CONFIRMED = BALANCE is a random-cloud outlier (causal & specific)")
    print("  GENERIC   = real but within random cloud (matched push)")
    print("  NULL      = no causal effect -> Arrow B (entropy->BALANCE)")
    del model; gc.collect(); torch.mps.empty_cache()


def bal_pl_at(perlayer_by_c, L):
    """Reshape per-layer storage {c: {L: [40]}} into {c: [40]} for layer L."""
    return {c: perlayer_by_c[c][L] for c in perlayer_by_c}


if __name__ == "__main__":
    main()
