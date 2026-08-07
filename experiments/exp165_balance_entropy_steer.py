"""exp165_balance_entropy_steer.py — CAUSAL (PREREG_exp165.md, frozen
before this script). Inject the norm-independent BALANCE direction into
GPT-2-medium's residual stream and measure whether attention entropy
moves — with the magnitude confound controlled by matched-norm RANDOM
directions (the deadly alternative: any push changes ‖q‖ -> entropy).

Steer blocks.2.hook_resid_post (drives attn at L3, the shallow peak),
measure attn entropy at L3 (primary) + L8/12/16 (propagation). Dose grid
c in units of the median resid_post[L2] norm. BALANCE vs 12 random vs 7
other-schema directions, all matched-norm.

Reuses, does NOT re-type:
  build_layer_dirs/proj_out ...... exp161
  attn_entropy_per_query ......... attn_entropy_lib
  frozen prompts ................. exp166_prompt_verify.CANDIDATE
  ActAdd hook idiom .............. exp17 (resid + strength*direction)

Run:               ./lakoff/bin/python3 exp165_balance_entropy_steer.py
Validate harness:  ./lakoff/bin/python3 -c "import exp165_balance_entropy_steer as e; e.validate_harness()"
"""
import os, gc
import numpy as np
from huggingface_hub import get_token
os.environ["HF_TOKEN"] = get_token() or ""

from attn_entropy_lib import attn_entropy_per_query
from markedness_norm_protocol import build_word_lists, collect_residuals, SCHEMA_NAMES
from exp161_balance_entropy_prereg import build_layer_dirs, proj_out, AXIS_WORDS
from exp166_prompt_verify import CANDIDATE as FRESH_PROMPTS

# ---- frozen constants (PREREG_exp165.md) ----
STEER_PRIMARY = 2          # inject at resid_post[2] -> drives attn at L3
MEAS_PRIMARY = 3           # primary DV layer (shallow peak)
PROP_LAYERS = [8, 12, 16]  # propagation, report only
ROBUST_STEER = {1: 2, 3: 4}   # {steer_L: measure_L} robustness points
DOSE = [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]
N_RANDOM = 12
N_RANDOM_ROBUST = 4
N_BOOT = 1000
SEED = 165
DEVICE = "mps"


# ====================================================================
# PURE classifier (prereg decision rule) — selftest-able, no model
# ====================================================================
def ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm = x - x.mean()
    return float((xm @ (y - y.mean())) / (xm @ xm))


def spearman(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.sqrt((rx @ rx) * (ry @ ry)) + 1e-12))


def classify_causal(slope_bal, ci_slope, ci_diff_vs_rand,
                    random_slopes, schema_slopes, spearman_bal):
    """Prereg decision rule.
      slope_bal       : OLS slope of mean entropy on c, BALANCE dir
      ci_slope        : (lo,hi) cluster-bootstrap CI on slope_bal
      ci_diff_vs_rand : (lo,hi) CI on (slope_bal - mean(random slopes))
      random_slopes   : list of 12 matched-norm random-direction slopes
      schema_slopes   : list of 7 other-schema slopes
      spearman_bal    : Spearman(c, entropy) for BALANCE
    Returns CAUSAL_DIRECTION_CONFIRMED / GENERIC_MAGNITUDE / NULL."""
    real = not (ci_slope[0] <= 0 <= ci_slope[1])         # slope CI excludes 0
    negative = slope_bal < 0
    monotone = spearman_bal < 0
    if not (real and negative and monotone):
        return "NULL"
    beats_random = (ci_diff_vs_rand[1] < 0) and (slope_bal < min(random_slopes))
    n_schema = len(schema_slopes)
    beats_majority = sum(slope_bal < s for s in schema_slopes) > n_schema / 2
    if beats_random and beats_majority:
        return "CAUSAL_DIRECTION_CONFIRMED"
    return "GENERIC_MAGNITUDE"      # real, signed, monotone, but = any push


def selftest_classifier():
    rs = [-0.10, -0.08, -0.12, -0.09, -0.11, -0.07, -0.13, -0.10, -0.09,
          -0.11, -0.08, -0.10]      # random null ~ -0.10 (magnitude floor)
    sch = [-0.12, -0.09, -0.15, -0.08, -0.11, -0.10, -0.13]
    # CONFIRMED: BALANCE steeper than all random, beats majority schema, CI<0
    assert classify_causal(-0.40, (-0.50, -0.30), (-0.34, -0.18),
                           rs, sch, -0.95) == "CAUSAL_DIRECTION_CONFIRMED"
    # GENERIC: real & monotone but NOT beating random null
    assert classify_causal(-0.11, (-0.16, -0.06), (-0.04, +0.03),
                           rs, sch, -0.90) == "GENERIC_MAGNITUDE"
    # NULL: slope CI includes 0
    assert classify_causal(-0.03, (-0.09, +0.05), (-0.02, +0.06),
                           rs, sch, -0.20) == "NULL"
    # NULL: wrong sign (entropy rises with +BALANCE)
    assert classify_causal(+0.30, (+0.20, +0.40), (+0.10, +0.30),
                           rs, sch, +0.80) == "NULL"
    # beats random but NOT majority schema -> GENERIC (not direction-unique enough)
    sch_strong = [-0.50, -0.48, -0.45, -0.47, -0.44, -0.49, -0.46]
    assert classify_causal(-0.40, (-0.50, -0.30), (-0.34, -0.18),
                           rs, sch_strong, -0.95) == "GENERIC_MAGNITUDE"
    print("  selftest_classifier: PASS (CONFIRMED / GENERIC / NULL / wrong-sign / "
          "beats-random-not-schema).")


def synthetic_steering_test(seed=SEED):
    """Discrimination (prereg): the slope-vs-random-null machinery must
    (A) flag a planted DIRECTIONAL effect as beating random, and
    (B) classify a planted PURE-MAGNITUDE effect as GENERIC (BALANCE=random).
    Toy 'attention entropy' as a linear fn of an injected direction +/- a
    magnitude term; same ActAdd geometry as the real run."""
    rng = np.random.default_rng(seed)
    D = 64
    t = rng.standard_normal(D); t /= np.linalg.norm(t)           # true direction
    x0 = rng.standard_normal((300, D))                            # 300 'tokens'
    m = float(np.median(np.linalg.norm(x0, axis=1)))
    rands = []
    for _ in range(N_RANDOM):
        r = rng.standard_normal(D); r /= np.linalg.norm(r); rands.append(r)
    schemas = []
    for _ in range(7):
        s = rng.standard_normal(D); s /= np.linalg.norm(s); schemas.append(s)

    def slope_for(direction, Hfn):
        means = []
        for c in DOSE:
            x = x0 + c * m * direction
            means.append(Hfn(x).mean())
        return ols_slope(DOSE, means)

    # (A) DIRECTIONAL: entropy falls with (x·t); plus a magnitude term shared by all
    H_dir = lambda x: 2.0 - 0.5 * (x @ t) + 0.05 * np.linalg.norm(x, axis=1)
    sB = slope_for(t, H_dir)
    sR = [slope_for(r, H_dir) for r in rands]
    sS = [slope_for(s, H_dir) for s in schemas]
    spr = spearman(DOSE, [H_dir(x0 + c * m * t).mean() for c in DOSE])
    vA = classify_causal(sB, (sB - 0.02, sB + 0.02), (sB - max(sR) - 0.02, sB - max(sR) + 0.02),
                         sR, sS, spr)
    assert vA == "CAUSAL_DIRECTION_CONFIRMED", f"(A) directional -> {vA} (sB={sB:.3f}, maxR={max(sR):.3f})"
    # (B) PURE MAGNITUDE: entropy depends ONLY on ‖x‖ -> every direction same slope
    H_mag = lambda x: 2.0 + 0.10 * np.linalg.norm(x, axis=1)
    sB2 = slope_for(t, H_mag)
    sR2 = [slope_for(r, H_mag) for r in rands]
    sS2 = [slope_for(s, H_mag) for s in schemas]
    spr2 = spearman(DOSE, [H_mag(x0 + c * m * t).mean() for c in DOSE])
    diff2 = sB2 - float(np.mean(sR2))
    vB = classify_causal(sB2, (sB2 - 0.01, sB2 + 0.01), (diff2 - 0.01, diff2 + 0.01),
                         sR2, sS2, spr2)
    assert vB in ("GENERIC_MAGNITUDE", "NULL"), f"(B) magnitude -> {vB}"
    print(f"  synthetic_steering_test: PASS  (directional sB={sB:+.3f} beats "
          f"maxRandom={max(sR):+.3f} -> CONFIRMED; pure-magnitude -> {vB}).")


def validate_harness():
    print("exp165 harness validation (no model load)")
    print("=" * 60)
    selftest_classifier()
    synthetic_steering_test()
    import hashlib
    h = hashlib.sha256("\n".join(FRESH_PROMPTS).encode()).hexdigest()[:16]
    assert h == "4d54ff4297bd7e2c", f"prompt drift {h}"
    print(f"  prompts: exp166 set, checksum {h} OK; dose grid {DOSE}")
    print("HARNESS OK — safe to execute the model run.")


# ====================================================================
# Model run
# ====================================================================
def _build_directions(model, all_words, est_words, test_words, L_s, rng):
    """Return unit directions at resid_post[L_s], all orthogonalised to
    d_norm_ho and renormalised: BALANCE (bal_pn), 7 other schemas,
    N_RANDOM random. Plus median per-token resid norm m (from words)."""
    residuals = collect_residuals(model, [L_s], all_words, log_every=0)
    d = build_layer_dirs(residuals, L_s, all_words, est_words, test_words)
    dnorm = d["d_norm_ho"]
    bal = d["bal_pn"]                                   # BALANCE ⊥ d_norm
    schema = {sn: proj_out(d["schema"][sn], dnorm)
              for sn in SCHEMA_NAMES if sn != "BALANCE"}
    randoms = []
    Dd = bal.shape[0]
    for _ in range(N_RANDOM):
        r = rng.standard_normal(Dd)
        r = proj_out(r / np.linalg.norm(r), dnorm)      # matched: ⊥ d_norm, unit
        randoms.append(r)
    return dict(bal=bal, schema=schema, randoms=randoms), residuals


def _median_resid_norm(model, L_s):
    import torch
    hook = f"blocks.{L_s}.hook_resid_post"
    norms = []
    for prompt in FRESH_PROMPTS:
        toks = model.to_tokens(prompt)
        with torch.no_grad():
            _, c = model.run_with_cache(toks, names_filter=hook)
        r = c[hook][0].float().cpu().numpy()
        norms.extend(np.linalg.norm(r[1:], axis=1).tolist())
    return float(np.median(norms))


def _measure(model, L_s, meas_layers, direction, m, c):
    """Per-prompt mean attn entropy at each meas layer, steering
    resid_post[L_s] by c*m*direction (all positions). Returns
    {L: np.array(len=40)}."""
    import torch
    addvec = torch.tensor((c * m) * direction, dtype=torch.float32, device=DEVICE)
    hook = f"blocks.{L_s}.hook_resid_post"

    def hook_fn(resid, hook):
        return resid + addvec.to(resid.dtype)

    pnames = [f"blocks.{L}.attn.hook_pattern" for L in meas_layers]
    per = {L: [] for L in meas_layers}
    with model.hooks(fwd_hooks=[(hook, hook_fn)]):
        for prompt in FRESH_PROMPTS:
            toks = model.to_tokens(prompt)
            strs = model.to_str_tokens(prompt)
            with torch.no_grad():
                _, c_cache = model.run_with_cache(toks, names_filter=pnames)
            for L in meas_layers:
                pat = c_cache[f"blocks.{L}.attn.hook_pattern"][0].float().cpu().numpy()
                ent = attn_entropy_per_query(pat)
                vals = [ent[q] for q in range(1, len(strs))
                        if not np.isnan(ent[q])
                        and strs[q].strip().lower().strip(".,!?'\"") not in AXIS_WORDS]
                per[L].append(np.mean(vals) if vals else np.nan)
    return {L: np.array(v, float) for L, v in per.items()}


def _slope_with_ci(per_prompt_by_c, rng):
    """per_prompt_by_c: dict c -> array(40). Returns (slope, (lo,hi))
    via prompt-cluster bootstrap."""
    cs = sorted(per_prompt_by_c)
    means = [np.nanmean(per_prompt_by_c[c]) for c in cs]
    slope = ols_slope(cs, means)
    n = len(next(iter(per_prompt_by_c.values())))
    boots = []
    for _ in range(N_BOOT):
        idx = rng.choice(n, n, replace=True)
        bmeans = [np.nanmean(per_prompt_by_c[c][idx]) for c in cs]
        boots.append(ols_slope(cs, bmeans))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return slope, (float(lo), float(hi)), {c: per_prompt_by_c[c] for c in cs}


def _run_direction(model, L_s, meas_layers, direction, m):
    by_c = {L: {} for L in meas_layers}
    for c in DOSE:
        per = _measure(model, L_s, meas_layers, direction, m, c)
        for L in meas_layers:
            by_c[L][c] = per[L]
    return by_c


def main():
    import torch
    from transformer_lens import HookedTransformer
    print("exp165 — CAUSAL BALANCE-steer -> attention entropy "
          "(PREREG_exp165.md, frozen before this script)")
    validate_harness()
    rng = np.random.default_rng(SEED)

    all_words, est_words, test_words = build_word_lists()
    print(f"\nvocab: {len(all_words)} (est {len(est_words)}/test {len(test_words)})")
    model = HookedTransformer.from_pretrained("gpt2-medium", device=DEVICE)
    model.eval()

    # ---------- PRIMARY: steer L2 -> measure L3 (+ propagation) ----------
    print(f"\n{'='*72}\nPRIMARY: steer resid_post[{STEER_PRIMARY}] -> "
          f"attn entropy L{MEAS_PRIMARY} (+ L{PROP_LAYERS})\n{'='*72}")
    dirs, _ = _build_directions(model, all_words, est_words, test_words,
                                STEER_PRIMARY, rng)
    m = _median_resid_norm(model, STEER_PRIMARY)
    print(f"  median resid_post[{STEER_PRIMARY}] norm m = {m:.2f}; "
          f"dose in units of m: {DOSE}")
    meas = [MEAS_PRIMARY] + PROP_LAYERS

    bal_by = _run_direction(model, STEER_PRIMARY, meas, dirs["bal"], m)
    rand_by = [_run_direction(model, STEER_PRIMARY, [MEAS_PRIMARY], r, m)
               for r in dirs["randoms"]]
    sch_by = {sn: _run_direction(model, STEER_PRIMARY, [MEAS_PRIMARY], v, m)
              for sn, v in dirs["schema"].items()}

    # slopes at primary DV layer
    rng2 = np.random.default_rng(SEED + 1)
    s_bal, ci_bal, bal_cells = _slope_with_ci(bal_by[MEAS_PRIMARY], rng2)
    spr_bal = spearman(sorted(bal_by[MEAS_PRIMARY]),
                       [np.nanmean(bal_by[MEAS_PRIMARY][c]) for c in sorted(bal_by[MEAS_PRIMARY])])
    rand_slopes = [_slope_with_ci(r[MEAS_PRIMARY], np.random.default_rng(SEED + 10 + i))[0]
                   for i, r in enumerate(rand_by)]
    sch_slopes = {sn: _slope_with_ci(v[MEAS_PRIMARY], np.random.default_rng(SEED + 50))[0]
                  for sn, v in sch_by.items()}
    mean_rand = float(np.mean(rand_slopes))

    # bootstrap CI on (slope_bal - mean_random): resample prompts jointly
    cs = sorted(bal_by[MEAS_PRIMARY])
    nP = len(bal_by[MEAS_PRIMARY][cs[0]])
    diffs = []
    rngd = np.random.default_rng(SEED + 99)
    for _ in range(N_BOOT):
        idx = rngd.choice(nP, nP, replace=True)
        sb = ols_slope(cs, [np.nanmean(bal_by[MEAS_PRIMARY][c][idx]) for c in cs])
        srs = [ols_slope(cs, [np.nanmean(r[MEAS_PRIMARY][c][idx]) for c in cs])
               for r in rand_by]
        diffs.append(sb - np.mean(srs))
    ci_diff = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))

    print(f"\n  dose-response at L{MEAS_PRIMARY} (mean entropy per c):")
    print("   c:    " + "  ".join(f"{c:+.2f}" for c in cs))
    print("   BAL:  " + "  ".join(f"{np.nanmean(bal_by[MEAS_PRIMARY][c]):.3f}" for c in cs))
    print(f"\n  slope_BAL          = {s_bal:+.4f}  CI[{ci_bal[0]:+.4f},{ci_bal[1]:+.4f}]"
          f"  Spearman={spr_bal:+.2f}")
    print(f"  random slopes      : mean {mean_rand:+.4f}, "
          f"min {min(rand_slopes):+.4f}, max {max(rand_slopes):+.4f} (n={len(rand_slopes)})")
    print(f"  slope_BAL - random : CI[{ci_diff[0]:+.4f},{ci_diff[1]:+.4f}]")
    print(f"  schema slopes      : " +
          ", ".join(f"{sn.split('-')[0][:4]} {sv:+.3f}" for sn, sv in sch_slopes.items()))
    n_beat = sum(s_bal < sv for sv in sch_slopes.values())
    print(f"  BALANCE steeper than {n_beat}/{len(sch_slopes)} schemas")

    verdict = classify_causal(s_bal, ci_bal, ci_diff, rand_slopes,
                              list(sch_slopes.values()), spr_bal)

    # propagation (report only)
    print("\n  propagation (BALANCE slope at deeper layers, report only):")
    for L in PROP_LAYERS:
        sp, cip, _ = _slope_with_ci(bal_by[L], np.random.default_rng(SEED + 200 + L))
        print(f"    L{L}: slope {sp:+.4f} CI[{cip[0]:+.4f},{cip[1]:+.4f}]")

    # ---------- ROBUSTNESS: other shallow steer points ----------
    print(f"\n{'='*72}\nROBUSTNESS: BALANCE + {N_RANDOM_ROBUST} random at other "
          f"shallow steer points\n{'='*72}")
    for L_s, L_m in ROBUST_STEER.items():
        d2, _ = _build_directions(model, all_words, est_words, test_words, L_s,
                                  np.random.default_rng(SEED + L_s))
        m2 = _median_resid_norm(model, L_s)
        b2 = _run_direction(model, L_s, [L_m], d2["bal"], m2)
        sb2, ci2, _ = _slope_with_ci(b2[L_m], np.random.default_rng(SEED + 300 + L_s))
        r2 = [_slope_with_ci(_run_direction(model, L_s, [L_m], d2["randoms"][i], m2)[L_m],
                             np.random.default_rng(SEED + 400 + i))[0]
              for i in range(N_RANDOM_ROBUST)]
        tag = "beats random" if sb2 < min(r2) else "≈ random"
        print(f"  steer L{L_s}->meas L{L_m}: slope_BAL {sb2:+.4f} "
              f"CI[{ci2[0]:+.4f},{ci2[1]:+.4f}] | random min {min(r2):+.4f} -> {tag}")

    print("\n" + "=" * 72)
    print(f"VERDICT vs PREREG_exp165.md:  >>> {verdict} <<<")
    print("=" * 72)
    print("  CAUSAL_DIRECTION_CONFIRMED = BALANCE causally & specifically steers entropy")
    print("  GENERIC_MAGNITUDE = real but = any matched-norm push (correlation stays correlational)")
    print("  NULL = no causal effect at this layer (-> Arrow B: entropy->BALANCE)")

    del model
    gc.collect(); torch.mps.empty_cache()


if __name__ == "__main__":
    main()
