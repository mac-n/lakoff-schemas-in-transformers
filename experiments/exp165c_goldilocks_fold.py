"""exp165c_goldilocks_fold.py — GOLDILOCKS re-test (PREREG_exp165c.md,
frozen before this finer-grid run). exp165b scored the all-layer steer
with a LINEAR slope and called it GENERIC; but BALANCE's dose-curve was a
∪. A linear slope is blind to a ∪. If BALANCE is a Goldilocks/homeostatic
primitive (Niamh), a ∪ is the PREDICTED shape — and "pushing BALANCE too
hard is unbalancing" (☯: the axis contains its own opposite). This tests
whether the bowl is BALANCE-SPECIFIC (curvature an outlier vs matched
random directions) on a finer, denser, symmetric dose grid.

Reuses exp165b's all-layer steering machinery wholesale; adds quadratic
curvature / minimum-location / asymmetry statistics.

Run:               ./lakoff/bin/python3 exp165c_goldilocks_fold.py
Validate harness:  ./lakoff/bin/python3 -c "import exp165c_goldilocks_fold as e; e.validate_harness()"
"""
import os, gc, hashlib
import numpy as np
from huggingface_hub import get_token
os.environ["HF_TOKEN"] = get_token() or ""

from markedness_norm_protocol import build_word_lists
from exp166_prompt_verify import CANDIDATE as FRESH_PROMPTS
from exp165_balance_entropy_steer import ols_slope, spearman
from exp165b_all_layer_steer import (_build_all_layer_dirs, _median_norms,
                                     _measure_all_layer)

# ---- frozen constants (PREREG_exp165c.md) ----
DOSE_C = [-0.50, -0.35, -0.25, -0.15, -0.10, -0.05, 0.0,
          0.05, 0.10, 0.15, 0.25, 0.35, 0.50]
OUTER = [0.25, 0.35, 0.50]        # for the asymmetry statistic
N_RANDOM = 16
N_BOOT = 1000
SEED = 165
DEVICE = "mps"
# RULE PARAMETERS — must equal PREREG_exp165c.md (asserted at runtime)
CURV_Z = 1.64
ASYM_Z = 1.0
SCHEMA_MAJORITY = 4
PROMPT_CHECKSUM = "4d54ff4297bd7e2c"


def _assert_rule_matches_prereg():
    with open(os.path.join(os.path.dirname(__file__) or ".",
                           "PREREG_exp165c.md")) as f:
        txt = f.read()
    assert "CURV_Z   = +1.64" in txt, "prereg CURV_Z text changed"
    assert "SCHEMA_MAJORITY = 4" in txt, "prereg SCHEMA_MAJORITY changed"
    assert CURV_Z == 1.64 and SCHEMA_MAJORITY == 4, "code != prereg rule"


# ====================================================================
# Curve statistics
# ====================================================================
def quad_stats(by_c):
    """by_c: {c: array(40)}. Returns (k, b, a, cstar, argmin, means)."""
    cs = sorted(by_c)
    means = np.array([np.nanmean(by_c[c]) for c in cs])
    k, b, a = np.polyfit(cs, means, 2)
    cstar = (-b / (2 * k)) if abs(k) > 1e-12 else float("nan")
    argmin = cs[int(np.argmin(means))]
    return float(k), float(b), float(a), float(cstar), float(argmin), means


def asymmetry(by_c):
    neg = np.mean([np.nanmean(by_c[-c]) for c in OUTER])
    pos = np.mean([np.nanmean(by_c[c]) for c in OUTER])
    return float(neg - pos)


def boot_curvature(by_c, rng, n=N_BOOT):
    cs = sorted(by_c); nP = len(by_c[cs[0]])
    ks = []
    for _ in range(n):
        idx = rng.choice(nP, nP, True)
        means = [np.nanmean(by_c[c][idx]) for c in cs]
        ks.append(np.polyfit(cs, means, 2)[0])
    return np.array(ks)


# ====================================================================
# PURE classifier (selftest-able)
# ====================================================================
def classify_goldilocks(k_bal, ci_k, z_k, ci_kdiff, cstar_bal, argmin_bal,
                        random_ks, schema_ks):
    real_bowl = (k_bal > 0) and (ci_k[0] > 0)
    if not real_bowl:
        return "NOT_GOLDILOCKS"
    specific = (z_k > CURV_Z) and (ci_kdiff[0] > 0)
    floor_balanced = (cstar_bal > 0) and (argmin_bal >= 0)
    beats_schema = sum(k_bal > sk for sk in schema_ks) >= SCHEMA_MAJORITY
    if specific and floor_balanced and beats_schema:
        return "GOLDILOCKS_CONFIRMED"
    return "GENERIC_SATURATION"


def selftest_classifier():
    rks = [0.30, 0.28, 0.33, 0.29, 0.31, 0.27, 0.34, 0.30,
           0.29, 0.32, 0.28, 0.31, 0.30, 0.33, 0.29, 0.30]   # ~0.30, sd~0.02
    sks = [0.30, 0.25, 0.40, 0.28, 0.33, 0.31, 0.29]
    # CONFIRMED: big bowl, outlier above cloud, floor on +side, beats >=4 schema
    assert classify_goldilocks(0.80, (0.70, 0.90), 24.0, (0.45, 0.55),
                               0.12, 0.05, rks, sks) == "GOLDILOCKS_CONFIRMED"
    # GENERIC_SATURATION: bowls but within random cloud (z small)
    assert classify_goldilocks(0.32, (0.25, 0.39), 0.9, (-0.02, 0.06),
                               0.10, 0.05, rks, sks) == "GENERIC_SATURATION"
    # NOT_GOLDILOCKS: no reliable bowl (CI includes 0)
    assert classify_goldilocks(0.05, (-0.03, 0.13), 0.5, (-0.20, 0.10),
                               0.10, 0.05, rks, sks) == "NOT_GOLDILOCKS"
    # NOT_GOLDILOCKS: negative curvature (∩, not ∪)
    assert classify_goldilocks(-0.40, (-0.50, -0.30), -5.0, (-0.6, -0.4),
                               0.0, 0.0, rks, sks) == "NOT_GOLDILOCKS"
    # outlier bowl but floor on WRONG side (vertex<0) -> GENERIC
    assert classify_goldilocks(0.80, (0.70, 0.90), 24.0, (0.45, 0.55),
                               -0.10, -0.05, rks, sks) == "GENERIC_SATURATION"
    print("  selftest_classifier: PASS (CONFIRMED / GENERIC_SATURATION / "
          "NOT_GOLDILOCKS(flat & ∩) / wrong-side-floor).")


def synthetic_goldilocks_test(seed=SEED):
    """End-to-end: quad-fit -> stats -> classify must distinguish a
    Goldilocks-bowl WORLD (BALANCE-specific extra fold) from a pure
    symmetric-saturation WORLD (everything bowls equally)."""
    rng = np.random.default_rng(seed)
    cs = DOSE_C
    SAT, GOLD, c0, base, noise = 0.30, 0.80, 0.15, 0.40, 0.004

    def curve(k_quad, vertex, rng_):
        return {c: np.array([base + k_quad * (c - vertex) ** 2
                             + rng_.normal(0, noise) for _ in range(40)])
                for c in cs}

    def stats_and_classify(t_by_c, rand_list, sch_list):
        k, b, a, cstar, argmin, _ = quad_stats(t_by_c)
        rks = [quad_stats(r)[0] for r in rand_list]
        sks = [quad_stats(s)[0] for s in sch_list]
        mr, sr = np.mean(rks), np.std(rks, ddof=1)
        z = (k - mr) / sr if sr > 0 else 0.0
        kdiff = k - mr
        return classify_goldilocks(k, (k - 0.03, k + 0.03), z,
                                   (kdiff - 0.03, kdiff + 0.03), cstar, argmin, rks, sks)

    # WORLD A: t has extra shifted bowl (SAT+GOLD, vertex c0>0); others SAT@0
    tA = curve(SAT + GOLD, c0, np.random.default_rng(seed + 1))
    randA = [curve(SAT, 0.0, np.random.default_rng(seed + 100 + i)) for i in range(N_RANDOM)]
    schA = [curve(SAT, 0.0, np.random.default_rng(seed + 200 + i)) for i in range(7)]
    vA = stats_and_classify(tA, randA, schA)
    assert vA == "GOLDILOCKS_CONFIRMED", f"world A -> {vA}"
    # WORLD B: pure symmetric saturation everywhere (t same as random)
    tB = curve(SAT, 0.0, np.random.default_rng(seed + 2))
    randB = [curve(SAT, 0.0, np.random.default_rng(seed + 300 + i)) for i in range(N_RANDOM)]
    schB = [curve(SAT, 0.0, np.random.default_rng(seed + 400 + i)) for i in range(7)]
    vB = stats_and_classify(tB, randB, schB)
    assert vB in ("GENERIC_SATURATION", "NOT_GOLDILOCKS"), f"world B -> {vB}"
    print(f"  synthetic_goldilocks_test: PASS (Goldilocks-world -> CONFIRMED; "
          f"saturation-world -> {vB}).")


def validate_harness():
    print("exp165c harness validation (no model load)")
    print("=" * 60)
    _assert_rule_matches_prereg()
    print(f"  rule asserted vs prereg: CURV_Z={CURV_Z}, SCHEMA_MAJORITY={SCHEMA_MAJORITY}")
    selftest_classifier()
    synthetic_goldilocks_test()
    h = hashlib.sha256("\n".join(FRESH_PROMPTS).encode()).hexdigest()[:16]
    assert h == PROMPT_CHECKSUM, f"prompt drift {h}"
    print(f"  prompts: exp166 set, checksum {h} OK; {len(DOSE_C)}-pt grid {DOSE_C}")
    print("HARNESS OK — safe to execute the model run.")


# ====================================================================
# Model run
# ====================================================================
def _run_curve(model, dir_by_layer, m):
    return {c: _measure_all_layer(model, dir_by_layer, m, c)[0] for c in DOSE_C}


def main():
    import torch
    from transformer_lens import HookedTransformer
    print("exp165c — GOLDILOCKS fold re-test (PREREG_exp165c.md)")
    validate_harness()
    rng = np.random.default_rng(SEED)
    all_words, est_words, test_words = build_word_lists()
    print(f"\nvocab: {len(all_words)}")
    model = HookedTransformer.from_pretrained("gpt2-medium", device=DEVICE)
    model.eval()

    print("\nbuilding all-layer directions + median norms...")
    bal, schema, randoms = _build_all_layer_dirs(model, all_words, est_words, test_words, rng)
    m = _median_norms(model)

    print(f"steering BALANCE over {len(DOSE_C)}-pt grid (all layers)...")
    bal_c = _run_curve(model, bal, m)
    print("steering 16 random all-layer directions...")
    rand_c = [_run_curve(model, rd, m) for rd in randoms]
    print("steering 7 schema all-layer directions...")
    sch_c = {sn: _run_curve(model, sd, m) for sn, sd in schema.items()}

    # --- statistics ---
    k_bal, b_bal, a_bal, cstar_bal, argmin_bal, means_bal = quad_stats(bal_c)
    spr = spearman(sorted(bal_c), list(means_bal))
    asym_bal = asymmetry(bal_c)
    ks_rand = [quad_stats(r)[0] for r in rand_c]
    ks_sch = {sn: quad_stats(s)[0] for sn, s in sch_c.items()}
    asym_rand = [asymmetry(r) for r in rand_c]
    mr_k, sr_k = float(np.mean(ks_rand)), float(np.std(ks_rand, ddof=1))
    z_k = (k_bal - mr_k) / sr_k if sr_k > 0 else 0.0
    mr_a, sr_a = float(np.mean(asym_rand)), float(np.std(asym_rand, ddof=1))
    z_a = (asym_bal - mr_a) / sr_a if sr_a > 0 else 0.0

    # bootstrap CI on k_bal and on (k_bal - mean_random_k)
    cs = sorted(DOSE_C); nP = len(bal_c[cs[0]])
    ks_b, kdiff_b = [], []
    rngb = np.random.default_rng(SEED + 7)
    for _ in range(N_BOOT):
        idx = rngb.choice(nP, nP, True)
        kb = np.polyfit(cs, [np.nanmean(bal_c[c][idx]) for c in cs], 2)[0]
        krs = [np.polyfit(cs, [np.nanmean(r[c][idx]) for c in cs], 2)[0] for r in rand_c]
        ks_b.append(kb); kdiff_b.append(kb - np.mean(krs))
    ci_k = (float(np.percentile(ks_b, 2.5)), float(np.percentile(ks_b, 97.5)))
    ci_kdiff = (float(np.percentile(kdiff_b, 2.5)), float(np.percentile(kdiff_b, 97.5)))

    print(f"\n{'='*72}\nGLOBAL DV dose-curve (mean attn entropy, all layers)\n{'='*72}")
    print("  c:    " + "  ".join(f"{c:+.2f}" for c in cs))
    print("  BAL:  " + "  ".join(f"{v:.3f}" for v in means_bal))
    print(f"\n  curvature k_BAL = {k_bal:+.4f}  CI[{ci_k[0]:+.4f},{ci_k[1]:+.4f}]  (∪ if >0)")
    print(f"  vertex c*_BAL   = {cstar_bal:+.3f}   empirical argmin = {argmin_bal:+.2f}  "
          f"(Goldilocks: floor on +balanced side)")
    print(f"  asymmetry A_BAL = {asym_bal:+.4f}  (imbalanced-minus-balanced outer; >0 = imbalance costs more)")
    print(f"  Spearman(c,ent) = {spr:+.2f}  (≈0 expected for a symmetric bowl)")
    print(f"\n  random curvature cloud: mean {mr_k:+.4f} sd {sr_k:.4f}  "
          f"min {min(ks_rand):+.4f} max {max(ks_rand):+.4f}")
    print(f"  z_k (BAL vs cloud)    = {z_k:+.2f}   (CONFIRM needs > {CURV_Z})")
    print(f"  k_BAL - mean_random   : CI[{ci_kdiff[0]:+.4f},{ci_kdiff[1]:+.4f}]")
    print(f"  z_asym (BAL vs cloud) = {z_a:+.2f}   (corroborating, ASYM_Z={ASYM_Z})")
    n_beat = sum(k_bal > sk for sk in ks_sch.values())
    print(f"  schema curvatures: " + ", ".join(f"{sn.split('-')[0][:4]} {sk:+.3f}"
                                                for sn, sk in ks_sch.items()))
    print(f"  BALANCE bowls more than {n_beat}/7 schemas (need >= {SCHEMA_MAJORITY})")

    # ☯ diagnostic (reported, not gated)
    base = float(means_bal[cs.index(0.0)])
    over = float(means_bal[cs.index(0.50)])
    imbal = float(means_bal[cs.index(-0.50)])
    print(f"\n  ☯ diagnostic: baseline(c=0)={base:.3f}  over-balanced(+0.5)={over:.3f}  "
          f"imbalanced(-0.5)={imbal:.3f}")
    print(f"    over-balancing raises entropy by {over-base:+.3f} vs baseline "
          f"({'toward imbalance signature' if over>base else 'no fold'})")

    verdict = classify_goldilocks(k_bal, ci_k, z_k, ci_kdiff, cstar_bal, argmin_bal,
                                  ks_rand, list(ks_sch.values()))
    print(f"\n{'='*72}\nVERDICT vs PREREG_exp165c.md:  >>> {verdict} <<<\n{'='*72}")
    print("  GOLDILOCKS_CONFIRMED = BALANCE-specific fold (bowl is a random-cloud outlier,")
    print("    floored on the balanced side) -> exp165b 'GENERIC' was a wrong-statistic false null")
    print("  GENERIC_SATURATION = everything bowls; the ∪ is not BALANCE-specific")
    print("  NOT_GOLDILOCKS = no reliable bowl")
    del model; gc.collect(); torch.mps.empty_cache()


if __name__ == "__main__":
    main()
