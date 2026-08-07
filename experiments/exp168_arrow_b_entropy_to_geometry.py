"""exp168_arrow_b_entropy_to_geometry.py — ARROW B (PREREG_exp168.md,
frozen before run). Manipulate attention ENTROPY directly via temperature
(scale pre-softmax scores by 1/tau) and measure whether the BALANCE
GEOMETRY moves. Arrow A (inject geometry -> watch entropy) failed
specificity; Arrow B asks the reverse, testing causal DIRECTION.

Reuses: build_layer_dirs/proj_out (exp161), attn_entropy_per_query (lib),
ols_slope/spearman (exp165), frozen prompts (exp166).

Run:               ./lakoff/bin/python3 exp168_arrow_b_entropy_to_geometry.py
Validate harness:  ./lakoff/bin/python3 -c "import exp168_arrow_b_entropy_to_geometry as e; e.validate_harness()"
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

TAU = [0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0]
LAYER_PRIMARY = 3
LAYER_REPORT = 8
N_BOOT = 1000
SEED = 168
DEVICE = "mps"
SCHEMA_MAJORITY = 4
PROMPT_CHECKSUM = "4d54ff4297bd7e2c"


def _assert_rule_matches_prereg():
    with open(os.path.join(os.path.dirname(__file__) or ".", "PREREG_exp168.md")) as f:
        txt = f.read()
    assert "SCHEMA_MAJORITY = 4" in txt, "prereg rule text changed"
    assert SCHEMA_MAJORITY == 4, "code != prereg rule"


# ====================================================================
# PURE classifier
# ====================================================================
def classify_arrow_b(slope_bal, ci_slope, spearman_bal, schema_slopes):
    real = not (ci_slope[0] <= 0 <= ci_slope[1])
    negative = slope_bal < 0
    monotone = spearman_bal < 0
    if not (real and negative and monotone):
        return "ARROW_B_NULL"
    beats = sum(slope_bal < s for s in schema_slopes) >= SCHEMA_MAJORITY
    return "ARROW_B_CONFIRMED_SPECIFIC" if beats else "ARROW_B_GENERIC"


def selftest_classifier():
    sch = [-0.10, -0.08, -0.05, +0.02, -0.06, -0.09, -0.04]
    # CONFIRMED: BALANCE steepest-negative, beats >=4
    assert classify_arrow_b(-0.30, (-0.40, -0.20), -0.95, sch) == "ARROW_B_CONFIRMED_SPECIFIC"
    # GENERIC: real negative but not steeper than majority
    assert classify_arrow_b(-0.06, (-0.10, -0.02), -0.90, sch) == "ARROW_B_GENERIC"
    # NULL: CI includes 0
    assert classify_arrow_b(-0.02, (-0.07, +0.03), -0.3, sch) == "ARROW_B_NULL"
    # NULL: wrong sign (entropy up -> BAL up)
    assert classify_arrow_b(+0.20, (+0.10, +0.30), +0.8, sch) == "ARROW_B_NULL"
    print("  selftest_classifier: PASS (CONFIRMED_SPECIFIC / GENERIC / NULL(flat & wrong-sign)).")


def synthetic_arrow_b_test(seed=SEED):
    """quad/linear fit -> classify must distinguish: (A) BAL driven by
    entropy steeper than schemas -> CONFIRMED; (B) all equal -> GENERIC;
    (C) BAL flat -> NULL."""
    rng = np.random.default_rng(seed)
    ent = np.array([2.0, 1.7, 1.5, 1.2, 1.0, 0.8, 0.6])   # achieved entropy per tau
    def slope_of(b):   # proj = a - b*ent + noise ; returns OLS slope vs ent
        proj = 0.5 - b * ent + rng.normal(0, 0.002, len(ent))
        return ols_slope(ent, proj), spearman(ent, proj)
    # A: BALANCE steep, schemas shallow
    sB, spB = slope_of(0.30); sS = [slope_of(0.05)[0] for _ in range(7)]
    vA = classify_arrow_b(sB, (sB - 0.02, sB + 0.02), spB, sS)
    assert vA == "ARROW_B_CONFIRMED_SPECIFIC", f"A -> {vA}"
    # B: all equal
    sB2, spB2 = slope_of(0.10); sS2 = [slope_of(0.10)[0] for _ in range(7)]
    vB = classify_arrow_b(sB2, (sB2 - 0.02, sB2 + 0.02), spB2, sS2)
    assert vB in ("ARROW_B_GENERIC", "ARROW_B_CONFIRMED_SPECIFIC"), f"B -> {vB}"
    # C: BAL flat
    sB3, spB3 = slope_of(0.0); sS3 = [slope_of(0.05)[0] for _ in range(7)]
    vC = classify_arrow_b(sB3, (sB3 - 0.02, sB3 + 0.02), spB3, sS3)
    assert vC == "ARROW_B_NULL", f"C -> {vC}"
    print(f"  synthetic_arrow_b_test: PASS (driven->CONFIRMED, equal->{vB}, flat->NULL).")


def validate_harness():
    print("exp168 harness validation (no model load)")
    print("=" * 60)
    _assert_rule_matches_prereg()
    print(f"  rule asserted vs prereg: SCHEMA_MAJORITY={SCHEMA_MAJORITY}")
    selftest_classifier()
    synthetic_arrow_b_test()
    h = hashlib.sha256("\n".join(FRESH_PROMPTS).encode()).hexdigest()[:16]
    assert h == PROMPT_CHECKSUM, f"prompt drift {h}"
    print(f"  prompts: exp166 set, checksum {h} OK; tau grid {TAU}")
    print("HARNESS OK — safe to execute the model run.")


# ====================================================================
# Model run
# ====================================================================
def _measure_tau(model, L, dirs, tau):
    """Returns (balpn_per_prompt[40], {sn: per_prompt[40]}, ent_per_prompt[40])."""
    import torch
    score_hook = f"blocks.{L}.attn.hook_attn_scores"
    resid_hook = f"blocks.{L}.hook_resid_post"
    patt_hook = f"blocks.{L}.attn.hook_pattern"

    def temp(scores, hook):
        return scores / tau

    others = [sn for sn in SCHEMA_NAMES if sn != "BALANCE"]
    balpn, ent_pp = [], []
    proj_pp = {sn: [] for sn in others}
    with model.hooks(fwd_hooks=[(score_hook, temp)]):
        for prompt in FRESH_PROMPTS:
            toks = model.to_tokens(prompt)
            strs = model.to_str_tokens(prompt)
            with torch.no_grad():
                _, c = model.run_with_cache(toks, names_filter=[resid_hook, patt_hook])
            resid = c[resid_hook][0].float().cpu().numpy()
            pat = c[patt_hook][0].float().cpu().numpy()
            ent = attn_entropy_per_query(pat)
            bvals, evals = [], []
            ovals = {sn: [] for sn in others}
            for q in range(1, resid.shape[0]):
                if np.isnan(ent[q]):
                    continue
                if strs[q].strip().lower().strip(".,!?'\"") in AXIS_WORDS:
                    continue
                u = resid[q] / np.linalg.norm(resid[q])
                bvals.append(float(u @ dirs["bal_pn"]))
                evals.append(ent[q])
                for sn in others:
                    ovals[sn].append(float(u @ dirs["schema_perp"][sn]))
            balpn.append(np.mean(bvals) if bvals else np.nan)
            ent_pp.append(np.mean(evals) if evals else np.nan)
            for sn in others:
                proj_pp[sn].append(np.mean(ovals[sn]) if ovals[sn] else np.nan)
    return (np.array(balpn), {sn: np.array(v) for sn, v in proj_pp.items()},
            np.array(ent_pp))


def _slope_vs_entropy(x_ent, y_proj, rng):
    """x_ent, y_proj: dict tau -> per_prompt[40]. Slope of mean-proj on
    mean-entropy across tau, with prompt-cluster bootstrap CI."""
    taus = sorted(x_ent)
    xs = [np.nanmean(x_ent[t]) for t in taus]
    ys = [np.nanmean(y_proj[t]) for t in taus]
    slope = ols_slope(xs, ys)
    spr = spearman(xs, ys)
    n = len(x_ent[taus[0]])
    boots = []
    for _ in range(N_BOOT):
        idx = rng.choice(n, n, True)
        bx = [np.nanmean(x_ent[t][idx]) for t in taus]
        by = [np.nanmean(y_proj[t][idx]) for t in taus]
        boots.append(ols_slope(bx, by))
    return slope, spr, (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def run_layer(model, L, all_words, est_words, test_words):
    residuals = collect_residuals(model, [L], all_words, log_every=0)
    d = build_layer_dirs(residuals, L, all_words, est_words, test_words)
    dirs = {"bal_pn": d["bal_pn"],
            "schema_perp": {sn: proj_out(d["schema"][sn], d["d_norm_ho"])
                            for sn in SCHEMA_NAMES if sn != "BALANCE"}}
    bal_by, ent_by = {}, {}
    sch_by = {sn: {} for sn in dirs["schema_perp"]}
    for tau in TAU:
        b, o, e = _measure_tau(model, L, dirs, tau)
        bal_by[tau] = b; ent_by[tau] = e
        for sn in sch_by:
            sch_by[sn][tau] = o[sn]
    return bal_by, sch_by, ent_by


def main():
    import torch
    from transformer_lens import HookedTransformer
    print("exp168 — ARROW B: entropy -> BALANCE geometry? (PREREG_exp168.md)")
    validate_harness()
    all_words, est_words, test_words = build_word_lists()
    print(f"\nvocab: {len(all_words)}")
    model = HookedTransformer.from_pretrained("gpt2-medium", device=DEVICE)
    model.eval()

    for L, gate in [(LAYER_PRIMARY, True), (LAYER_REPORT, False)]:
        print(f"\n{'='*72}\nLAYER {L} {'(PRIMARY, gated)' if gate else '(reported)'}\n{'='*72}")
        bal_by, sch_by, ent_by = run_layer(model, L, all_words, est_words, test_words)

        # manipulation validity: achieved entropy vs tau
        ent_means = {t: float(np.nanmean(ent_by[t])) for t in TAU}
        print("  manipulation check — achieved entropy by tau:")
        print("    tau:  " + "  ".join(f"{t:.2f}" for t in TAU))
        print("    ent:  " + "  ".join(f"{ent_means[t]:.3f}" for t in TAU))
        ent_ordered = [ent_means[t] for t in TAU]
        valid = all(ent_ordered[i] <= ent_ordered[i + 1] + 1e-6 for i in range(len(TAU) - 1))
        print(f"    monotone(tau↑ -> entropy↑): {'YES' if valid else 'NO — INVALID'}")

        rng = np.random.default_rng(SEED + L)
        s_bal, spr_bal, ci_bal = _slope_vs_entropy(ent_by, bal_by, rng)
        sch_slopes = {sn: _slope_vs_entropy(ent_by, sch_by[sn],
                                            np.random.default_rng(SEED + 100))[0]
                      for sn in sch_by}

        # BALANCE projection by tau (for the curve)
        print("\n  BALANCE projection by tau (sharpen→flatten):")
        print("    tau:  " + "  ".join(f"{t:.2f}" for t in TAU))
        print("    BAL:  " + "  ".join(f"{np.nanmean(bal_by[t]):+.3f}" for t in TAU))
        print(f"\n  slope(BAL-proj vs ACHIEVED entropy) = {s_bal:+.4f} "
              f"CI[{ci_bal[0]:+.4f},{ci_bal[1]:+.4f}]  Spearman={spr_bal:+.2f}")
        print(f"  (predicted NEGATIVE: lower entropy -> higher BALANCE proj)")
        print("  schema slopes: " + ", ".join(f"{sn.split('-')[0][:4]} {sv:+.3f}"
                                               for sn, sv in sch_slopes.items()))
        n_beat = sum(s_bal < sv for sv in sch_slopes.values())
        print(f"  BALANCE more negative than {n_beat}/7 schemas (need >= {SCHEMA_MAJORITY})")

        if gate:
            if not valid:
                print("\n  >>> INVALID (manipulation did not move entropy monotonically) <<<")
            else:
                verdict = classify_arrow_b(s_bal, ci_bal, spr_bal, list(sch_slopes.values()))
                print(f"\n{'='*72}\nVERDICT vs PREREG_exp168.md:  >>> {verdict} <<<\n{'='*72}")
                print("  CONFIRMED_SPECIFIC = entropy causally shapes BALANCE geometry (arrow B, specific)")
                print("  GENERIC = entropy shapes geometry broadly (directional asymmetry vs A, not specific)")
                print("  NULL = entropy doesn't shape geometry -> correlation is common-cause")
    del model; gc.collect(); torch.mps.empty_cache()


if __name__ == "__main__":
    main()
