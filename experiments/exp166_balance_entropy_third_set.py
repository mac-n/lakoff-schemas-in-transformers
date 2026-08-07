"""exp166_balance_entropy_third_set.py — CONFIRMATORY (PREREG_exp166.md,
frozen before this script). Third, never-run prompt set. Tests whether
the BALANCE<->attention-entropy coupling replicates in the two bands
exp164 found post-hoc:
    Pythia L11-12   (committed gate)
    Llama  L5-7     (committed gate)
with GPT-2 {8,12,16} as a POSITIVE CONTROL (must reproduce, or the
prompt set is a dud -> INVALID).

Reuses, does NOT re-type:
  - frozen verified prompts ....... exp166_prompt_verify.CANDIDATE
  - per-layer direction machinery . exp161.build_layer_dirs
  - quadratic control stack (C2q) . exp164.covar_stacks / zsc / rankz
  - entropy + partial_corr ........ attn_entropy_lib
The gate read is C2q (the bands were defined under the quadratic stack).

Run: ./lakoff/bin/python3 exp166_balance_entropy_third_set.py
Validate harness only (no model load):
  ./lakoff/bin/python3 -c "import exp166_balance_entropy_third_set as e; e.validate_harness()"
"""
import os, gc, hashlib
import numpy as np
import torch
from huggingface_hub import get_token
os.environ["HF_TOKEN"] = get_token() or ""

from attn_entropy_lib import attn_entropy_per_query, partial_corr
from markedness_norm_protocol import build_word_lists, collect_residuals, SCHEMA_NAMES
from exp161_balance_entropy_prereg import build_layer_dirs, AXIS_WORDS
from exp164_depth_map_nonlinear_controls import covar_stacks, zsc
from exp166_prompt_verify import (CANDIDATE as FRESH_PROMPTS, check as verify_prompts,
                                  EXP160_PROMPTS, EXP161_PROMPTS)

# ---- frozen constants (match PREREG_exp166.md) ----
PROMPT_CHECKSUM = "4d54ff4297bd7e2c"        # sha256[:16] of "\n".join(prompts)
DECISION_LO = 0.10                           # magnitude floor (Niamh: keep 0.10)
CARRIER_MIN = 0.50
N_BOOT = 1000
SEED = 166

# model -> layers to run (gate band + non-gated context), band, role
MODELS_166 = {
    "pythia-410m":  dict(repo="pythia-410m",
                         layers=[5, 11, 12, 18], band=[11, 12], role="gate"),
    "Llama-3.2-1B": dict(repo="meta-llama/Llama-3.2-1B",
                         layers=[5, 6, 7, 13], band=[5, 6, 7], role="gate"),
    "gpt2-medium":  dict(repo="gpt2-medium",
                         layers=[3, 8, 12, 16], band=[8, 12, 16], role="control"),
}


# ====================================================================
# PURE classifiers (prereg decision rule) — selftest-able, no model
# ====================================================================
def contiguous_runs(ordered, good):
    """Maximal contiguous runs (length>=2) of `ordered` whose members are
    all in `good`. Returns list of (start, end)."""
    runs, cur = [], []
    for L in ordered:
        if L in good:
            cur.append(L)
        else:
            if len(cur) >= 2:
                runs.append((cur[0], cur[-1]))
            cur = []
    if len(cur) >= 2:
        runs.append((cur[0], cur[-1]))
    return runs


def band_status(stats, band, lo=DECISION_LO, carrier_min=CARRIER_MIN):
    """stats[L] = dict(c2q, ci=(lo,hi), carrier). Prereg gate (a-d):
      (a) every band layer C2q negative, (b) contiguous run >=2 with
      (c) |C2q|>=lo and CI excludes 0, (d) carrier>=min at band layers.
    Returns ('FIRES'|'NULL'|'INVALID', runs, detail)."""
    if any(stats[L]["carrier"] < carrier_min for L in band):
        return "INVALID", [], "carrier<min at a band layer"
    all_neg = all(stats[L]["c2q"] < 0 for L in band)
    good = {L for L in band
            if stats[L]["c2q"] < 0 and stats[L]["ci"][1] < 0
            and abs(stats[L]["c2q"]) >= lo}
    runs = contiguous_runs(band, good)
    if all_neg and runs:
        return "FIRES", runs, f"all-neg & contiguous good run(s) {runs}"
    if not all_neg:
        return "NULL", runs, "a band layer not negative"
    return "NULL", runs, "no contiguous >=2 run clears CI+floor"


def control_status(stats, layers=(8, 12, 16), lo=DECISION_LO, carrier_min=CARRIER_MIN):
    """GPT-2 positive control: >=2 of `layers` negative, CI excludes 0,
    |C2q|>=lo, carrier ok. Returns 'PASS'|'FAIL'|'INVALID'."""
    if any(stats[L]["carrier"] < carrier_min for L in layers):
        return "INVALID"
    n_good = sum(1 for L in layers
                 if stats[L]["c2q"] < 0 and stats[L]["ci"][1] < 0
                 and abs(stats[L]["c2q"]) >= lo)
    return "PASS" if n_good >= 2 else "FAIL"


def verdict166(py_status, ll_status, ctrl_status):
    """Combine the three per-model outcomes into the prereg decision."""
    if ctrl_status != "PASS":
        return f"INVALID_PROMPTSET (GPT-2 control={ctrl_status})"
    inval = [n for n, s in (("Pythia", py_status), ("Llama", ll_status))
             if s == "INVALID"]
    note = f"  [carrier-INVALID: {inval}]" if inval else ""
    if py_status == "FIRES" and ll_status == "FIRES":
        return "UNIVERSALITY_INVERSION_CONFIRMED" + note
    if py_status == "FIRES" or ll_status == "FIRES":
        which = "Pythia" if py_status == "FIRES" else "Llama"
        other = ll_status if which == "Pythia" else py_status
        return f"PARTIAL — {which} confirmed; other model = {other}" + note
    return "INVERSION_NOT_SUPPORTED" + note


def selftest_classifiers():
    def fake(c2q, ci_hi, carrier=0.9):
        return dict(c2q=c2q, ci=(ci_hi - 0.2, ci_hi), carrier=carrier)
    # --- band_status ---
    # clean fire: both band layers neg, CI<0, |c2q|>=0.10, contiguous
    s = {11: fake(-0.16, -0.05), 12: fake(-0.21, -0.12)}
    assert band_status(s, [11, 12])[0] == "FIRES"
    # one band layer positive -> NULL (gate (a))
    s = {11: fake(+0.05, +0.15), 12: fake(-0.21, -0.12)}
    assert band_status(s, [11, 12])[0] == "NULL"
    # negative but |c2q| below floor at one -> only a singleton good -> NULL
    s = {5: fake(-0.06, -0.01), 6: fake(-0.17, -0.05), 7: fake(-0.24, -0.16)}
    st, runs, _ = band_status(s, [5, 6, 7])
    assert st == "FIRES" and runs == [(6, 7)], (st, runs)  # 6-7 contiguous good
    # below floor breaks contiguity to singletons -> NULL
    s = {5: fake(-0.17, -0.05), 6: fake(-0.06, -0.01), 7: fake(-0.24, -0.16)}
    assert band_status(s, [5, 6, 7])[0] == "NULL"  # 5 and 7 not contiguous
    # CI includes 0 at one of a 2-run -> NULL
    s = {11: fake(-0.16, +0.02), 12: fake(-0.21, -0.12)}
    assert band_status(s, [11, 12])[0] == "NULL"
    # carrier fail -> INVALID (precondition; exp158 lesson)
    s = {11: fake(-0.16, -0.05, carrier=0.3), 12: fake(-0.21, -0.12)}
    assert band_status(s, [11, 12])[0] == "INVALID"
    # --- control_status ---
    c = {8: fake(-0.30, -0.20), 12: fake(-0.26, -0.16), 16: fake(-0.17, -0.06)}
    assert control_status(c) == "PASS"
    c = {8: fake(-0.30, -0.20), 12: fake(-0.05, +0.05), 16: fake(-0.04, +0.06)}
    assert control_status(c) == "FAIL"          # only 1 good -> dud prompt set
    c = {8: fake(-0.30, -0.20, carrier=0.2), 12: fake(-0.26, -0.16), 16: fake(-0.17, -0.06)}
    assert control_status(c) == "INVALID"
    # --- verdict166 ---
    assert verdict166("FIRES", "FIRES", "PASS") == "UNIVERSALITY_INVERSION_CONFIRMED"
    assert verdict166("FIRES", "NULL", "PASS").startswith("PARTIAL — Pythia")
    assert verdict166("NULL", "FIRES", "PASS").startswith("PARTIAL — Llama")
    assert verdict166("NULL", "NULL", "PASS") == "INVERSION_NOT_SUPPORTED"
    assert verdict166("FIRES", "FIRES", "FAIL").startswith("INVALID_PROMPTSET")
    assert "carrier-INVALID" in verdict166("INVALID", "FIRES", "PASS")
    print("  selftest_classifiers: PASS (gate (a) all-neg, contiguity, CI, "
          "floor, carrier-precondition, control, verdict).")


def synthetic_quad_stack_test(seed=SEED):
    """Discrimination test (prereg): the quadratic stack must (A) KILL a
    planted curved norm-leak that survives linear control, and (B) SPARE a
    genuine coupling independent of norm. Mirrors exp164's validation."""
    rng = np.random.default_rng(seed)
    n = 800
    pos = rng.integers(1, 25, n).astype(float)
    znorm = rng.standard_normal(n)
    norm = 50.0 + 8.0 * znorm
    dnorm = znorm + 0.2 * rng.standard_normal(n)          # held-out norm proxy
    qcurve = zsc(znorm ** 2)                              # curvature in norm
    # CASE A — pure curved leak: bal & ent both ride norm^2, no genuine link.
    balA = qcurve + 1.0 * rng.standard_normal(n)
    entA = -qcurve + 0.5 * rng.standard_normal(n)
    # CASE B — genuine: shared latent g, independent of norm; quad stack must keep it.
    g = rng.standard_normal(n)
    balB = g + 0.4 * qcurve + 1.0 * rng.standard_normal(n)
    entB = -g + 0.5 * rng.standard_normal(n)
    linA, quadA = covar_stacks(pos, norm, dnorm)
    c2A = partial_corr(balA, entA, linA)
    c2qA = partial_corr(balA, entA, quadA)
    linB, quadB = covar_stacks(pos, norm, dnorm)
    c2B = partial_corr(balB, entB, linB)
    c2qB = partial_corr(balB, entB, quadB)
    assert c2A < -0.15, f"leak should survive LINEAR control: c2A={c2A:+.3f}"
    assert abs(c2qA) < DECISION_LO, f"quad stack should KILL leak: c2qA={c2qA:+.3f}"
    assert c2qB < -0.15, f"quad stack should SPARE genuine: c2qB={c2qB:+.3f}"
    print(f"  synthetic_quad_stack_test: PASS  "
          f"(leak c2={c2A:+.2f}->c2q={c2qA:+.2f} killed; "
          f"genuine c2={c2B:+.2f}->c2q={c2qB:+.2f} spared).")


def check_prompts():
    h = hashlib.sha256("\n".join(FRESH_PROMPTS).encode()).hexdigest()[:16]
    assert h == PROMPT_CHECKSUM, f"prompt checksum drift: {h} != {PROMPT_CHECKSUM}"
    ok, rep = verify_prompts(FRESH_PROMPTS, prior_sets={"exp160": EXP160_PROMPTS,
                                                        "exp161": EXP161_PROMPTS})
    assert ok, f"prompt overlap check FAILED at runtime: {rep}"
    print(f"  prompts: 40 frozen, checksum {h} OK, overlap check OK.")


def validate_harness():
    """Everything that needs NO model. Run before any execution."""
    print("exp166 harness validation (no model load)")
    print("=" * 60)
    selftest_classifiers()
    synthetic_quad_stack_test()
    check_prompts()
    print("HARNESS OK — safe to execute the model run.")


# ====================================================================
# Model run
# ====================================================================
def run_model(tag, cfg, all_words, est_words, test_words):
    from transformer_lens import HookedTransformer
    rng = np.random.default_rng(SEED)
    print(f"\n{'=' * 72}\n{tag}  (role={cfg['role']}, band={cfg['band']})\n{'=' * 72}")
    model = HookedTransformer.from_pretrained(cfg["repo"], device="mps")
    model.eval()
    LAYERS = cfg["layers"]
    rhooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    phooks = [f"blocks.{L}.attn.hook_pattern" for L in LAYERS]

    residuals = collect_residuals(model, LAYERS, all_words, log_every=0)
    dirs = {L: build_layer_dirs(residuals, L, all_words, est_words, test_words)
            for L in LAYERS}

    acc = {L: dict(bal_pn=[], dnorm=[], ent=[], pos=[], norm=[], pid=[])
           for L in LAYERS}
    n_excl = 0
    for pid, prompt in enumerate(FRESH_PROMPTS):
        toks = model.to_tokens(prompt)
        strs = model.to_str_tokens(prompt)
        with torch.no_grad():
            _, c = model.run_with_cache(toks, names_filter=rhooks + phooks)
        for L in LAYERS:
            resid = c[f"blocks.{L}.hook_resid_post"][0].float().cpu().numpy()
            pat = c[f"blocks.{L}.attn.hook_pattern"][0].float().cpu().numpy()
            ent = attn_entropy_per_query(pat)
            for q in range(1, resid.shape[0]):
                if np.isnan(ent[q]):
                    continue
                if strs[q].strip().lower().strip(".,!?'\"") in AXIS_WORDS:
                    if L == LAYERS[0]:
                        n_excl += 1
                    continue                                   # C5
                r = resid[q]; nrm = np.linalg.norm(r); u = r / nrm
                a = acc[L]
                a["bal_pn"].append(float(u @ dirs[L]["bal_pn"]))
                a["dnorm"].append(float(u @ dirs[L]["d_norm_ho"]))
                a["ent"].append(ent[q]); a["pos"].append(q)
                a["norm"].append(nrm); a["pid"].append(pid)
        del c
    print(f"  C5 axis-word exclusion: {n_excl} tokens dropped/layer-set")

    out = {}
    print(f"  {'L':>3} {'carrier':>8} {'C2(lin)':>9} {'C2q(quad)':>10} "
          f"{'C2q 95%CI':>17} {'n':>5} {'band?':>6}")
    for L in LAYERS:
        a = acc[L]
        pos = np.array(a["pos"], float); nrm = np.array(a["norm"], float)
        ent = np.array(a["ent"], float); dnp = np.array(a["dnorm"], float)
        bal = np.array(a["bal_pn"], float); pid = np.array(a["pid"], int)
        lin, quad = covar_stacks(pos, nrm, dnp)
        c2 = partial_corr(bal, ent, lin)
        c2q = partial_corr(bal, ent, quad)
        boots = []
        upids = np.unique(pid)
        idx_of = {p: np.where(pid == p)[0] for p in upids}
        for _ in range(N_BOOT):
            sel = np.concatenate([idx_of[p] for p in
                                  rng.choice(upids, len(upids), replace=True)])
            _, q_sel = covar_stacks(pos[sel], nrm[sel], dnp[sel])
            boots.append(partial_corr(bal[sel], ent[sel], q_sel))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        carrier = dirs[L]["carrier_out"]
        inband = "BAND" if L in cfg["band"] else "ctx"
        cflag = " *carrier<0.5" if carrier < CARRIER_MIN else ""
        out[L] = dict(c2=c2, c2q=c2q, ci=(float(lo), float(hi)),
                      carrier=carrier, n=len(ent))
        print(f"  {L:>3} {carrier:>8.2f} {c2:>+9.3f} {c2q:>+10.3f} "
              f"[{lo:+.3f},{hi:+.3f}] {len(ent):>5} {inband:>6}{cflag}")

    del model, residuals, acc
    gc.collect(); torch.mps.empty_cache()
    return out


def main():
    print("exp166 — CONFIRMATORY BALANCE<->entropy, third prompt set "
          "(PREREG_exp166.md, frozen before this script)")
    print("Harness validation first (convention: selftests + synthetic + "
          "prompt integrity before any model):")
    validate_harness()

    all_words, est_words, test_words = build_word_lists()
    print(f"\nvocab: {len(all_words)} words (est {len(est_words)} / "
          f"test {len(test_words)}; BALANCE vocab held out of d_norm est)")

    results = {tag: run_model(tag, cfg, all_words, est_words, test_words)
               for tag, cfg in MODELS_166.items()}

    print("\n" + "=" * 72)
    print("VERDICT vs PREREG_exp166.md")
    print("=" * 72)
    py = results["pythia-410m"]
    ll = results["Llama-3.2-1B"]
    gp = results["gpt2-medium"]
    py_st, py_runs, py_d = band_status(py, MODELS_166["pythia-410m"]["band"])
    ll_st, ll_runs, ll_d = band_status(ll, MODELS_166["Llama-3.2-1B"]["band"])
    ctrl = control_status(gp, tuple(MODELS_166["gpt2-medium"]["band"]))

    def fmt(stats, band):
        return ", ".join(f"L{L}:{stats[L]['c2q']:+.2f}"
                         f"[{stats[L]['ci'][0]:+.2f},{stats[L]['ci'][1]:+.2f}]"
                         f"(car{stats[L]['carrier']:.2f})" for L in band)
    print(f"  Pythia L11-12  {py_st:<8} | {fmt(py, [11, 12])}  ({py_d})")
    print(f"  Llama  L5-7    {ll_st:<8} | {fmt(ll, [5, 6, 7])}  ({ll_d})")
    print(f"  GPT-2  control {ctrl:<8} | {fmt(gp, [8, 12, 16])}")
    print(f"\n  >>> {verdict166(py_st, ll_st, ctrl)} <<<")
    # non-gated context (no bet; handoff lead #6 etc.)
    print("\n  non-gated context (no bet):")
    print(f"    Pythia L5 {py[5]['c2q']:+.3f}  L18 {py[18]['c2q']:+.3f} "
          f"(lead #6: L18 isolated cell)")
    print(f"    Llama  L13 {ll[13]['c2q']:+.3f}")
    print(f"    GPT-2  L3 (shallow peak) {gp[3]['c2q']:+.3f}")


if __name__ == "__main__":
    main()
