"""
exp164_depth_map_nonlinear_controls.py — diagnostic depth map of the
BALANCE<->attention-entropy coupling at EVERY layer of all three models,
under both the linear (exp161 C2) and nonlinear/quadratic (C2q) control
stacks.

PRE-REGISTRATION: PREREG_exp164.md (frozen before this file). Mapping
run — no V-gate; committed predictions P1 (Pythia L12 collapses under
the quadratic stack) and P2 (GPT-2's coupling is robust to it). Any new
positive is a LEAD for its own fresh-prompt prereg, never a result.

Mechanistic motivation for the quadratic stack (written in the prereg
before running): entropy is thermostatted by query magnitude, a
QUADRATIC readout of residual direction; linear partials cannot strip a
curved norm->entropy channel.

Conventions: shared machinery imported (attn_entropy_lib,
markedness_norm_protocol, and exp161's frozen prompts + direction
builder — no transcription, exp161 catch-#4 lesson); import-safe;
pure classifiers selftested first, including the carrier-fail case.
"""

import gc
import os

import numpy as np
import torch
from huggingface_hub import get_token

os.environ["HF_TOKEN"] = get_token() or ""

from markedness_norm_protocol import build_word_lists, collect_residuals
from attn_entropy_lib import attn_entropy_per_query, partial_corr
from exp161_balance_entropy_prereg import (FRESH_PROMPTS, AXIS_WORDS,
                                           build_layer_dirs,
                                           _selfcheck_prompts)

N_BOOT = 1000
SEED = 164
CARRIER_MIN = 0.50

ALL_MODELS = {
    "pythia-410m":  dict(repo="pythia-410m",             n_layers=24),
    "gpt2-medium":  dict(repo="gpt2-medium",             n_layers=24),
    "Llama-3.2-1B": dict(repo="meta-llama/Llama-3.2-1B", n_layers=16),
}


def zsc(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std()


def rankz(v):
    v = np.asarray(v, float)
    return zsc(np.argsort(np.argsort(v)).astype(float))


def covar_stacks(pos, norm, dnorm):
    """(linear, quadratic) covariate lists. z-scored before squaring."""
    zp, zn, zd = zsc(pos), zsc(norm), zsc(dnorm)
    lin = [zp, zn, zd]
    quad = lin + [zn * zn, zd * zd, zn * zd, rankz(norm)]
    return lin, quad


# ---------------- pure classifiers (prereg interpretation rules) -------

def classify_p1(c2q_L12, carrier_L12, lo=0.10, carrier_min=CARRIER_MIN):
    """Pythia L12 under the quadratic stack."""
    if carrier_L12 < carrier_min:
        return "FLAGGED"          # carrier failed; no claim either way
    return "D1_leakage" if abs(c2q_L12) < lo else "D2_survives"


def classify_p2(rows, hi=0.15, peak=0.30):
    """GPT-2 robustness. rows: {L: c2q} for L in (3, 8, 12, 16)."""
    ok_decision = all(rows[L] <= -hi for L in (8, 12, 16))
    ok_peak = rows[3] <= -peak
    if ok_decision and ok_peak:
        return "P2_HIT"
    if not ok_decision:
        return "REQUALIFY_V1a"    # exp165 waits
    return "P2_PARTIAL"           # decision layers fine, peak attenuated


def llama_leads(layers, c2q, ci, carrier, carrier_min=CARRIER_MIN):
    """Contiguous runs of >=2 layers: C2q negative, CI excludes 0,
    carrier ok. Returns list of (start_L, end_L) runs."""
    good = [L for L in layers
            if c2q[L] < 0 and ci[L][1] < 0 and carrier[L] >= carrier_min]
    runs, cur = [], []
    for L in layers:
        if L in good:
            cur.append(L)
        else:
            if len(cur) >= 2:
                runs.append((cur[0], cur[-1]))
            cur = []
    if len(cur) >= 2:
        runs.append((cur[0], cur[-1]))
    return runs


def selftest_classifiers():
    # P1: collapse / survive / carrier-fail (precondition-fails case)
    assert classify_p1(-0.03, 0.9) == "D1_leakage"
    assert classify_p1(-0.21, 0.9) == "D2_survives"
    assert classify_p1(-0.03, 0.3) == "FLAGGED"
    assert classify_p1(-0.21, 0.3) == "FLAGGED"
    # P2: robust / requalify / partial
    assert classify_p2({3: -0.40, 8: -0.30, 12: -0.25, 16: -0.16}) == "P2_HIT"
    assert classify_p2({3: -0.40, 8: -0.10, 12: -0.25, 16: -0.16}) == "REQUALIFY_V1a"
    assert classify_p2({3: -0.20, 8: -0.30, 12: -0.25, 16: -0.16}) == "P2_PARTIAL"
    # Llama runs: detection, gap-splitting, carrier veto, singletons ignored
    Ls = list(range(6))
    c2q = {0: -.2, 1: -.2, 2: +.1, 3: -.2, 4: -.2, 5: -.2}
    ci = {L: (-.3, -.1) for L in Ls}; ci[2] = (0.0, 0.2)
    car = {L: 0.8 for L in Ls}
    assert llama_leads(Ls, c2q, ci, car) == [(0, 1), (3, 5)]
    car[4] = 0.3
    assert llama_leads(Ls, c2q, ci, car) == [(0, 1)]   # run broken by carrier
    ci[0] = (-.3, +.1)
    assert llama_leads(Ls, c2q, ci, car) == []         # singleton L1 ignored
    print("selftest_classifiers: all branches fire correctly "
          "(incl. carrier-fail FLAGGED).")


# ---------------- run ----------------

def run_model(tag, cfg, all_words, est_words, test_words):
    from transformer_lens import HookedTransformer
    rng = np.random.default_rng(SEED)
    print(f"\n{'='*72}\n{tag}\n{'='*72}")
    model = HookedTransformer.from_pretrained(cfg["repo"], device="mps")
    model.eval()
    LAYERS = list(range(cfg["n_layers"]))
    rhooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    phooks = [f"blocks.{L}.attn.hook_pattern" for L in LAYERS]

    residuals = collect_residuals(model, LAYERS, all_words, log_every=0)
    dirs = {L: build_layer_dirs(residuals, L, all_words, est_words,
                                test_words) for L in LAYERS}

    acc = {L: dict(bal_pn=[], dnorm=[], ent=[], pos=[], norm=[], pid=[])
           for L in LAYERS}
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
                    continue                                       # C5
                r = resid[q]; nrm = np.linalg.norm(r); u = r / nrm
                a = acc[L]
                a["bal_pn"].append(float(u @ dirs[L]["bal_pn"]))
                a["dnorm"].append(float(u @ dirs[L]["d_norm_ho"]))
                a["ent"].append(ent[q]); a["pos"].append(q)
                a["norm"].append(nrm); a["pid"].append(pid)
        del c

    out = {}
    print(f"  {'L':>3} {'carrier':>8} {'C2(lin)':>9} {'C2q(quad)':>10} "
          f"{'C2q 95%CI':>17} {'n':>5}")
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
                                  rng.choice(upids, len(upids),
                                             replace=True)])
            _, q_sel = covar_stacks(pos[sel], nrm[sel], dnp[sel])
            boots.append(partial_corr(bal[sel], ent[sel], q_sel))
        lo, hi = np.percentile(boots, [2.5, 97.5])

        carrier = dirs[L]["carrier_out"]
        flag = " *carrier<0.5" if carrier < CARRIER_MIN else ""
        out[L] = dict(c2=c2, c2q=c2q, ci=(float(lo), float(hi)),
                      carrier=carrier, n=len(ent))
        print(f"  {L:>3} {carrier:>8.2f} {c2:>+9.3f} {c2q:>+10.3f} "
              f"[{lo:+.3f},{hi:+.3f}] {len(ent):>5}{flag}")

    del model, residuals, acc
    gc.collect(); torch.mps.empty_cache()
    return out


def main():
    print("exp164 — depth map + nonlinear control stack "
          "(PREREG_exp164.md, frozen before this script)")
    print("Selftests first...")
    selftest_classifiers()
    _selfcheck_prompts()
    print("Prompt set imported from exp161 module (frozen, integrity OK).")

    all_words, est_words, test_words = build_word_lists()
    results = {tag: run_model(tag, cfg, all_words, est_words, test_words)
               for tag, cfg in ALL_MODELS.items()}

    print("\n" + "=" * 72)
    print("PREREG OUTCOMES (PREREG_exp164.md)")
    print("=" * 72)
    py = results["pythia-410m"]
    p1 = classify_p1(py[12]["c2q"], py[12]["carrier"])
    print(f"  P1 (Pythia L12 under quad stack): C2q = {py[12]['c2q']:+.3f} "
          f"CI [{py[12]['ci'][0]:+.3f},{py[12]['ci'][1]:+.3f}] -> {p1}")

    g = results["gpt2-medium"]
    p2 = classify_p2({L: g[L]["c2q"] for L in (3, 8, 12, 16)})
    print(f"  P2 (GPT-2 robustness): "
          + ", ".join(f"L{L}:{g[L]['c2q']:+.2f}" for L in (3, 8, 12, 16))
          + f" -> {p2}")

    ll = results["Llama-3.2-1B"]
    Ls = sorted(ll)
    runs = llama_leads(Ls, {L: ll[L]["c2q"] for L in Ls},
                       {L: ll[L]["ci"] for L in Ls},
                       {L: ll[L]["carrier"] for L in Ls})
    if runs:
        print(f"  Q2 (Llama): LEAD — contiguous runs {runs} "
              f"(goes to fresh-prompt prereg, NOT a result)")
    else:
        print("  Q2 (Llama): no contiguous >=2-layer run survives "
              "(criterion in prereg); singletons logged only.")
    print("\n  D1 = L12 was leakage -> quad stack becomes standard; "
          "exp161 re-quoted under it")
    print("  D2 = L12 survives -> Pythia-only fresh-prompt prereg")
    print("  REQUALIFY_V1a -> exp165 waits")


if __name__ == "__main__":
    main()
