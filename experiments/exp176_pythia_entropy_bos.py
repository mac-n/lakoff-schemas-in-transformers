"""
exp176_pythia_entropy_bos.py — Pythia's BALANCE<->attention-entropy test,
rerun WITH a start token. Frozen rules: PREREG_exp176.md (commit 238a03d,
2026-10-02 17:27 IST). Written AFTER the freeze.

Everything is exp166's machinery, imported. `collect` and `stats` below
mirror exp166.run_model line for line; the only differences are the device
(CPU) and that the token cloud and the statistics are split so that one
collection can be read at more than one layer list.

Runs: (1) replication gate, Pythia without a start token;
      (2) positive control, GPT-2-medium untouched;
      (3) the test, Pythia with a start token.
"""
import json
import re
import sys

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)

from attn_entropy_lib import attn_entropy_per_query, partial_corr
from markedness_norm_protocol import build_word_lists, collect_residuals
from exp161_balance_entropy_prereg import build_layer_dirs, AXIS_WORDS
from exp164_depth_map_nonlinear_controls import covar_stacks
from exp166_balance_entropy_third_set import (FRESH_PROMPTS, DECISION_LO, CARRIER_MIN,
                                              N_BOOT, SEED, band_status, control_status,
                                              validate_harness)

REPL_TOL = 0.02
DEVICE = "cpu"
PRIMARY = (8, 12, 16)
SECONDARY_BAND = [11, 12]
TEST_LAYERS = [3, 5, 8, 11, 12, 16, 18, 20]
GATE_LAYERS = [5, 11, 12, 18]
GATE_C2Q = {5: -0.047, 11: -0.007, 12: -0.048, 18: 0.007}
GATE_CARRIER = {5: 0.96, 11: 0.92, 12: 0.91, 18: 0.89}
GPT2_LAYERS = [3, 8, 12, 16]
GPT2_C2Q = {3: -0.322, 8: -0.266, 12: -0.320, 16: -0.145}

PREREG = open(f"{ROOT}/PREREG_exp176.md").read()
for _name, _val in [("DECISION_LO", DECISION_LO), ("CARRIER_MIN", CARRIER_MIN),
                    ("N_BOOT", N_BOOT), ("REPL_TOL", REPL_TOL)]:
    _m = re.search(rf"\b{_name} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_val), f"{_name} drifted from prereg"
assert SEED == 166


def collect(model, layers, all_words, est_words, test_words):
    """Mirrors exp166.run_model up to the end of the token-cloud loop."""
    rhooks = [f"blocks.{L}.hook_resid_post" for L in layers]
    phooks = [f"blocks.{L}.attn.hook_pattern" for L in layers]
    residuals = collect_residuals(model, layers, all_words, log_every=0)
    dirs = {L: build_layer_dirs(residuals, L, all_words, est_words, test_words) for L in layers}
    acc = {L: dict(bal_pn=[], dnorm=[], ent=[], pos=[], norm=[], pid=[]) for L in layers}
    n_excl = 0
    for pid, prompt in enumerate(FRESH_PROMPTS):
        toks = model.to_tokens(prompt)
        strs = model.to_str_tokens(prompt)
        with torch.no_grad():
            _, c = model.run_with_cache(toks, names_filter=rhooks + phooks)
        for L in layers:
            resid = c[f"blocks.{L}.hook_resid_post"][0].float().cpu().numpy()
            pat = c[f"blocks.{L}.attn.hook_pattern"][0].float().cpu().numpy()
            ent = attn_entropy_per_query(pat)
            for q in range(1, resid.shape[0]):
                if np.isnan(ent[q]):
                    continue
                if strs[q].strip().lower().strip(".,!?'\"") in AXIS_WORDS:
                    if L == layers[0]:
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
    return dirs, acc


def stats(dirs, acc, layers, marks):
    """Mirrors exp166.run_model's statistics block (fresh rng, layers in order)."""
    rng = np.random.default_rng(SEED)
    out = {}
    print(f"  {'L':>3} {'carrier':>8} {'C2(lin)':>9} {'C2q(quad)':>10} {'C2q 95%CI':>17} {'n':>5}  cos(BAL,d_norm_ho)")
    for L in layers:
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
            sel = np.concatenate([idx_of[p] for p in rng.choice(upids, len(upids), replace=True)])
            _, q_sel = covar_stacks(pos[sel], nrm[sel], dnp[sel])
            boots.append(partial_corr(bal[sel], ent[sel], q_sel))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        carrier = dirs[L]["carrier_out"]
        out[L] = dict(c2=float(c2), c2q=float(c2q), ci=(float(lo), float(hi)),
                      carrier=float(carrier), n=len(ent), cos_bal_dnorm=float(dirs[L]["cos_bal_dnorm"]))
        print(f"  {L:>3} {carrier:>8.2f} {c2:>+9.3f} {c2q:>+10.3f} [{lo:+.3f},{hi:+.3f}] {len(ent):>5}"
              f"  {dirs[L]['cos_bal_dnorm']:+.3f}  {marks.get(L, '')}")
    return out


def primary_rule(st, with_carrier=True):
    """exp166's control_status at {8, 12, 16}; with_carrier=False drops only
    the carrier precondition (the registered second tier)."""
    if with_carrier:
        return control_status(st, PRIMARY)
    n_good = sum(1 for L in PRIMARY if st[L]["c2q"] < 0 and st[L]["ci"][1] < 0
                 and abs(st[L]["c2q"]) >= DECISION_LO)
    return "PASS" if n_good >= 2 else "FAIL"


def main():
    from transformer_lens import HookedTransformer
    print("exp176 — Pythia BALANCE<->entropy with a start token (prereg frozen at 238a03d)")
    validate_harness()
    all_words, est_words, test_words = build_word_lists()
    print(f"\nvocab: {len(all_words)} words (est {len(est_words)} / test {len(test_words)})")
    results = {}

    # ---- 1. replication gate: Pythia as exp166 ran it
    print("\n" + "=" * 72 + "\n1. REPLICATION GATE — pythia-410m WITHOUT a start token (as exp166)\n" + "=" * 72)
    model = HookedTransformer.from_pretrained("pythia-410m", device=DEVICE); model.eval()
    assert model.cfg.default_prepend_bos is False and model.to_tokens("the").shape[1] == 1
    layers_all = sorted(set(GATE_LAYERS) | set(TEST_LAYERS))
    dirs0, acc0 = collect(model, layers_all, all_words, est_words, test_words)
    gate = stats(dirs0, acc0, GATE_LAYERS, {11: "exp166 band", 12: "exp166 band"})
    ok = all(abs(gate[L]["c2q"] - GATE_C2Q[L]) <= REPL_TOL and
             abs(gate[L]["carrier"] - GATE_CARRIER[L]) <= REPL_TOL for L in GATE_LAYERS) \
        and gate[GATE_LAYERS[0]]["n"] == 473
    print(f"  exp166 had C2q {GATE_C2Q}, carriers {GATE_CARRIER}, n 473")
    print("REPLICATION GATE", "PASS" if ok else "FAIL")
    results["gate"] = gate
    if not ok:
        print("STOP: harness drift. Nothing below was computed.")
        json.dump(results, open(f"{ROOT}/exp176_results.json", "w"), indent=1, default=str)
        sys.exit(2)
    print("\n  (descriptive) the same no-start-token run read at the test layers:")
    nobos = stats(dirs0, acc0, TEST_LAYERS, {L: "primary" for L in PRIMARY})
    results["pythia_nobos_testlayers"] = nobos

    # ---- 3 (collected now, while the model is loaded): Pythia with a start token
    print("\n" + "=" * 72 + "\n3. THE TEST — pythia-410m WITH a start token\n" + "=" * 72)
    model.cfg.default_prepend_bos = True
    t = model.to_tokens("the")
    assert t.shape[1] == 2 and int(t[0, 0]) == model.tokenizer.bos_token_id, "start token not prepended"
    dirs1, acc1 = collect(model, TEST_LAYERS, all_words, est_words, test_words)
    test = stats(dirs1, acc1, TEST_LAYERS, {**{L: "primary" for L in PRIMARY}, 11: "secondary band"})
    results["pythia_bos"] = test
    del model

    # ---- 2. positive control: GPT-2-medium, untouched
    print("\n" + "=" * 72 + "\n2. POSITIVE CONTROL — gpt2-medium, untouched\n" + "=" * 72)
    model = HookedTransformer.from_pretrained("gpt2-medium", device=DEVICE); model.eval()
    assert model.cfg.default_prepend_bos is True and model.to_tokens("the").shape[1] == 2
    dirs2, acc2 = collect(model, GPT2_LAYERS, all_words, est_words, test_words)
    gpt = stats(dirs2, acc2, GPT2_LAYERS, {L: "control" for L in (8, 12, 16)})
    ctrl = control_status(gpt, (8, 12, 16))
    ctrl_match = all(abs(gpt[L]["c2q"] - GPT2_C2Q[L]) <= REPL_TOL for L in GPT2_LAYERS)
    print(f"  exp166 had C2q {GPT2_C2Q}")
    print(f"POSITIVE CONTROL {ctrl}; matches exp166 within {REPL_TOL}: {ctrl_match}")
    results["gpt2"] = gpt
    del model

    # ---- verdict
    print("\n" + "=" * 72 + "\nVERDICT vs PREREG_exp176.md\n" + "=" * 72)
    if ctrl != "PASS" or not ctrl_match:
        verdict = f"INVALID (GPT-2 control {ctrl}, match {ctrl_match})"
    else:
        full = primary_rule(test, with_carrier=True)
        if full == "PASS":
            verdict = "JOINS"
        elif full == "FAIL":
            verdict = "NULL"
        else:
            tier2 = primary_rule(test, with_carrier=False)
            verdict = "INVALID_CARRIER -> " + ("CARRIERLESS_JOINS" if tier2 == "PASS" else "CARRIERLESS_NULL")
    fmt = lambda st, Ls: ", ".join(f"L{L}:{st[L]['c2q']:+.3f}[{st[L]['ci'][0]:+.2f},{st[L]['ci'][1]:+.2f}](car{st[L]['carrier']:.2f})" for L in Ls)
    print(f"  PRIMARY {{8,12,16}} with a start token:    {fmt(test, PRIMARY)}")
    print(f"  same layers without a start token:       {fmt(nobos, PRIMARY)}")
    sec, _, sec_detail = band_status(test, SECONDARY_BAND)
    print(f"  SECONDARY band {{11,12}} with start token: {sec}  ({sec_detail})  {fmt(test, SECONDARY_BAND)}")
    print(f"\n  >>> {verdict} <<<")
    results.update({"verdict": verdict, "secondary": sec, "control": ctrl, "control_match": ctrl_match})
    json.dump(results, open(f"{ROOT}/exp176_results.json", "w"), indent=1, default=str)
    print("\nresults saved to exp176_results.json")


if __name__ == "__main__":
    main()
