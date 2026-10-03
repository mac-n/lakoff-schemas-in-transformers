"""
exp177_fragment_reading.py — is the BALANCE<->attention-entropy coupling a
word-fragment effect? Frozen rules: PREREG_exp177.md (commit fa6339c,
2026-10-02 17:53 IST). Written AFTER the freeze.

exp166's machinery is imported. The token-cloud loop mirrors
exp166.run_model; the additions are the token typing, the extra axes, and
the per-analysis masks / covariates.

Usage:  ./lakoff/bin/python3 exp177_fragment_reading.py gpt2-medium
        ./lakoff/bin/python3 exp177_fragment_reading.py Llama-3.2-1B
"""
import json
import re
import sys

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML
from attn_entropy_lib import attn_entropy_per_query, partial_corr
from markedness_norm_protocol import build_word_lists, collect_residuals, COMMON, RARE, corrf
from exp161_balance_entropy_prereg import build_layer_dirs, proj_out, AXIS_WORDS
from exp164_depth_map_nonlinear_controls import covar_stacks, zsc
from exp166_balance_entropy_third_set import (FRESH_PROMPTS, DECISION_LO, CARRIER_MIN,
                                              N_BOOT, SEED, band_status, control_status,
                                              validate_harness)

REPL_TOL = 0.02
MIN_POLE = 5
DEVICE = "cpu"
MODELS = {
    "gpt2-medium": dict(repo="gpt2-medium", layers=[3, 8, 12, 16], judge="control", judged=(8, 12, 16),
                        base={3: -0.322, 8: -0.266, 12: -0.320, 16: -0.145}),
    "Llama-3.2-1B": dict(repo="meta-llama/Llama-3.2-1B", layers=[5, 6, 7, 13], judge="band", judged=[5, 6, 7],
                         base={5: -0.139, 6: -0.168, 7: -0.202, 13: -0.268}),
}
PREREG = open(f"{ROOT}/PREREG_exp177.md").read()
for _name, _val in [("DECISION_LO", DECISION_LO), ("CARRIER_MIN", CARRIER_MIN), ("N_BOOT", N_BOOT),
                    ("REPL_TOL", REPL_TOL), ("MIN_POLE", MIN_POLE)]:
    _m = re.search(rf"\b{_name} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_val), f"{_name} drifted from prereg"
TYPES = ["WHOLE", "INITIAL_PIECE", "CONT_PIECE", "OTHER"]


def token_types(strs):
    """Type of each token at positions >= 1 (index 0 is the start token)."""
    def alpha(s):
        return s.strip().isalpha()
    out = [None]
    for q in range(1, len(strs)):
        s = strs[q]
        if not alpha(s):
            out.append("OTHER"); continue
        starts = s.startswith(" ") or q == 1
        nxt = strs[q + 1] if q + 1 < len(strs) else None
        cont_next = nxt is not None and alpha(nxt) and not nxt.startswith(" ")
        out.append(("INITIAL_PIECE" if cont_next else "WHOLE") if starts else "CONT_PIECE")
    return out


def strip_tools(residuals, L, all_words):
    """The aniso + freq strip exactly as exp161.build_layer_dirs does it."""
    aniso = np.stack([residuals[w][L] for w in all_words]).mean(axis=0)
    aniso = aniso / np.linalg.norm(aniso)
    mean_acts = lambda words: np.mean([residuals[w][L] for w in words], axis=0)
    freq = mean_acts(COMMON) - mean_acts(RARE)
    freq = freq / np.linalg.norm(freq)
    freq_orth = proj_out(freq, aniso)

    def strip(d):
        d = d - (d @ aniso) * aniso
        d = d - (d @ freq_orth) * freq_orth
        return d / np.linalg.norm(d)
    return strip, mean_acts


def main(tag):
    from transformer_lens import HookedTransformer
    cfg = MODELS[tag]
    LAYERS = cfg["layers"]
    print(f"exp177 — fragment reading of the BALANCE<->entropy coupling — {tag} (prereg frozen at fa6339c)")
    validate_harness()
    all_words, est_words, test_words = build_word_lists()
    model = HookedTransformer.from_pretrained(cfg["repo"], device=DEVICE); model.eval()
    assert model.cfg.default_prepend_bos is True and model.to_tokens("the").shape[1] == 2

    # ---- word states under both protocols
    res_bare = collect_residuals(model, LAYERS, all_words, log_every=0)
    ntok_bare = {w: int(model.to_tokens(w).shape[1] - 1) for w in all_words}
    hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    res_sp, ntok_sp = {}, {}
    for w in all_words:
        toks = model.to_tokens(" " + w)
        ntok_sp[w] = int(toks.shape[1] - 1)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=hook_names)
        res_sp[w] = {L: cache[f"blocks.{L}.hook_resid_post"][0, -1, :].float().cpu().numpy() for L in LAYERS}
    pairs = LAKOFF_SCHEMAS_MML["BALANCE"]
    pos = sorted(set(p[0] for p in pairs)); neg = sorted(set(p[1] for p in pairs))
    pos1 = [w for w in pos if ntok_sp[w] == 1]; neg1 = [w for w in neg if ntok_sp[w] == 1]
    print(f"\nBALANCE poles, one token as a bare word: {sum(ntok_bare[w] == 1 for w in pos)}/{len(pos)} positive,"
          f" {sum(ntok_bare[w] == 1 for w in neg)}/{len(neg)} negative")
    print(f"BALANCE poles, one token with a leading space: {len(pos1)}/{len(pos)} positive, {len(neg1)}/{len(neg)} negative")
    print(f"  fragment-free axis uses: + {pos1}\n                           - {neg1}")
    clean_ok = len(pos1) >= MIN_POLE and len(neg1) >= MIN_POLE

    dirs = {}
    for L in LAYERS:
        base = build_layer_dirs(res_bare, L, all_words, est_words, test_words)
        strip_b, _ = strip_tools(res_bare, L, all_words)
        unit = lambda w, R=res_bare: R[w][L] / np.linalg.norm(R[w][L])
        multi = np.array([1.0 if ntok_bare[w] > 1 else 0.0 for w in est_words])
        zm = (multi - multi.mean()) / multi.std()
        d_tok = np.sum([zm[k] * unit(w) for k, w in enumerate(est_words)], axis=0)
        d_tok = strip_b(d_tok / np.linalg.norm(d_tok))
        tok_carrier = corrf([float(unit(w) @ d_tok) for w in test_words],
                            [1.0 if ntok_bare[w] > 1 else 0.0 for w in test_words])
        bal = base["schema"]["BALANCE"]
        basis, _ = np.linalg.qr(np.stack([base["d_norm_ho"], d_tok]).T)
        bal_ptok = bal - basis @ (basis.T @ bal)
        bal_ptok = bal_ptok / np.linalg.norm(bal_ptok)
        sp = build_layer_dirs(res_sp, L, all_words, est_words, test_words)
        strip_s, mean_s = strip_tools(res_sp, L, all_words)
        if clean_ok:
            raw = mean_s(pos1) - mean_s(neg1)
            clean_full = strip_s(raw / np.linalg.norm(raw))
            bal_clean = proj_out(clean_full, sp["d_norm_ho"])
        else:
            clean_full = bal_clean = None
        dirs[L] = dict(base=base, sp=sp, d_tok=d_tok, tok_carrier=tok_carrier, bal_ptok=bal_ptok,
                       bal_clean=bal_clean,
                       cos_bal_dtok=float(bal @ d_tok),
                       cos_bal_clean=float(bal @ clean_full) if clean_ok else float("nan"),
                       cos_bal_c0=float(bal @ sp["schema"]["BALANCE"]))

    # ---- the token cloud (mirrors exp166.run_model's loop)
    rhooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    phooks = [f"blocks.{L}.attn.hook_pattern" for L in LAYERS]
    keys = ["bal_pn", "bal_ptok", "bal_c0", "bal_clean", "dnorm", "dnorm_sp", "dtok", "ent", "pos", "norm", "pid", "type"]
    acc = {L: {k: [] for k in keys} for L in LAYERS}
    n_excl = 0
    for pid, prompt in enumerate(FRESH_PROMPTS):
        toks = model.to_tokens(prompt)
        strs = model.to_str_tokens(prompt)
        types = token_types(strs)
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
                a, d = acc[L], dirs[L]
                a["bal_pn"].append(float(u @ d["base"]["bal_pn"]))
                a["bal_ptok"].append(float(u @ d["bal_ptok"]))
                a["bal_c0"].append(float(u @ d["sp"]["bal_pn"]))
                a["bal_clean"].append(float(u @ d["bal_clean"]) if clean_ok else 0.0)
                a["dnorm"].append(float(u @ d["base"]["d_norm_ho"]))
                a["dnorm_sp"].append(float(u @ d["sp"]["d_norm_ho"]))
                a["dtok"].append(float(u @ d["d_tok"]))
                a["ent"].append(ent[q]); a["pos"].append(q)
                a["norm"].append(nrm); a["pid"].append(pid); a["type"].append(types[q])
        del c
    print(f"  C5 axis-word exclusion: {n_excl} tokens dropped/layer-set")
    ty = np.array(acc[LAYERS[0]]["type"])
    print("  cloud tokens by type: " + ", ".join(f"{t} {int((ty == t).sum())}" for t in TYPES))

    def run(name, bal_key, dn_key, whole_only, extra_fn, carrier_key):
        rng = np.random.default_rng(SEED)
        out = {}
        for L in LAYERS:
            a = acc[L]
            typ = np.array(a["type"])
            m = (typ == "WHOLE") if whole_only else np.ones(len(typ), bool)
            pos_ = np.array(a["pos"], float)[m]; nrm = np.array(a["norm"], float)[m]
            ent = np.array(a["ent"], float)[m]; dnp = np.array(a[dn_key], float)[m]
            bal = np.array(a[bal_key], float)[m]; pid = np.array(a["pid"], int)[m]
            extra = extra_fn(a, m) if extra_fn else []
            _, quad = covar_stacks(pos_, nrm, dnp)
            c2q = partial_corr(bal, ent, quad + extra)
            upids = np.unique(pid)
            idx_of = {p: np.where(pid == p)[0] for p in upids}
            boots = []
            for _ in range(N_BOOT):
                sel = np.concatenate([idx_of[p] for p in rng.choice(upids, len(upids), replace=True)])
                _, q_sel = covar_stacks(pos_[sel], nrm[sel], dnp[sel])
                boots.append(partial_corr(bal[sel], ent[sel], q_sel + [e[sel] for e in extra]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out[L] = dict(c2q=float(c2q), ci=(float(lo), float(hi)),
                          carrier=float(dirs[L][carrier_key]["carrier_out"]), n=int(m.sum()))
        if cfg["judge"] == "control":
            status = control_status(out, tuple(cfg["judged"]))
        else:
            status = band_status(out, cfg["judged"])[0]
        line = "  ".join(f"L{L} {out[L]['c2q']:+.3f} [{out[L]['ci'][0]:+.2f},{out[L]['ci'][1]:+.2f}]" for L in LAYERS)
        print(f"  {name:38s} n {out[LAYERS[0]]['n']:>3}  {line}  carriers "
              + "/".join(f"{out[L]['carrier']:.2f}" for L in LAYERS) + f"  -> {status}")
        return out, status

    def extra_B(a, m):
        typ = np.array(a["type"])[m]
        ex = [zsc(np.array(a["dtok"], float)[m])]
        for t in ("INITIAL_PIECE", "CONT_PIECE", "OTHER"):
            dmy = (typ == t).astype(float)
            if dmy.std() > 0:
                ex.append(dmy)
        return ex

    print("\n" + "=" * 72 + f"\nANALYSES — C2q by layer (judged layers: {list(cfg['judged'])})\n" + "=" * 72)
    res = {}
    res["P0"], st0 = run("P0 exp166 exactly (gate)", "bal_pn", "dnorm", False, None, "base")
    gate = all(abs(res["P0"][L]["c2q"] - cfg["base"][L]) <= REPL_TOL for L in LAYERS)
    print(f"  exp166 had {cfg['base']}  ->  REPLICATION GATE {'PASS' if gate else 'FAIL'}")
    if not gate:
        print("STOP: harness drift. Nothing below was computed.")
        sys.exit(2)
    res["A"], stA = run("A  original axis, whole words only", "bal_pn", "dnorm", True, None, "base")
    res["B"], stB = run("B  tokenization direction controlled", "bal_ptok", "dnorm", False, extra_B, "base")
    res["C0"], stC0 = run("C0 leading-space protocol, all poles", "bal_c0", "dnorm_sp", False, None, "sp")
    if clean_ok:
        res["C"], stC = run("C  FRAGMENT-FREE AXIS, whole words", "bal_clean", "dnorm_sp", True, None, "sp")
    else:
        stC = "UNINFORMATIVE"
    surv = lambda s: s in ("PASS", "FIRES")

    print("\nDescriptive")
    for L in LAYERS:
        a = acc[L]; typ = np.array(a["type"]); ent = np.array(a["ent"]); bal = np.array(a["bal_pn"])
        by = "  ".join(f"{t}: entropy {ent[typ == t].mean():.3f}, BALANCE {bal[typ == t].mean():+.4f}"
                       for t in TYPES if (typ == t).sum() >= 5)
        d = dirs[L]
        print(f"  L{L:<2} cos(BALANCE, d_tok_ho) {d['cos_bal_dtok']:+.3f} (d_tok carrier {d['tok_carrier']:+.2f})"
              f" | cos(BALANCE, fragment-free BALANCE) {d['cos_bal_clean']:+.3f}"
              f" | cos(BALANCE, leading-space BALANCE) {d['cos_bal_c0']:+.3f}")
        print(f"       by token type — {by}")

    print("\n" + "=" * 72 + "\nVERDICT vs PREREG_exp177.md\n" + "=" * 72)
    if stC == "INVALID":
        verdict = "UNRESOLVED_CARRIER"
    elif stC == "UNINFORMATIVE":
        verdict = "UNINFORMATIVE (fragment-free pole too small)"
    elif surv(stC):
        verdict = "FRAGMENT_READING_REJECTED"
    elif surv(stC0):
        verdict = "FRAGMENT_READING_SUPPORTED"
    else:
        verdict = "UNRESOLVED"
    flag = ""
    if surv(stC) and not (surv(stA) and surv(stB)):
        flag = "  [INCONSISTENT: C survives but A or B does not]"
    print(f"  A {stA}   B {stB}   C0 {stC0}   C {stC}")
    print(f"\n  >>> {tag}: {verdict}{flag} <<<")
    json.dump({"model": tag, "analyses": {k: {str(L): v for L, v in r.items()} for k, r in res.items()},
               "status": {"P0": st0, "A": stA, "B": stB, "C0": stC0, "C": stC}, "verdict": verdict + flag,
               "poles_clean": {"pos": pos1, "neg": neg1}},
              open(f"{ROOT}/exp177_results_{tag}.json", "w"), indent=1)
    print(f"\nresults saved to exp177_results_{tag}.json")


if __name__ == "__main__":
    main(sys.argv[1])
