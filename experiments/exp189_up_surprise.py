"""
exp189_up_surprise.py — does the UP readout track the model's surprise?
Frozen rules: PREREG_exp189.md (commit d872b04, 2026-10-02 20:42 IST).
Written AFTER the freeze.

Usage:  ./lakoff/bin/python3 exp189_up_surprise.py pythia-410m
        ./lakoff/bin/python3 exp189_up_surprise.py gpt2-medium
"""
import json
import re
import sys

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)
import exp180_paired_retry as e180
import exp185_bundle_vs_default as e185
from exp177_fragment_reading import token_types
from exp181_entropy_on_request import texts

N_NULL = 200
SEED_NULL = 189
N_BOOT = 1000
LAYERS_MAJ = 3
LAYERS = [4, 8, 12, 16, 20]
COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]
PREREG = open(f"{ROOT}/PREREG_exp189.md").read()
for _n, _v in [("N_NULL", N_NULL), ("SEED_NULL", SEED_NULL), ("N_BOOT", N_BOOT), ("LAYERS_MAJ", LAYERS_MAJ)]:
    _m = re.search(rf"\b{_n} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_v), f"{_n} drifted from prereg"


def resid_on(v, X):
    b, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ b


def pcorr(x, y, X):
    a, b = resid_on(x, X), resid_on(y, X)
    a = a - a.mean(); b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else 0.0


def main(name):
    from wordfreq import zipf_frequency
    run = e180.Runner2(name)
    model, tk, dev = run.model, run.tk, run.dev
    hook = {L: f"blocks.{L}.hook_resid_post" for L in LAYERS}
    frames = e180.STIM["frames"]; P = e180.STIM["pairs"]
    pool = sorted({w for ps in list(P.values()) + [e185.BUNDLE_OLD, e185.BUNDLE_NEW, e185.NOLINK_OLD, e185.NOLINK_NEW]
                   for p in ps for w in p})
    assert all(len(tk.encode(" " + w, add_special_tokens=False)) == 1 for w in pool)

    # ---- clean word states inside exp180's sentences
    def state(w):
        acc = {L: 0.0 for L in LAYERS}
        for fr in frames:
            ws = fr.split(" "); slot = ws.index("{w}"); ws[slot] = w
            g = [tk.encode(" " + x, add_special_tokens=False) for x in ws]
            ids = [run.bos] + [t for x in g for t in x]
            p = sum(len(x) for x in g[: slot + 1])
            with torch.no_grad():
                _, cache = model.run_with_cache(torch.tensor([ids], device=dev), names_filter=list(hook.values()))
            for L in LAYERS:
                acc[L] = acc[L] + cache[hook[L]][0, p, :].float().cpu().numpy().astype(np.float64)
        return {L: acc[L] / len(frames) for L in LAYERS}

    S = {w: state(w) for w in pool + COMMON + RARE}
    up_w = [p[0] for p in P["vertical"]]; dn_w = [p[1] for p in P["vertical"]]
    vp_w = [p[0] for p in P["valence"]]; vn_w = [p[1] for p in P["valence"]]
    rng = np.random.default_rng(SEED_NULL)
    draws = [list(rng.choice(pool, 62, replace=False)) for _ in range(N_NULL)]
    axes = {}
    for L in LAYERS:
        allv = np.stack([S[w][L] for w in S]); aniso = allv.mean(axis=0); aniso /= np.linalg.norm(aniso)
        ma = lambda ws: np.mean([S[w][L] for w in ws], axis=0)
        fq = ma(COMMON) - ma(RARE); fq /= np.linalg.norm(fq)
        fo = fq - (fq @ aniso) * aniso; fo /= np.linalg.norm(fo)

        def clean(a, b):
            d = ma(a) - ma(b); d /= np.linalg.norm(d)
            d = d - (d @ aniso) * aniso; d = d - (d @ fo) * fo
            return d / np.linalg.norm(d)
        axes[L] = dict(UP=clean(up_w, dn_w), VAL=clean(vp_w, vn_w),
                       NULL=np.stack([clean(d[:31], d[31:]) for d in draws]))
    print(f"  clean axes built from {len(S)} word states; cos(UP, VALENCE) by layer: "
          + "  ".join(f"L{L} {float(axes[L]['UP'] @ axes[L]['VAL']):+.3f}" for L in LAYERS))

    # ---- token cloud on ordinary sentences
    T = texts()
    skip = set(up_w) | set(dn_w)
    U = {L: [] for L in LAYERS}; norm = {L: [] for L in LAYERS}
    surp, pent, pos, zipf, nonword, ttype, pid = [], [], [], [], [], [], []
    for i, text in enumerate(T):
        ids = [run.bos] + tk.encode(text, add_special_tokens=False)
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            logits, cache = model.run_with_cache(x, names_filter=list(hook.values()))
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        ent = (-(logp.exp() * logp).sum(dim=-1)).cpu().numpy()
        lp = logp.cpu().numpy()
        strs = [tk.decode([t]) for t in ids]
        types = token_types(strs)
        for q in range(1, len(ids)):
            w = strs[q].strip().lower().strip(".,!?'\"")
            if w in skip:
                continue
            surp.append(-lp[q - 1, ids[q]]); pent.append(ent[q]); pos.append(q); pid.append(i); ttype.append(types[q])
            isw = w.isalpha()
            zipf.append(zipf_frequency(w, "en") if isw else 0.0); nonword.append(0.0 if isw else 1.0)
            for L in LAYERS:
                r = cache[hook[L]][0, q, :].float().cpu().numpy().astype(np.float64)
                n = np.linalg.norm(r); U[L].append(r / n); norm[L].append(n)
    surp, pent, pos, zipf, nonword, pid = map(np.array, (surp, pent, pos, zipf, nonword, pid))
    ttype = np.array(ttype)
    print(f"  tokens analysed: {len(surp)} from {len(T)} sentences; mean surprisal {surp.mean():.2f} nats; "
          f"corr(surprisal, predictive entropy) {np.corrcoef(surp, pent)[0, 1]:+.2f}")
    dummies = [(ttype == t).astype(float) for t in ("INITIAL_PIECE", "CONT_PIECE", "OTHER")]
    dummies = [d for d in dummies if d.std() > 0]
    rng_b = np.random.default_rng(1890)
    upids = np.unique(pid); idx_of = {p: np.where(pid == p)[0] for p in upids}
    out = {}
    print("\n" + "=" * 76 + f"\nRESULTS — {name}: partial correlation of the UP readout with each target\n" + "=" * 76)
    for target_name, target in (("surprisal", surp), ("predictive entropy", pent)):
        print(f"  [{target_name}{' — PRIMARY' if target_name == 'surprisal' else ' — secondary'}]")
        res = {}
        for L in LAYERS:
            Um = np.stack(U[L]); nr = np.array(norm[L])
            up = Um @ axes[L]["UP"]; val = Um @ axes[L]["VAL"]
            X = np.column_stack([np.ones(len(up)), val, pos, zipf, nonword, nr] + dummies)
            r = pcorr(up, target, X)
            boots = []
            for _ in range(N_BOOT):
                sel = np.concatenate([idx_of[p] for p in rng_b.choice(upids, len(upids), replace=True)])
                boots.append(pcorr(up[sel], target[sel], X[sel]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            tr = resid_on(target, X); tr = tr - tr.mean()
            NP = Um @ axes[L]["NULL"].T                                      # [tokens, N_NULL]
            NR = NP - X @ np.linalg.lstsq(X, NP, rcond=None)[0]; NR = NR - NR.mean(axis=0)
            null = (NR.T @ tr) / (np.linalg.norm(NR, axis=0) * np.linalg.norm(tr))
            p5, p95 = np.percentile(null, [5, 95])
            Xv = np.column_stack([np.ones(len(up)), pos, zipf, nonword, nr] + dummies)
            res[L] = dict(r=r, ci=[float(lo), float(hi)], null_p5=float(p5), null_p95=float(p95),
                          null_median=float(np.median(null)), pct=float((null < r).mean() * 100),
                          above=bool(r > 0 and lo > 0 and r > p95), below=bool(r < 0 and hi < 0 and r < p5),
                          val_r=pcorr(val, target, Xv))
            x = res[L]
            print(f"    L{L:<2} UP {r:+.3f} [{lo:+.3f},{hi:+.3f}] | random-word axes: median {x['null_median']:+.3f} "
                  f"[5% {p5:+.3f}, 95% {p95:+.3f}]; UP at the {x['pct']:.0f}th percentile | VALENCE readout {x['val_r']:+.3f}"
                  f"{'   ABOVE RANDOM' if x['above'] else ('   BELOW RANDOM' if x['below'] else '')}")
        na = sum(res[L]["above"] for L in LAYERS); nb = sum(res[L]["below"] for L in LAYERS)
        v = "UP_TRACKS_SURPRISE" if na >= LAYERS_MAJ else ("UP_TRACKS_SURENESS" if nb >= LAYERS_MAJ else "NOT_BEYOND_RANDOM")
        out[target_name] = {"layers": {str(L): res[L] for L in LAYERS}, "label": v}
        print(f"    -> {v}  (above random at {na}/5 layers, below at {nb}/5)")
    print(f"\n  >>> {name}: {out['surprisal']['label']} <<<")
    json.dump({"model": name, "n_tokens": int(len(surp)), "results": out, "verdict": out["surprisal"]["label"]},
              open(f"{ROOT}/exp189_results_{name}.json", "w"), indent=1)
    print(f"results saved to exp189_results_{name}.json")


if __name__ == "__main__":
    main(sys.argv[1])
