"""
exp190_agreement_followup.py — follow-up to exp188's lean. Frozen rules:
PREREG_exp190.md (commit 1d42a2a, 2026-10-02 21:06 IST). Written AFTER the
freeze. Steering, measures and statistic are exp188's; this adds a second
site, a second model, three strengths and 100 random-word directions.

Usage:  ./lakoff/bin/python3 exp190_agreement_followup.py pythia-410m 4
        (model in {pythia-410m, gpt2-medium}; site in {4, 8})
        ./lakoff/bin/python3 exp190_agreement_followup.py summary
Each (model, site) saves every cell as it finishes and resumes.
"""
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)
import exp180_paired_retry as e180
import exp185_bundle_vs_default as e185
from exp181_entropy_on_request import texts

N_NULL = 100
SEED_NULL = 190
ALPHA = 0.05
N_PERM = 10000
N_LAYERS = 24
CS = [0.5, 1.0, 1.5]
CELLS = [("pythia-410m", 4), ("pythia-410m", 8), ("gpt2-medium", 4), ("gpt2-medium", 8)]
COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]
MEASURES = ["straight", "write_cos", "end_norm"]
PREREG = open(f"{ROOT}/PREREG_exp190.md").read()
for _n, _v in [("N_NULL", N_NULL), ("SEED_NULL", SEED_NULL), ("ALPHA", ALPHA), ("N_PERM", N_PERM)]:
    _m = re.search(rf"\b{_n} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_v), f"{_n} drifted from prereg"


def run_cell(name, site):
    run = e180.Runner2(name)
    model, tk, dev = run.model, run.tk, run.dev
    site_hook = f"blocks.{site}.hook_resid_post"
    hooks = [f"blocks.{L}.hook_resid_post" for L in range(site, N_LAYERS)]
    frames = e180.STIM["frames"]; P = e180.STIM["pairs"]
    pool = sorted({w for ps in list(P.values()) + [e185.BUNDLE_OLD, e185.BUNDLE_NEW, e185.NOLINK_OLD, e185.NOLINK_NEW]
                   for p in ps for w in p})

    def state(w):
        acc = 0.0
        for fr in frames:
            ws = fr.split(" "); slot = ws.index("{w}"); ws[slot] = w
            g = [tk.encode(" " + x, add_special_tokens=False) for x in ws]
            ids = [run.bos] + [t for x in g for t in x]
            p = sum(len(x) for x in g[: slot + 1])
            with torch.no_grad():
                _, cache = model.run_with_cache(torch.tensor([ids], device=dev), names_filter=[site_hook])
            acc = acc + cache[site_hook][0, p, :].float().cpu().numpy().astype(np.float64)
        return acc / len(frames)

    S = {w: state(w) for w in pool + COMMON + RARE}
    allv = np.stack(list(S.values())); aniso = allv.mean(axis=0); aniso /= np.linalg.norm(aniso)
    ma = lambda ws: np.mean([S[w] for w in ws], axis=0)
    fq = ma(COMMON) - ma(RARE); fq /= np.linalg.norm(fq)
    fo = fq - (fq @ aniso) * aniso; fo /= np.linalg.norm(fo)

    def clean(a, b):
        d = ma(a) - ma(b); d /= np.linalg.norm(d)
        d = d - (d @ aniso) * aniso; d = d - (d @ fo) * fo
        return d / np.linalg.norm(d)

    dirs = {"UP": clean([p[0] for p in P["vertical"]], [p[1] for p in P["vertical"]]),
            "VALENCE": clean([p[0] for p in P["valence"]], [p[1] for p in P["valence"]])}
    rng = np.random.default_rng(SEED_NULL)
    for k in range(N_NULL):
        ws = list(rng.choice(pool, 62, replace=False))
        dirs[f"NULL_{k:03d}"] = clean(ws[:31], ws[31:])
    T = texts()
    toks = [torch.tensor([[run.bos] + tk.encode(t, add_special_tokens=False)], device=dev) for t in T]

    def measure(vec):
        out = np.zeros((len(T), len(MEASURES)))
        fwd = []
        if vec is not None:
            def hook(resid, hook):
                resid[:, 1:, :] = resid[:, 1:, :] + vec
                return resid
            fwd = [(site_hook, hook)]
        for i, x in enumerate(toks):
            with torch.no_grad(), model.hooks(fwd_hooks=fwd):
                _, cache = model.run_with_cache(x, names_filter=hooks)
            Rs = torch.stack([cache[h][0, 1:, :].float() for h in hooks])
            Wr = Rs[1:] - Rs[:-1]
            net = Wr.sum(dim=0).norm(dim=1); path = Wr.norm(dim=2).sum(dim=0)
            cosw = torch.nn.functional.cosine_similarity(Wr[:-1], Wr[1:], dim=2).mean(dim=0)
            out[i] = [float((net / path).mean()), float(cosw.mean()), float(Rs[-1].norm(dim=1).mean())]
        return out

    path_ = f"{ROOT}/exp190_cells_{name}_L{site}.npz"
    cells = dict(np.load(path_)) if os.path.exists(path_) else {}
    if "base" not in cells:
        cells["base"] = measure(None)
        tot, cnt = 0.0, 0
        for x in toks:
            with torch.no_grad():
                _, cache = model.run_with_cache(x, names_filter=[site_hook])
            n = cache[site_hook][0, 1:, :].float().norm(dim=1)
            tot += float(n.sum()); cnt += int(n.numel())
        cells["site_norm"] = np.array(tot / cnt)
        np.savez(path_, **cells)
    site_norm = float(cells["site_norm"])
    print(f"[{name} L{site}] site norm {site_norm:.2f}; unsteered "
          + "  ".join(f"{m} {v:.4f}" for m, v in zip(MEASURES, cells["base"].mean(axis=0))), flush=True)
    todo = [(n, c) for n in ("UP", "VALENCE") for c in CS] + [(f"NULL_{k:03d}", 1.0) for k in range(N_NULL)]
    for j, (n, c) in enumerate(todo):
        for sign in (+1, -1):
            key = f"{n}|{c}|{sign:+d}"
            if key in cells:
                continue
            cells[key] = measure(torch.tensor(sign * c * site_norm * dirs[n], dtype=torch.float32, device=dev))
            np.savez(path_, **cells)
        if (j + 1) % 25 == 0:
            print(f"[{name} L{site}] {j + 1}/{len(todo)} directions x strengths done", flush=True)
    print(f"[{name} L{site}] cell complete", flush=True)


def summary():
    rng = np.random.default_rng(1900)
    out = {}
    held = 0
    print("\n" + "=" * 76 + "\nexp190 RESULTS — A = measure with +d minus measure with −d, mean over 120 sentences\n" + "=" * 76)
    for name, site in CELLS:
        cells = dict(np.load(f"{ROOT}/exp190_cells_{name}_L{site}.npz"))
        res = {}
        print(f"\n--- {name}, steering at layer {site} ---")
        for mi, m in enumerate(MEASURES):
            A = lambda n, c: cells[f"{n}|{c}|+1"][:, mi] - cells[f"{n}|{c}|-1"][:, mi]
            null = np.array([A(f"NULL_{k:03d}", 1.0).mean() for k in range(N_NULL)])
            up = A("UP", 1.0)
            r = dict(up=float(up.mean()), p=e180.signflip(up, rng), pos=int((up > 0).sum()),
                     null_p5=float(np.percentile(null, 5)), null_p95=float(np.percentile(null, 95)),
                     pct=float((null < up.mean()).mean() * 100), val=float(A("VALENCE", 1.0).mean()),
                     val_pct=float((null < A("VALENCE", 1.0).mean()).mean() * 100),
                     dose_up=[float(A("UP", c).mean()) for c in CS], dose_val=[float(A("VALENCE", c).mean()) for c in CS])
            r["holds"] = bool(r["up"] > 0 and r["p"] < ALPHA and r["up"] > r["null_p95"])
            res[m] = r
            print(f"  {m:9s} A(UP) {r['up']:+.5f} (p {r['p']:.4f}, {r['pos']}/120) | null 5% {r['null_p5']:+.5f}, 95% {r['null_p95']:+.5f};"
                  f" UP at the {r['pct']:.0f}th percentile{'  HOLDS' if r['holds'] else ''} | VALENCE {r['val']:+.5f} ({r['val_pct']:.0f}th)"
                  f" | UP by strength {CS}: " + " ".join(f"{x:+.4f}" for x in r["dose_up"]))
        held += res["straight"]["holds"]
        out[f"{name}_L{site}"] = res
    v = "LEAD_HOLDS" if held >= 3 else ("LEAD_FAILS" if held <= 1 else "MIXED")
    print(f"\n  cells holding on STRAIGHT at strength 1.0: {held} of 4")
    print(f"  >>> {v} <<<")
    json.dump({"verdict": v, "held": int(held), "cells": out}, open(f"{ROOT}/exp190_results.json", "w"), indent=1)
    print("results saved to exp190_results.json")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run_cell(sys.argv[1], int(sys.argv[2]))
