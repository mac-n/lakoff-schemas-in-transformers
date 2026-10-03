"""
exp195_which_pushes_more.py — exp193's design in a second model and a second
layer. Frozen rules: PREREG_exp195.md (commit 9a7e37c, 3 Oct 2026 01:16 IST).
Written AFTER the freeze. Directions are built from word states inside
exp180's sentences at the steering layer (so this works for either model).

Usage:  ./lakoff/bin/python3 exp195_which_pushes_more.py gpt2-medium 4
        ./lakoff/bin/python3 exp195_which_pushes_more.py pythia-410m 8
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
from exp181_entropy_on_request import texts

N_CTRL = 30
SEED = 195
N_LAYERS = 24
MIN_POLE = 5
PREREG = open(f"{ROOT}/PREREG_exp195.md").read()
for _n, _v in [("N_CTRL", N_CTRL), ("SEED", SEED)]:
    _m = re.search(rf"\b{_n} = ([0-9]+)", PREREG)
    assert _m and int(_m.group(1)) == _v
MEASURES = ["straight", "write_cos", "end_norm", "kl", "dent"]


def main(name, site):
    run = e180.Runner2(name)
    model, tk, dev = run.model, run.tk, run.dev
    site_hook = f"blocks.{site}.hook_resid_post"
    hooks = [f"blocks.{L}.hook_resid_post" for L in range(site, N_LAYERS)]
    W = json.load(open(f"{ROOT}/exp175_stimuli.json"))["words"]
    one = lambda w: len(tk.encode(" " + w, add_special_tokens=False)) == 1
    lists = {"UP-DOWN": (W["UP"], W["DOWN"]), "VALENCE": (W["VALPOS"], W["VALNEG"])}
    for n, d in W["schemas"].items():
        lists[n] = (d["pos"], d["neg"])
    lists = {n: ([w for w in a if one(w)], [w for w in b if one(w)]) for n, (a, b) in lists.items()}
    nonbinding = [n for n, (a, b) in lists.items() if min(len(a), len(b)) < MIN_POLE]
    allw = sorted({w for a, b in lists.values() for w in a + b} | set(W["COMMON"]) | set(W["RARE"]))   # COMMON/RARE may be multi-token; state() takes the last token
    frames = e180.STIM["frames"]

    def state(w):
        acc = 0.0
        for fr in frames:
            ws = fr.split(" "); slot = ws.index("{w}"); ws[slot] = w
            g = [tk.encode(" " + x, add_special_tokens=False) for x in ws]
            ids = [run.bos] + [t for x in g for t in x]; p = sum(len(x) for x in g[: slot + 1])
            with torch.no_grad():
                _, cache = model.run_with_cache(torch.tensor([ids], device=dev), names_filter=[site_hook])
            acc = acc + cache[site_hook][0, p, :].float().cpu().numpy().astype(np.float64)
        return acc / len(frames)

    S = {w: state(w) for w in allw}
    R = np.stack([S[w] for w in allw]); ix = {w: i for i, w in enumerate(allw)}
    aniso = R.mean(0); aniso /= np.linalg.norm(aniso)
    ma = lambda ws: R[[ix[w] for w in ws]].mean(0)
    fq = ma(W["COMMON"]) - ma(W["RARE"]); fq /= np.linalg.norm(fq)
    assert np.isfinite(fq).all()
    fo = fq - (fq @ aniso) * aniso; fo /= np.linalg.norm(fo)

    def strip(d):
        d = d / np.linalg.norm(d); d = d - (d @ aniso) * aniso; d = d - (d @ fo) * fo
        return d / np.linalg.norm(d)

    dirs = {f"SCHEMA:{n}": strip(ma(a) - ma(b)) for n, (a, b) in lists.items()}
    bal = set(lists["BALANCE"][0]) | set(lists["BALANCE"][1])
    pool = [w for w in allw if w not in set(lists["UP-DOWN"][0]) | set(lists["UP-DOWN"][1]) | bal | set(W["COMMON"]) | set(W["RARE"])]
    rng = np.random.default_rng(SEED)
    for k in range(N_CTRL):
        w = rng.choice(pool); dirs[f"SINGLE:{k:02d}"] = strip(R[ix[w]] - R.mean(0))
        a, b = rng.choice(pool, 2, replace=False); dirs[f"PAIR:{k:02d}"] = strip(R[ix[a]] - R[ix[b]])
        ws = list(rng.choice(pool, 62, replace=False)); dirs[f"MUSH:{k:02d}"] = strip(ma(ws[:31]) - ma(ws[31:]))
        dirs[f"GAUSS:{k:02d}"] = strip(rng.normal(size=R.shape[1]))
    print(f"[{name} L{site}] {len(dirs)} directions; pool {len(pool)}; non-binding axes {nonbinding}", flush=True)
    T = texts()
    toks = [torch.tensor([[run.bos] + tk.encode(t, add_special_tokens=False)], device=dev) for t in T]
    clean_logp = []

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
                logits, cache = model.run_with_cache(x, names_filter=hooks)
            Rs = torch.stack([cache[h][0, 1:, :].float() for h in hooks]); Wr = Rs[1:] - Rs[:-1]
            net = Wr.sum(dim=0).norm(dim=1); path = Wr.norm(dim=2).sum(dim=0)
            cosw = torch.nn.functional.cosine_similarity(Wr[:-1], Wr[1:], dim=2).mean(dim=0)
            lp = torch.log_softmax(logits[0].float(), dim=-1); ent = float((-(lp.exp() * lp).sum(dim=-1)).mean())
            if vec is None:
                clean_logp.append(lp); kl = 0.0; dent = 0.0
            else:
                cl = clean_logp[i]; kl = float((cl.exp() * (cl - lp)).sum(dim=-1).mean()); dent = ent - float((-(cl.exp() * cl).sum(dim=-1)).mean())
            out[i] = [float((net / path).mean()), float(cosw.mean()), float(Rs[-1].norm(dim=1).mean()), kl, dent]
        return out

    path_ = f"{ROOT}/exp195_cells_{name}_L{site}.npz"
    cells = dict(np.load(path_)) if os.path.exists(path_) else {}
    base = measure(None)
    if "base" not in cells:
        cells["base"] = base
        tot, cnt = 0.0, 0
        for x in toks:
            with torch.no_grad():
                _, cache = model.run_with_cache(x, names_filter=[site_hook])
            n = cache[site_hook][0, 1:, :].float().norm(dim=1); tot += float(n.sum()); cnt += int(n.numel())
        cells["site_norm"] = np.array(tot / cnt); np.savez(path_, **cells)
    site_norm = float(cells["site_norm"])
    names = list(dirs)
    for j, n in enumerate(names):
        for sign in (+1, -1):
            key = f"{n}|{sign:+d}"
            if key in cells:
                continue
            cells[key] = measure(torch.tensor(sign * site_norm * dirs[n], dtype=torch.float32, device=dev)); np.savez(path_, **cells)
        if (j + 1) % 20 == 0:
            print(f"[{name} L{site}] {j + 1}/{len(names)} directions done", flush=True)
    b = cells["base"].mean(axis=0)
    rows = {}
    for n in names:
        p, m = cells[f"{n}|+1"].mean(axis=0), cells[f"{n}|-1"].mean(axis=0)
        rows[n] = dict(drop=float(b[0] - (p[0] + m[0]) / 2), kl=float((p[3] + m[3]) / 2), dent=float((p[4] + m[4]) / 2), plus=float(p[0]), minus=float(m[0]))
    fam = lambda f: [rows[n] for n in names if n.startswith(f + ":")]
    print(f"\n=== {name}, layer {site}: drop in later-layer agreement (unsteered {b[0]:.4f}) ===")
    for n in names:
        if n.startswith("SCHEMA:"):
            r = rows[n]; print(f"  {n:26s} drop {r['drop']:+.4f}  (+d {r['plus']:.4f}, -d {r['minus']:.4f})  KL {r['kl']:.3f}{'  (non-binding)' if n.split(':')[1] in nonbinding else ''}")
    for f in ("SINGLE", "PAIR", "MUSH", "GAUSS"):
        d = np.array([r["drop"] for r in fam(f)]); print(f"  {f:26s} drop median {np.median(d):+.4f} [5% {np.percentile(d, 5):+.4f}, 95% {np.percentile(d, 95):+.4f}]   KL median {np.median([r['kl'] for r in fam(f)]):.3f}")
    conc = np.array([r["drop"] for r in fam("SINGLE") + fam("PAIR")]); c5, c95 = np.percentile(conc, [5, 95])
    sch = {n: rows[n]["drop"] for n in names if n.startswith("SCHEMA:") and n.split(":")[1] not in nonbinding}
    up = rows["SCHEMA:UP-DOWN"]["drop"]; sch_mean = float(np.mean(list(sch.values())))
    q1 = "UP_STANDS_OUT" if (up == max(sch.values()) and up > c95) else "UP_NOT_SPECIAL"
    q2 = "MEANING_MATTERS" if sch_mean > c95 else ("SCHEMA_WEAKER" if sch_mean < c5 else "CONCENTRATION_SUFFICIENT")
    allr = [rows[n] for n in names]
    r_kl = float(np.corrcoef([r["drop"] for r in allr], [r["kl"] for r in allr])[0, 1])
    r_ent = float(np.corrcoef([r["drop"] for r in allr], [r["dent"] for r in allr])[0, 1])
    print(f"  Q1: UP-DOWN drop {up:+.4f}; largest {max(sch, key=sch.get)} ({max(sch.values()):+.4f}); concentrated controls [{c5:+.4f}, {c95:+.4f}] -> {q1}")
    print(f"  Q2: mean schema drop {sch_mean:+.4f} -> {q2}")
    print(f"  Q3: corr(drop, KL) {r_kl:+.2f}; corr(drop, entropy change) {r_ent:+.2f}")
    print(f"\n  >>> {name} L{site}: {q1} | {q2} <<<")
    json.dump({"model": name, "site": site, "q1": q1, "q2": q2, "r_drop_kl": r_kl, "r_drop_entropy": r_ent, "rows": rows,
               "nonbinding": nonbinding, "unsteered": b.tolist()}, open(f"{ROOT}/exp195_results_{name}_L{site}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
