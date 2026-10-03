"""
exp193_which_pushes.py — which pushes make the later layers disagree?
Frozen rules: PREREG_exp193.md (commit ced5b93, 3 Oct 2026 00:23 IST).
Written AFTER the freeze. Steering and agreement measures are exp188's; the
output KL and entropy are computed in the same pass. Cells saved as they
finish; resumes.
"""
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)
from exp181_entropy_on_request import texts

SITE = 4
N_CTRL = 30
SEED = 193
N_LAYERS = 24
PREREG = open(f"{ROOT}/PREREG_exp193.md").read()
for _n, _v in [("SITE", SITE), ("N_CTRL", N_CTRL), ("SEED", SEED)]:
    _m = re.search(rf"\b{_n} = ([0-9]+)", PREREG)
    assert _m and int(_m.group(1)) == _v, f"{_n} drifted from prereg"
MEASURES = ["straight", "write_cos", "end_norm", "kl", "dent"]


def main():
    from transformer_lens import HookedTransformer
    print("exp193 — which pushes make the later layers disagree? (prereg frozen at ced5b93)")
    dev = "mps"
    model = HookedTransformer.from_pretrained("pythia-410m", device=dev); model.eval()
    tk = model.tokenizer; BOS = int(tk.bos_token_id)
    hooks = [f"blocks.{L}.hook_resid_post" for L in range(SITE, N_LAYERS)]
    site_hook = f"blocks.{SITE}.hook_resid_post"

    z = np.load(f"{ROOT}/exp175_cache_wordstates.npz", allow_pickle=True)
    words = list(z["words"]); R = z["R"].astype(np.float64).mean(axis=1)[:, SITE, :]
    ix = {w: i for i, w in enumerate(words)}
    W = json.load(open(f"{ROOT}/exp175_stimuli.json"))["words"]
    aniso = R.mean(axis=0); aniso /= np.linalg.norm(aniso)
    ma = lambda ws: R[[ix[w] for w in ws]].mean(axis=0)
    fq = ma(W["COMMON"]) - ma(W["RARE"]); fq /= np.linalg.norm(fq)
    fo = fq - (fq @ aniso) * aniso; fo /= np.linalg.norm(fo)

    def strip(d):
        d = d / np.linalg.norm(d)
        d = d - (d @ aniso) * aniso; d = d - (d @ fo) * fo
        return d / np.linalg.norm(d)

    dirs = {"SCHEMA:UP-DOWN": strip(ma(W["UP"]) - ma(W["DOWN"])), "SCHEMA:VALENCE": strip(ma(W["VALPOS"]) - ma(W["VALNEG"]))}
    for n, d in W["schemas"].items():
        dirs[f"SCHEMA:{n}"] = strip(ma(d["pos"]) - ma(d["neg"]))
    bal = set(W["schemas"]["BALANCE"]["pos"]) | set(W["schemas"]["BALANCE"]["neg"])
    pool = [w for w in words if w not in set(W["UP"]) | set(W["DOWN"]) | bal | set(W["COMMON"]) | set(W["RARE"])]
    rng = np.random.default_rng(SEED)
    for k in range(N_CTRL):
        w = rng.choice(pool); dirs[f"SINGLE:{k:02d}"] = strip(R[ix[w]] - R.mean(axis=0))
        a, b = rng.choice(pool, 2, replace=False); dirs[f"PAIR:{k:02d}"] = strip(R[ix[a]] - R[ix[b]])
        ws = list(rng.choice(pool, 62, replace=False)); dirs[f"MUSH:{k:02d}"] = strip(ma(ws[:31]) - ma(ws[31:]))
        g = rng.normal(size=R.shape[1]); dirs[f"GAUSS:{k:02d}"] = strip(g)
    print(f"  {len(dirs)} directions; pool {len(pool)} words", flush=True)

    T = texts()
    toks = [torch.tensor([[BOS] + tk.encode(t, add_special_tokens=False)], device=dev) for t in T]
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
            Rs = torch.stack([cache[h][0, 1:, :].float() for h in hooks])
            Wr = Rs[1:] - Rs[:-1]
            net = Wr.sum(dim=0).norm(dim=1); path = Wr.norm(dim=2).sum(dim=0)
            cosw = torch.nn.functional.cosine_similarity(Wr[:-1], Wr[1:], dim=2).mean(dim=0)
            lp = torch.log_softmax(logits[0].float(), dim=-1)
            ent = float((-(lp.exp() * lp).sum(dim=-1)).mean())
            if vec is None:
                clean_logp.append(lp); kl = 0.0; dent = 0.0; base_ent = ent
            else:
                cl = clean_logp[i]
                kl = float((cl.exp() * (cl - lp)).sum(dim=-1).mean())
                dent = ent - float((-(cl.exp() * cl).sum(dim=-1)).mean())
            out[i] = [float((net / path).mean()), float(cosw.mean()), float(Rs[-1].norm(dim=1).mean()), kl, dent]
        return out

    path_ = f"{ROOT}/exp193_cells.npz"
    cells = dict(np.load(path_)) if os.path.exists(path_) else {}
    base = measure(None)                                   # always recomputed: fills clean_logp
    if "base" not in cells:
        cells["base"] = base
        tot, cnt = 0.0, 0
        for x in toks:
            with torch.no_grad():
                _, cache = model.run_with_cache(x, names_filter=[site_hook])
            n = cache[site_hook][0, 1:, :].float().norm(dim=1); tot += float(n.sum()); cnt += int(n.numel())
        cells["site_norm"] = np.array(tot / cnt); np.savez(path_, **cells)
    site_norm = float(cells["site_norm"])
    print(f"  site norm {site_norm:.2f}; unsteered straight {base[:, 0].mean():.4f}", flush=True)
    names = list(dirs)
    for j, n in enumerate(names):
        for sign in (+1, -1):
            key = f"{n}|{sign:+d}"
            if key in cells:
                continue
            cells[key] = measure(torch.tensor(sign * site_norm * dirs[n], dtype=torch.float32, device=dev))
            np.savez(path_, **cells)
        if (j + 1) % 20 == 0:
            print(f"  {j + 1}/{len(names)} directions done", flush=True)

    # ---- analysis
    b = cells["base"].mean(axis=0)
    rows = {}
    for n in names:
        p, m = cells[f"{n}|+1"].mean(axis=0), cells[f"{n}|-1"].mean(axis=0)
        rows[n] = dict(drop=float(b[0] - (p[0] + m[0]) / 2), asym=float(p[0] - m[0]), kl=float((p[3] + m[3]) / 2),
                       dent=float((p[4] + m[4]) / 2), cos_drop=float(b[1] - (p[1] + m[1]) / 2),
                       norm_change=float((p[2] + m[2]) / 2 - b[2]), plus=float(p[0]), minus=float(m[0]))
    fam = lambda f: [rows[n] for n in names if n.startswith(f + ":")]
    print("\n" + "=" * 76 + "\nRESULTS — drop = unsteered agreement minus sign-averaged steered agreement (unsteered "
          f"{b[0]:.4f})\n" + "=" * 76)
    print(f"  {'direction':24s} {'drop':>8} {'+d':>8} {'-d':>8} {'asym':>8} {'KL':>8} {'d entropy':>10} {'norm chg':>9}")
    for n in names:
        if n.startswith("SCHEMA:"):
            r = rows[n]
            print(f"  {n:24s} {r['drop']:+8.4f} {r['plus']:8.4f} {r['minus']:8.4f} {r['asym']:+8.4f} {r['kl']:8.4f} {r['dent']:+10.4f} {r['norm_change']:+9.3f}")
    for f in ("SINGLE", "PAIR", "MUSH", "GAUSS"):
        d = np.array([r["drop"] for r in fam(f)]); k = np.array([r["kl"] for r in fam(f)])
        print(f"  {f:24s} drop median {np.median(d):+.4f} [5% {np.percentile(d, 5):+.4f}, 95% {np.percentile(d, 95):+.4f}]   KL median {np.median(k):.4f}")
    conc = np.array([r["drop"] for r in fam("SINGLE") + fam("PAIR")])
    c5, c95 = np.percentile(conc, [5, 95])
    sch = {n: rows[n]["drop"] for n in names if n.startswith("SCHEMA:")}
    up = sch["SCHEMA:UP-DOWN"]; sch_mean = float(np.mean(list(sch.values())))
    q1 = "UP_STANDS_OUT" if (up == max(sch.values()) and up > c95) else "UP_NOT_SPECIAL"
    q2 = "MEANING_MATTERS" if sch_mean > c95 else ("SCHEMA_WEAKER" if sch_mean < c5 else "CONCENTRATION_SUFFICIENT")
    allr = [rows[n] for n in names]
    r_kl = float(np.corrcoef([r["drop"] for r in allr], [r["kl"] for r in allr])[0, 1])
    r_ent = float(np.corrcoef([r["drop"] for r in allr], [r["dent"] for r in allr])[0, 1])
    print(f"\n  Q1: UP-DOWN drop {up:+.4f}; largest schema drop {max(sch, key=sch.get)} ({max(sch.values()):+.4f}); "
          f"concentrated controls 5–95%: [{c5:+.4f}, {c95:+.4f}] -> {q1}")
    print(f"  Q2: mean schema drop {sch_mean:+.4f} vs concentrated controls -> {q2}")
    print(f"  Q3: corr(drop, output KL) over all {len(allr)} directions = {r_kl:+.2f}; corr(drop, entropy change) = {r_ent:+.2f}"
          f"{'   (drop reads as a proxy for output change)' if r_kl > 0.8 else ''}")
    print(f"\n  >>> {q1} | {q2} <<<")
    json.dump({"q1": q1, "q2": q2, "r_drop_kl": r_kl, "r_drop_entropy": r_ent, "rows": rows,
               "conc_p5": float(c5), "conc_p95": float(c95), "unsteered": b.tolist()},
              open(f"{ROOT}/exp193_results.json", "w"), indent=1)
    print("results saved to exp193_results.json")


if __name__ == "__main__":
    main()
