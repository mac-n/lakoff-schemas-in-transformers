"""
exp188_steer_agreement.py — does steering UP at one early layer make the
later layers agree? Frozen rules: PREREG_exp188.md (commit d9d6704,
2026-10-02 20:38 IST). Written AFTER the freeze.

One steering site (SITE), every non-start position. Agreement is measured on
the WRITES of the blocks after the site (output minus input of each block).
Results are written per direction; the run resumes.
"""
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)
import exp180_paired_retry as e180          # sign-flip test
from exp181_entropy_on_request import texts

SITE = 4
N_NULL = 40
SEED_NULL = 188
ALPHA = 0.05
N_PERM = 10000
N_LAYERS = 24
CS = [0.5, 1.0]
PREREG = open(f"{ROOT}/PREREG_exp188.md").read()
for _n, _v in [("SITE", SITE), ("N_NULL", N_NULL), ("SEED_NULL", SEED_NULL), ("ALPHA", ALPHA), ("N_PERM", N_PERM)]:
    _m = re.search(rf"\b{_n} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_v), f"{_n} drifted from prereg"
MEASURES = ["straight", "write_cos", "end_norm", "path", "net"]


def main():
    from transformer_lens import HookedTransformer
    print("exp188 — steering UP and layer agreement (prereg frozen at d9d6704)")
    dev = "mps"
    model = HookedTransformer.from_pretrained("pythia-410m", device=dev); model.eval()
    tk = model.tokenizer; BOS = int(tk.bos_token_id)
    hooks = [f"blocks.{L}.hook_resid_post" for L in range(SITE, N_LAYERS)]
    site_hook = f"blocks.{SITE}.hook_resid_post"

    # ---- clean directions at the site, from exp175's cached in-sentence word states
    z = np.load(f"{ROOT}/exp175_cache_wordstates.npz", allow_pickle=True)
    words = list(z["words"]); R = z["R"].astype(np.float64).mean(axis=1)[:, SITE, :]      # [words, d]
    ix = {w: i for i, w in enumerate(words)}
    W = json.load(open(f"{ROOT}/exp175_stimuli.json"))["words"]
    aniso = R.mean(axis=0); aniso /= np.linalg.norm(aniso)
    ma = lambda ws: R[[ix[w] for w in ws]].mean(axis=0)
    fq = ma(W["COMMON"]) - ma(W["RARE"]); fq /= np.linalg.norm(fq)
    fo = fq - (fq @ aniso) * aniso; fo /= np.linalg.norm(fo)

    def clean(a, b):
        d = ma(a) - ma(b); d /= np.linalg.norm(d)
        d = d - (d @ aniso) * aniso; d = d - (d @ fo) * fo
        return d / np.linalg.norm(d)

    dirs = {"UP": clean(W["UP"], W["DOWN"]), "VALENCE": clean(W["VALPOS"], W["VALNEG"])}
    skip = set(W["UP"]) | set(W["DOWN"]) | set(W["COMMON"]) | set(W["RARE"])
    pool = [w for w in words if w not in skip]
    rng = np.random.default_rng(SEED_NULL)
    for k in range(N_NULL):
        ws = list(rng.choice(pool, len(W["UP"]) + len(W["DOWN"]), replace=False))
        dirs[f"NULL_{k:02d}"] = clean(ws[:len(W["UP"])], ws[len(W["UP"]):])
    print(f"  directions: UP, VALENCE, {N_NULL} random-word; pool {len(pool)} words; cos(UP, VALENCE) {float(dirs['UP'] @ dirs['VALENCE']):+.3f}")

    T = texts()
    toks = [torch.tensor([[BOS] + tk.encode(t, add_special_tokens=False)], device=dev) for t in T]

    def measure(vec):
        """vec: torch [d] added at the site (all non-start positions), or None. Returns [sentences, measures]."""
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
            Rs = torch.stack([cache[h][0, 1:, :].float() for h in hooks])        # [layers SITE..23, tokens, d] (steered at SITE)
            Wr = Rs[1:] - Rs[:-1]                                                # writes of blocks SITE+1..23
            norms = Wr.norm(dim=2)                                               # [writes, tokens]
            net = Wr.sum(dim=0).norm(dim=1); path = norms.sum(dim=0)
            cosw = torch.nn.functional.cosine_similarity(Wr[:-1], Wr[1:], dim=2).mean(dim=0)
            out[i] = [float((net / path).mean()), float(cosw.mean()), float(Rs[-1].norm(dim=1).mean()),
                      float(path.mean()), float(net.mean())]
            if vec is None and i == 0:
                pass
        return out

    res_path = f"{ROOT}/exp188_cells.npz"
    cells = dict(np.load(res_path)) if os.path.exists(res_path) else {}
    if "base" not in cells:
        cells["base"] = measure(None)
        # mean norm of the unsteered state at the site, over all sentence tokens
        tot, cnt = 0.0, 0
        for x in toks:
            with torch.no_grad():
                _, cache = model.run_with_cache(x, names_filter=[site_hook])
            n = cache[site_hook][0, 1:, :].float().norm(dim=1)
            tot += float(n.sum()); cnt += int(n.numel())
        cells["site_norm"] = np.array(tot / cnt)
        np.savez(res_path, **cells)
    site_norm = float(cells["site_norm"])
    b = cells["base"].mean(axis=0)
    print(f"  mean state norm at layer {SITE}: {site_norm:.2f}")
    print("  unsteered: " + "  ".join(f"{m} {v:.4f}" for m, v in zip(MEASURES, b)))
    for name, d in dirs.items():
        for c in CS:
            for sign in (+1, -1):
                key = f"{name}|{c}|{sign:+d}"
                if key in cells:
                    continue
                vec = torch.tensor(sign * c * site_norm * d, dtype=torch.float32, device=dev)
                cells[key] = measure(vec)
                np.savez(res_path, **cells)
        print(f"  {name} done", flush=True)

    rng2 = np.random.default_rng(1880)
    results = {}
    print("\n" + "=" * 76 + "\nA(d, C) = measure with +C*d minus measure with −C*d, mean over 120 sentences\n" + "=" * 76)
    for mi, mname in enumerate(MEASURES):
        results[mname] = {}
        for c in CS:
            A = {n: cells[f"{n}|{c}|+1"][:, mi] - cells[f"{n}|{c}|-1"][:, mi] for n in dirs}
            null = np.array([A[f"NULL_{k:02d}"].mean() for k in range(N_NULL)])
            r = dict(up=float(A["UP"].mean()), up_p=e180.signflip(A["UP"], rng2), up_pos=int((A["UP"] > 0).sum()),
                     val=float(A["VALENCE"].mean()), null_p5=float(np.percentile(null, 5)),
                     null_median=float(np.median(null)), null_p95=float(np.percentile(null, 95)),
                     up_pct=float((null < A["UP"].mean()).mean() * 100),
                     both_signs_vs_base=[float(cells[f"UP|{c}|+1"][:, mi].mean() - cells["base"][:, mi].mean()),
                                         float(cells[f"UP|{c}|-1"][:, mi].mean() - cells["base"][:, mi].mean())])
            results[mname][str(c)] = r
            print(f"  {mname:9s} C={c}: A(UP) {r['up']:+.5f} (p {r['up_p']:.4f}, {r['up_pos']}/120 sentences) | "
                  f"random-word null: median {r['null_median']:+.5f} [5% {r['null_p5']:+.5f}, 95% {r['null_p95']:+.5f}]; "
                  f"UP at the {r['up_pct']:.0f}th percentile | A(VALENCE) {r['val']:+.5f} | "
                  f"vs unsteered: +UP {r['both_signs_vs_base'][0]:+.5f}, −UP {r['both_signs_vs_base'][1]:+.5f}")

    def label(mname):
        rs = [results[mname][str(c)] for c in CS]
        if all(r["up"] > 0 and r["up_p"] < ALPHA and r["up"] > r["null_p95"] for r in rs):
            return "UP_RAISES_AGREEMENT"
        if all(r["up"] < 0 and r["up_p"] < ALPHA and r["up"] < r["null_p5"] for r in rs):
            return "WRONG_SIGN"
        return "NOT_BEYOND_RANDOM"

    v = label("straight")
    print(f"\n  >>> STRAIGHT (primary): {v} <<<")
    print(f"  same rule on WRITE_COS: {label('write_cos')}; on the endpoint norm: {label('end_norm')}")
    json.dump({"verdict": v, "write_cos": label("write_cos"), "end_norm": label("end_norm"),
               "results": results, "site_norm": site_norm, "base": dict(zip(MEASURES, b.tolist()))},
              open(f"{ROOT}/exp188_results.json", "w"), indent=1)
    print("results saved to exp188_results.json")


if __name__ == "__main__":
    main()
