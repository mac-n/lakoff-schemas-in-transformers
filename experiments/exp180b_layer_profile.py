"""exp180b_layer_profile.py — DESCRIPTIVE follow-up to exp180 (see the
FOLLOW-UP section of PREREG_exp180.md, written before this run): the same
holes, with each layer's contribution kept. No verdict."""
import json, os, sys
import numpy as np
ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)
import exp180_paired_retry as e180
N_LAYERS = 24

def main(name):
    run = e180.Runner2(name)
    torch = run.torch
    STIM = e180.STIM; frames = STIM["frames"]; pairs = STIM["pairs"]
    words = sorted({w for ps in pairs.values() for p in ps for w in p})
    mu = np.load(f"{ROOT}/exp180_cache_mu_{name}.npz")["mu"]
    mu_list = [torch.tensor(mu[L], dtype=torch.float32, device=run.dev) for L in range(N_LAYERS)]
    def frame_text(w, fr):
        ws = fr.split(" "); slot = ws.index("{w}"); ws[slot] = w
        g = run.groups(ws)
        return [t for x in g for t in x], sum(len(x) for x in g[: slot + 1])
    path = f"{ROOT}/exp180b_rows_{name}.jsonl"
    rows = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line); rows[(r["word"], r["frame"])] = r
    with open(path, "a") as fh:
        for wi, w in enumerate(words):
            for fi, fr in enumerate(frames):
                if (w, fi) in rows:
                    continue
                ids, p = frame_text(w, fr)
                toks, resid, logp = run.clean(ids)
                kl = run.drop_kl(toks, [(p, L) for L in range(N_LAYERS)], mu_list, logp)   # [24, T1]
                r = {"word": w, "frame": fi, "reach": [float(kl[L, p + 1:].mean()) for L in range(N_LAYERS)],
                     "next": [float(kl[L, p]) for L in range(N_LAYERS)]}
                fh.write(json.dumps(r) + "\n"); fh.flush(); rows[(w, fi)] = r
            if (wi + 1) % 40 == 0:
                print(f"  {wi + 1}/{len(words)} words", flush=True)
    old = {(json.loads(l)["word"], json.loads(l)["frame"]): json.loads(l) for l in open(f"{ROOT}/exp180_rows_{name}.jsonl")}
    worst = max(abs(sum(rows[k]["reach"]) - old[k]["E_reach"]) / old[k]["E_reach"] for k in rows)
    print(f"\n{name}: layer contributions sum to exp180's totals (worst relative difference {worst:.1e})")
    nf = len(frames)
    R = {w: np.array([[rows[(w, f)]["reach"][L] for L in range(23)] for f in range(nf)]) for w in words}
    lnR = {w: np.log(R[w]).mean(axis=0) for w in words}
    share = np.mean([R[w].mean(axis=0) / R[w].mean(axis=0).sum() for w in words], axis=0)
    rng = np.random.default_rng(1800)
    out = {"share": share.tolist()}
    print(" layer  share of hole | vertical: UP minus DOWN (p) | other opposites: first minus second (p)")
    for L in range(23):
        line = f"  {L:>3}     {100 * share[L]:5.1f}%     |"
        for setname in ("vertical", "polar"):
            d = np.array([lnR[a][L] - lnR[b][L] for a, b in pairs[setname]])
            p = e180.signflip(d, rng)
            out.setdefault(setname, []).append({"mean": float(d.mean()), "p": p, "n_pos": int((d > 0).sum())})
            line += f"   {d.mean():+.3f} ({p:.3f}) {'*' if p < 0.05 else ' '} {int((d > 0).sum()):>2}/{len(d)}   |"
        print(line)
    npos = sum(x["mean"] > 0 for x in out["vertical"])
    print(f"  vertical advantage positive at {npos}/23 layers; significant (p<0.05) at {sum(x['p'] < 0.05 and x['mean'] > 0 for x in out['vertical'])}")
    json.dump(out, open(f"{ROOT}/exp180b_results_{name}.json", "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1])
