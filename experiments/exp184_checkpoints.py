"""exp184_checkpoints.py — when does the opposite-pair asymmetry appear in
training? Frozen rules: PREREG_exp184.md (commit 0b2c8ad, 2026-10-02 19:55
IST). Written AFTER the freeze. exp180's stimuli, measure and analysis,
imported; one run per saved Pythia 410M checkpoint. Resumes per checkpoint.
"""
import json, os, re, sys
import numpy as np
ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)
import exp180_paired_retry as e180

ALPHA, N_PERM, GATE_TOL, NOISE_FLOOR, N_LAYERS, IDENT_TOL = 0.05, 10000, 0.01, 0.0001, 24, 0.001
STEPS = [0, 512, 4000, 16000, 64000, None]          # None = the final model (step 143000)
PREREG = open(f"{ROOT}/PREREG_exp184.md").read()
for _n, _v in [("ALPHA", ALPHA), ("N_PERM", N_PERM), ("GATE_TOL", GATE_TOL), ("NOISE_FLOOR", NOISE_FLOOR)]:
    _m = re.search(rf"\b{_n} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_v), f"{_n} drifted from prereg"


class Runner3(e180.Runner2):
    def __init__(self, step):
        import torch
        from transformer_lens import HookedTransformer
        self.torch = torch
        kw = {} if step is None else {"checkpoint_value": step}
        self.model = HookedTransformer.from_pretrained("pythia-410m", device="mps", **kw)
        self.model.eval()
        assert self.model.cfg.n_layers == N_LAYERS
        self.dev = self.model.cfg.device
        self.hooks_all = [f"blocks.{L}.hook_resid_post" for L in range(N_LAYERS)]
        self.tk = self.model.tokenizer
        self.bos = int(self.tk.bos_token_id)


def one(step):
    from wordfreq import zipf_frequency
    tag = "final" if step is None else f"step{step}"
    res_path = f"{ROOT}/exp184_results_{tag}.json"
    if os.path.exists(res_path):
        return json.load(open(res_path))
    print(f"\n{'=' * 72}\nCHECKPOINT {tag}\n{'=' * 72}", flush=True)
    run = Runner3(step)
    torch = run.torch
    STIM = e180.STIM; frames = STIM["frames"]; pairs = STIM["pairs"]
    words = sorted({w for ps in pairs.values() for p in ps for w in p})

    def frame_text(w, fr):
        ws = fr.split(" "); slot = ws.index("{w}"); ws[slot] = w
        g = run.groups(ws)
        return [t for x in g for t in x], sum(len(x) for x in g[: slot + 1])
    tot = np.zeros((N_LAYERS, run.model.cfg.d_model)); cnt = 0
    for w in words:
        for fr in frames:
            ids, _ = frame_text(w, fr)
            toks, resid, _ = run.clean(ids)
            tot += resid[:, 1:, :].float().sum(dim=1).cpu().numpy().astype(np.float64); cnt += toks.shape[1] - 1
    mu = tot / cnt
    mu_list = [torch.tensor(mu[L], dtype=torch.float32, device=run.dev) for L in range(N_LAYERS)]
    # mechanics, without the "a real drop must be large" condition (not guaranteed in an untrained model)
    ids, p = frame_text("the", frames[0])
    toks, resid, logp = run.clean(ids)
    jobs = [(p, L) for L in range(N_LAYERS)]
    own = run.drop_kl(toks, jobs, [resid[L] for L in range(N_LAYERS)], logp)
    kl = run.drop_kl(toks, jobs, mu_list, logp)
    c1, c2, c3 = float(np.abs(own).max()), float(np.abs(kl[N_LAYERS - 1, p + 1:]).max()), float(np.abs(kl[:, :p]).max())
    print(f"  mechanics: own-state {c1:.1e}, last-layer later positions {c2:.1e}, earlier positions {c3:.1e}"
          f" -> {'ok' if max(c1, c2, c3) < IDENT_TOL else 'FAIL'}")
    if max(c1, c2, c3) >= IDENT_TOL:
        print("STOP: the drop mechanics are wrong."); sys.exit(2)
    path = f"{ROOT}/exp184_rows_{tag}.jsonl"
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
                e_next, e_reach = run.holes(toks, logp, mu_list, [p])
                r = {"word": w, "frame": fi, "E_next": float(e_next[0]), "E_reach": float(e_reach[0])}
                fh.write(json.dumps(r) + "\n"); fh.flush(); rows[(w, fi)] = r
    nf = len(frames)
    level = float(np.mean([rows[(w, f)]["E_reach"] for w in words for f in range(nf)]))
    value = {w: {"reach": float(np.mean([np.log(max(rows[(w, f)]["E_reach"], 1e-30)) for f in range(nf)])),
                 "next": float(np.mean([np.log(max(rows[(w, f)]["E_next"], 1e-30)) for f in range(nf)]))} for w in words}
    zipf = {w: zipf_frequency(w, "en") for w in words}
    out = e180.analyse(value, zipf, pairs, "reach")
    out["level"] = level; out["tag"] = tag
    json.dump(out, open(res_path, "w"), indent=1)
    del run
    torch.mps.empty_cache()
    return out


if __name__ == "__main__":
    print("exp184 — training checkpoints (prereg frozen at 0b2c8ad)")
    res = [one(s) for s in STEPS]
    print("\n" + "=" * 72 + "\nRESULTS\n" + "=" * 72)
    print(" checkpoint   mean hole    up/down pairs: lead (p, pairs positive)    other opposites: lead (p)")
    shows = {}
    for r in res:
        v, c = r["vertical"], r["polar"]
        below = r["level"] < NOISE_FLOOR
        shows[r["tag"]] = (not below) and v["mean"] > 0 and v["p"] < ALPHA
        print(f"  {r['tag']:>9s}   {r['level']:9.5f}    {v['mean']:+.3f} (p {v['p']:.4f}, {v['n_positive']}/{v['n']})"
              f"{'  BELOW FLOOR' if below else ''}              {c['mean']:+.3f} (p {c['p']:.4f})")
    gate = abs(res[-1]["vertical"]["mean"] - 0.203) <= GATE_TOL
    print(f"\n  gate: final checkpoint {res[-1]['vertical']['mean']:+.3f} against exp180's +0.203 -> {'PASS' if gate else 'FAIL'}")
    if not gate:
        verdict = "INVALID (gate failed)"
    elif shows["step0"]:
        verdict = "AT_INIT"
    elif shows["final"] and res[0]["level"] >= NOISE_FLOOR:
        first = next(t for t in ["step512", "step4000", "step16000", "step64000", "final"] if shows[t])
        verdict = f"LEARNED (first shows at {first})"
    else:
        verdict = "UNCLEAR"
    print(f"\n  >>> {verdict} <<<")
    json.dump({"verdict": verdict, "shows": shows}, open(f"{ROOT}/exp184_results.json", "w"), indent=1)
