"""
exp180_paired_retry.py — paired-opposites retry of the word-level "holding
against gravity" lean. Frozen rules: PREREG_exp180.md (commit 8b60ab6,
2026-10-02 18:40 IST). Written AFTER the freeze.

The measure and its machinery are exp175's (Runner.clean / drop_kl / holes,
mechanics_checks), reused by import; only the model loader is generalised
so the same code runs Pythia 410M and GPT-2-medium.

Usage:  ./lakoff/bin/python3 exp180_paired_retry.py --selftest-only
        ./lakoff/bin/python3 exp180_paired_retry.py pythia-410m
        ./lakoff/bin/python3 exp180_paired_retry.py gpt2-medium
"""
import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)
import exp175_holding_up as e175

ALPHA = 0.05
N_PERM = 10000
N_BOOT = 2000
IDENT_TOL = 0.001
N_LAYERS = 24
PREREG = open(f"{ROOT}/PREREG_exp180.md").read()
for _name, _val in [("ALPHA", ALPHA), ("N_PERM", N_PERM), ("N_BOOT", N_BOOT), ("IDENT_TOL", IDENT_TOL)]:
    _m = re.search(rf"\b{_name} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_val), f"{_name} drifted from prereg"
_blob = open(f"{ROOT}/exp180_stimuli.json", "rb").read()
STIM_SHA = hashlib.sha256(_blob).hexdigest()
assert STIM_SHA in PREREG, "stimuli do not match the sha256 in the prereg"
STIM = json.loads(_blob)
assert e175.IDENT_TOL == IDENT_TOL and e175.N_LAYERS == N_LAYERS


class Runner2(e175.Runner):
    def __init__(self, name):
        import torch
        from transformer_lens import HookedTransformer
        self.torch = torch
        print(f"\nLoading {name}...")
        self.model = HookedTransformer.from_pretrained(name, device="mps")
        self.model.eval()
        assert self.model.cfg.n_layers == N_LAYERS
        self.dev = self.model.cfg.device
        self.hooks_all = [f"blocks.{L}.hook_resid_post" for L in range(N_LAYERS)]
        self.tk = self.model.tokenizer
        self.bos = int(self.tk.bos_token_id)
        assert self.tk.decode([self.bos]) == "<|endoftext|>"


# ---- analysis ---------------------------------------------------------------
def signflip(d, rng):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    return float((np.sum(np.abs((signs * d).mean(axis=1)) >= obs - 1e-15) + 1) / (N_PERM + 1))


def analyse(value, zipf, pairs, key="reach", seed=180):
    """value: {word: {'reach': x, 'next': y}}; pairs: {set: [[a, b], ...]}."""
    rng = np.random.default_rng(seed)
    out = {}
    diffs = {}
    for name, ps in pairs.items():
        d = np.array([value[a][key] - value[b][key] for a, b in ps])
        dz = np.array([zipf[a] - zipf[b] for a, b in ps])
        diffs[name] = d
        X = np.column_stack([np.ones(len(d)), dz])
        icpt = float(np.linalg.lstsq(X, d, rcond=None)[0][0])
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(d), len(d))
            boots.append(np.linalg.lstsq(X[idx], d[idx], rcond=None)[0][0])
        out[name] = dict(n=len(d), mean=float(d.mean()), d_z=float(d.mean() / d.std(ddof=1)),
                         p=signflip(d, rng), n_positive=int((d > 0).sum()), intercept=icpt,
                         intercept_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                         mean_zipf_diff=float(dz.mean()))
    v, c = diffs["vertical"], diffs["polar"]
    obs = v.mean() - c.mean()
    both = np.concatenate([v, c]); k = len(v)
    X = rng.permuted(np.tile(both, (N_PERM, 1)), axis=1)
    perm = X[:, :k].mean(axis=1) - X[:, k:].mean(axis=1)
    out["spec"] = dict(diff=float(obs), p=float((np.sum(np.abs(perm) >= abs(obs) - 1e-15) + 1) / (N_PERM + 1)))
    return out


def verdict(out):
    v = out["vertical"]; s = out["spec"]
    if v["mean"] > 0 and v["p"] < ALPHA and v["intercept_ci"][0] > 0:
        return "HOLDS_SPECIFIC" if (s["diff"] > 0 and s["p"] < ALPHA) else "HOLDS_GENERIC"
    if v["mean"] < 0 and v["p"] < ALPHA:
        return "WRONG_SIGN"
    return "NULL"


def report(out, label):
    print(f"  [{label}]")
    for name in ("vertical", "polar", "valence"):
        r = out[name]
        print(f"    {name:8s} {r['n']} pairs: first minus second = {r['mean']:+.3f}  d_z {r['d_z']:+.2f}  "
              f"sign-flip p {r['p']:.4f}  ({r['n_positive']}/{r['n']} pairs positive)  "
              f"| at equal frequency {r['intercept']:+.3f} CI[{r['intercept_ci'][0]:+.3f},{r['intercept_ci'][1]:+.3f}]"
              f"  (mean zipf gap {r['mean_zipf_diff']:+.2f})")
    print(f"    vertical minus polar: {out['spec']['diff']:+.3f}  p {out['spec']['p']:.4f}")


def self_test():
    print("SYNTHETIC SELF-TEST (real verdict path)")
    ok = True
    want = {"up_only": "HOLDS_SPECIFIC", "all_pairs": "HOLDS_GENERIC", "nothing": "NULL", "frequency": "NULL"}
    for kind, expect in want.items():
        rng = np.random.default_rng(1)
        value, zipf, pairs = {}, {}, {}
        k = 0
        for name, n in (("vertical", 31), ("polar", 30), ("valence", 20)):
            ps = []
            for _ in range(n):
                a, b = f"w{k}", f"w{k + 1}"; k += 2
                za, zb = rng.normal(4.5, 0.8), rng.normal(4.5, 0.8)
                if kind == "frequency" and name == "vertical":
                    za += 1.0
                shift = 0.3 if (kind == "all_pairs" or (kind == "up_only" and name == "vertical")) else 0.0
                base = rng.normal(0, 0.3)
                va = base + shift + 0.25 * rng.normal() + (0.4 * za if kind == "frequency" else 0)
                vb = base + 0.25 * rng.normal() + (0.4 * zb if kind == "frequency" else 0)
                value[a] = {"reach": va, "next": va}; value[b] = {"reach": vb, "next": vb}
                zipf[a], zipf[b] = za, zb
                ps.append([a, b])
            pairs[name] = ps
        got = verdict(analyse(value, zipf, pairs))
        good = got == expect
        ok &= good
        print(f"  world {kind}: {got} (want {expect}) {'ok' if good else 'FAIL'}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return ok


def main(name):
    from wordfreq import zipf_frequency
    run = Runner2(name)
    torch = run.torch
    frames = STIM["frames"]; pairs = STIM["pairs"]
    words = sorted({w for ps in pairs.values() for p in ps for w in p})
    assert all(len(run.tk.encode(" " + w, add_special_tokens=False)) == 1 for w in words)

    def frame_text(w, fr):
        ws = fr.split(" "); slot = ws.index("{w}"); ws[slot] = w
        g = run.groups(ws)
        return [t for x in g for t in x], sum(len(x) for x in g[: slot + 1])

    d_model = run.model.cfg.d_model
    mu_path = f"{ROOT}/exp180_cache_mu_{name}.npz"
    if os.path.exists(mu_path) and str(np.load(mu_path)["sha"]) == STIM_SHA:
        mu = np.load(mu_path)["mu"]; print("  layer averages loaded from cache")
    else:
        tot = np.zeros((N_LAYERS, d_model)); cnt = 0
        for wi, w in enumerate(words):
            for fr in frames:
                ids, _ = frame_text(w, fr)
                toks, resid, _ = run.clean(ids)
                tot += resid[:, 1:, :].float().sum(dim=1).cpu().numpy().astype(np.float64); cnt += toks.shape[1] - 1
        mu = tot / cnt
        np.savez(mu_path, mu=mu, sha=STIM_SHA, n_tokens=cnt)
        print(f"  layer averages over {cnt} tokens ({len(words)} words x {len(frames)} frames) (saved)")
    mu_list = [torch.tensor(mu[L], dtype=torch.float32, device=run.dev) for L in range(N_LAYERS)]
    checks = [frame_text(w, frames[i]) for i, w in enumerate(["the", "and", "with", "that"])]
    if not e175.mechanics_checks(run, checks, mu_list):
        print("STOP: the drop mechanics are wrong. Nothing below was computed.")
        sys.exit(2)

    rows_path = f"{ROOT}/exp180_rows_{name}.jsonl"
    rows = {}
    if os.path.exists(rows_path):
        for line in open(rows_path):
            r = json.loads(line); rows[(r["word"], r["frame"])] = r
        print(f"  resuming: {len(rows)} rows on disk")
    with open(rows_path, "a") as fh:
        for wi, w in enumerate(words):
            for fi, fr in enumerate(frames):
                if (w, fi) in rows:
                    continue
                ids, p = frame_text(w, fr)
                toks, resid, logp = run.clean(ids)
                e_next, e_reach = run.holes(toks, logp, mu_list, [p])
                r = {"word": w, "frame": fi, "E_next": float(e_next[0]), "E_reach": float(e_reach[0])}
                fh.write(json.dumps(r) + "\n"); fh.flush(); rows[(w, fi)] = r
            if (wi + 1) % 40 == 0:
                print(f"  holes {wi + 1}/{len(words)} words (saved as they finish)", flush=True)
    nf = len(frames)
    value = {w: {"reach": float(np.mean([np.log(rows[(w, f)]["E_reach"]) for f in range(nf)])),
                 "next": float(np.mean([np.log(rows[(w, f)]["E_next"]) for f in range(nf)]))} for w in words}
    zipf = {w: zipf_frequency(w, "en") for w in words}
    print("\n" + "=" * 72 + f"\nRESULTS — {name}\n" + "=" * 72)
    out = analyse(value, zipf, pairs, "reach")
    report(out, "reach (PRIMARY)")
    v = verdict(out)
    out_next = analyse(value, zipf, pairs, "next")
    report(out_next, "next-word readout (reported)")
    print(f"\n  >>> {name}: {v} <<<")
    json.dump({"model": name, "reach": out, "next": out_next, "verdict": v, "values": value, "zipf": zipf},
              open(f"{ROOT}/exp180_results_{name}.json", "w"), indent=1)
    print(f"results saved to exp180_results_{name}.json")


if __name__ == "__main__":
    print("exp180 — paired-opposites retry (prereg frozen at 8b60ab6); stimuli sha matches prereg")
    if not self_test():
        sys.exit(1)
    if "--selftest-only" in sys.argv:
        sys.exit(0)
    main(sys.argv[1])
