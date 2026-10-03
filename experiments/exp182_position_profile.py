"""
exp182_position_profile.py — position-by-position test of "UPness is like
potential energy". Frozen rules: PREREG_exp182.md (commit 1a471e5,
2026-10-02 19:26 IST). Written AFTER the freeze.

The measure and its machinery are exp175's / exp180's, reused by import.
The only change: the KL is kept separately for each position k = 0..12
after the word instead of being averaged over later positions.

Usage:  ./lakoff/bin/python3 exp182_position_profile.py --selftest-only
        ./lakoff/bin/python3 exp182_position_profile.py pythia-410m
        ./lakoff/bin/python3 exp182_position_profile.py gpt2-medium
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
import exp180_paired_retry as e180

ALPHA = 0.05
N_PERM = 10000
K = 12
IDENT_TOL = 0.001
N_LAYERS = 24
PREREG = open(f"{ROOT}/PREREG_exp182.md").read()
for _name, _val in [("ALPHA", ALPHA), ("N_PERM", N_PERM), ("K", K), ("IDENT_TOL", IDENT_TOL)]:
    _m = re.search(rf"\b{_name} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_val), f"{_name} drifted from prereg"
_blob = open(f"{ROOT}/exp182_stimuli.json", "rb").read()
STIM_SHA = hashlib.sha256(_blob).hexdigest()
assert STIM_SHA in PREREG, "stimuli do not match the sha256 in the prereg"
STIM = json.loads(_blob)
assert STIM["K"] == K and e180.N_PERM == N_PERM
KS = np.arange(1, K + 1, dtype=float)


def slopes_of(profile, ps):
    """profile: {word: [K+1] ln values}. Per pair: gap_k and the OLS slope of gap on k, k = 1..K."""
    gaps = np.array([np.array(profile[a]) - np.array(profile[b]) for a, b in ps])       # [pairs, K+1]
    x = KS - KS.mean()
    sl = (gaps[:, 1:] @ x) / (x @ x)
    return gaps, sl


def analyse(profile, pairs, seed=182):
    rng = np.random.default_rng(seed)
    out = {}
    sl = {}
    for name, ps in pairs.items():
        gaps, s = slopes_of(profile, ps)
        sl[name] = s
        out[name] = dict(n=len(ps), slope_mean=float(s.mean()), slope_dz=float(s.mean() / s.std(ddof=1)),
                         slope_p=e180.signflip(s, rng), n_slope_pos=int((s > 0).sum()),
                         gap_by_k=[float(g) for g in gaps.mean(axis=0)],
                         gap_k0=float(gaps[:, 0].mean()), gap_k0_p=e180.signflip(gaps[:, 0], rng),
                         gap_later=float(gaps[:, 1:].mean()), gap_later_p=e180.signflip(gaps[:, 1:].mean(axis=1), rng))
    v, c = sl["vertical"], sl["polar"]
    obs = v.mean() - c.mean()
    both = np.concatenate([v, c]); k = len(v)
    X = rng.permuted(np.tile(both, (N_PERM, 1)), axis=1)
    perm = X[:, :k].mean(axis=1) - X[:, k:].mean(axis=1)
    out["spec"] = dict(diff=float(obs), p=float((np.sum(np.abs(perm) >= abs(obs) - 1e-15) + 1) / (N_PERM + 1)))
    return out


def verdict(out):
    v, s = out["vertical"], out["spec"]
    if v["slope_mean"] > 0 and v["slope_p"] < ALPHA:
        return "RISES_SPECIFIC" if (s["diff"] > 0 and s["p"] < ALPHA) else "RISES_GENERIC"
    if v["slope_mean"] < 0 and v["slope_p"] < ALPHA:
        return "FADES"
    return "FLAT"


def report(out):
    print("  gap (first word minus second word, in ln KL) at each distance k from the word:")
    print("     k:        " + " ".join(f"{k:>6d}" for k in range(K + 1)))
    for name in ("vertical", "polar", "valence"):
        print(f"     {name:9s} " + " ".join(f"{g:+6.3f}" for g in out[name]["gap_by_k"]))
    for name in ("vertical", "polar", "valence"):
        r = out[name]
        print(f"  {name:8s} slope over k=1..{K}: {r['slope_mean']:+.4f} per position  d_z {r['slope_dz']:+.2f}  p {r['slope_p']:.4f}"
              f"  ({r['n_slope_pos']}/{r['n']} pairs rising) | gap at the word itself {r['gap_k0']:+.3f} (p {r['gap_k0_p']:.4f})"
              f" | mean gap k=1..{K} {r['gap_later']:+.3f} (p {r['gap_later_p']:.4f})")
    print(f"  vertical slope minus polar slope: {out['spec']['diff']:+.4f}  p {out['spec']['p']:.4f}")


def self_test():
    print("SYNTHETIC SELF-TEST (real verdict path)")
    ok = True
    want = {"rises_up_only": "RISES_SPECIFIC", "rises_all": "RISES_GENERIC", "fades": "FADES", "flat": "FLAT"}
    for kind, expect in want.items():
        rng = np.random.default_rng(2)
        profile, pairs = {}, {}
        n = 0
        for name, cnt in (("vertical", 31), ("polar", 30), ("valence", 20)):
            ps = []
            for _ in range(cnt):
                a, b = f"w{n}", f"w{n + 1}"; n += 2
                base = rng.normal(0, 0.3) - 0.15 * np.arange(K + 1)
                rate = {"rises_up_only": 0.03 if name == "vertical" else 0.0, "rises_all": 0.03,
                        "fades": -0.03, "flat": 0.0}[kind]
                profile[a] = list(base + 0.15 + rate * np.arange(K + 1) + rng.normal(0, 0.08, K + 1))
                profile[b] = list(base + rng.normal(0, 0.08, K + 1))
                ps.append([a, b])
            pairs[name] = ps
        got = verdict(analyse(profile, pairs))
        good = got == expect
        ok &= good
        print(f"  world {kind}: {got} (want {expect}) {'ok' if good else 'FAIL'}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return ok


def main(name):
    run = e180.Runner2(name)
    torch = run.torch
    frames = STIM["frames"]; pairs = STIM["pairs"]
    words = sorted({w for ps in pairs.values() for p in ps for w in p})
    assert all(len(run.tk.encode(" " + w, add_special_tokens=False)) == 1 for w in words)

    def frame_text(w, fr):
        ws = fr.split(" "); slot = ws.index("{w}"); ws[slot] = w
        g = run.groups(ws)
        return [t for x in g for t in x], sum(len(x) for x in g[: slot + 1])

    d_model = run.model.cfg.d_model
    mu_path = f"{ROOT}/exp182_cache_mu_{name}.npz"
    if os.path.exists(mu_path) and str(np.load(mu_path)["sha"]) == STIM_SHA:
        mu = np.load(mu_path)["mu"]; print("  layer averages loaded from cache")
    else:
        tot = np.zeros((N_LAYERS, d_model)); cnt = 0
        for w in words:
            for fr in frames:
                ids, _ = frame_text(w, fr)
                toks, resid, _ = run.clean(ids)
                tot += resid[:, 1:, :].float().sum(dim=1).cpu().numpy().astype(np.float64); cnt += toks.shape[1] - 1
        mu = tot / cnt
        np.savez(mu_path, mu=mu, sha=STIM_SHA, n_tokens=cnt)
        print(f"  layer averages over {cnt} tokens (saved)")
    mu_list = [torch.tensor(mu[L], dtype=torch.float32, device=run.dev) for L in range(N_LAYERS)]
    checks = [frame_text(w, frames[i]) for i, w in enumerate(["the", "and", "with", "that"])]
    if not e175.mechanics_checks(run, checks, mu_list):
        print("STOP: the drop mechanics are wrong. Nothing below was computed.")
        sys.exit(2)

    rows_path = f"{ROOT}/exp182_rows_{name}.jsonl"
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
                assert toks.shape[1] - 1 - p >= K, "frame too short after the slot"
                kl = run.drop_kl(toks, [(p, L) for L in range(N_LAYERS)], mu_list, logp)     # [24, T1]
                r = {"word": w, "frame": fi, "by_k": [float(kl[:, p + k].sum()) for k in range(K + 1)]}
                fh.write(json.dumps(r) + "\n"); fh.flush(); rows[(w, fi)] = r
            if (wi + 1) % 40 == 0:
                print(f"  {wi + 1}/{len(words)} words (saved as they finish)", flush=True)
    nf = len(frames)
    profile = {w: [float(np.mean([np.log(rows[(w, f)]["by_k"][k]) for f in range(nf)])) for k in range(K + 1)]
               for w in words}
    print("\n" + "=" * 72 + f"\nRESULTS — {name}\n" + "=" * 72)
    out = analyse(profile, pairs)
    report(out)
    v = verdict(out)
    print(f"\n  >>> {name}: {v} <<<")
    json.dump({"model": name, "analysis": out, "verdict": v, "profile": profile},
              open(f"{ROOT}/exp182_results_{name}.json", "w"), indent=1)
    print(f"results saved to exp182_results_{name}.json")


if __name__ == "__main__":
    print("exp182 — position-by-position test (prereg frozen at 1a471e5); stimuli sha matches prereg")
    if not self_test():
        sys.exit(1)
    if len(sys.argv) < 2 or sys.argv[1] == "--selftest-only":
        sys.exit(0)
    main(sys.argv[1])
