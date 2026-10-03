"""
exp183_free_generation.py — is the UP lead "spent" when the model writes on
its own? Frozen rules: PREREG_exp183.md (commit befa567, 2026-10-02 19:38
IST). Written AFTER the freeze.

The prompt ends at the word; the model samples 40 tokens; along each sampled
text, KL(clean || word dropped to the layer mean at one layer), summed over
DROP_LAYERS. The drop is exp175's (checked against exp175's drop_kl below).

# NOT ALL LAYERS: the hole is summed over 8 of the 24 layers (DROP_LAYERS),
# a cost choice flagged in the prereg's "Deviations and stubs".

Usage:  ./lakoff/bin/python3 exp183_free_generation.py --selftest-only
        ./lakoff/bin/python3 exp183_free_generation.py pythia-410m
        ./lakoff/bin/python3 exp183_free_generation.py gpt2-medium
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
N_SAMPLES = 3
T_GEN = 40
NOISE_FLOOR = 0.0001
DROP_LAYERS = [0, 2, 4, 6, 8, 10, 12, 14]
N_LAYERS = 24
BIN = 5
N_BINS = T_GEN // BIN
PREREG = open(f"{ROOT}/PREREG_exp183.md").read()
for _name, _val in [("ALPHA", ALPHA), ("N_PERM", N_PERM), ("N_SAMPLES", N_SAMPLES), ("T_GEN", T_GEN),
                    ("NOISE_FLOOR", NOISE_FLOOR)]:
    _m = re.search(rf"\b{_name} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_val), f"{_name} drifted from prereg"
assert "DROP_LAYERS = " + " ".join(str(L) for L in DROP_LAYERS) in PREREG
STIM = json.load(open(f"{ROOT}/exp182_stimuli.json"))          # frames and pairs come from exp182 / exp180
assert hashlib.sha256(open(f"{ROOT}/exp182_stimuli.json", "rb").read()).hexdigest() == \
    "e141ee5bdf271ceb484521713eb883d445f58d1784d395c19061bc8a3f34734a"


# ---- analysis ---------------------------------------------------------------
def summarise(obs):
    """obs: {word: array [n_obs, T_GEN + 1] of KL by k}. Returns per-word bin
    values (mean over observations of ln(mean KL in bin)), totals, and the
    all-word mean KL per bin (for the noise floor)."""
    binned, total, level = {}, {}, np.zeros(N_BINS)
    for w, a in obs.items():
        a = np.asarray(a, float)
        b = np.stack([a[:, 1 + BIN * j: 1 + BIN * (j + 1)].mean(axis=1) for j in range(N_BINS)], axis=1)
        binned[w] = np.log(np.maximum(b, 1e-30)).mean(axis=0)
        total[w] = float(np.log(a[:, 1:].sum(axis=1)).mean())
        level += b.mean(axis=0)
    return binned, total, level / len(obs)


def analyse(obs, pairs, seed=183):
    rng = np.random.default_rng(seed)
    binned, total, level = summarise(obs)
    valid = [j for j in range(N_BINS) if level[j] >= NOISE_FLOOR]
    out = {"bin_level": [float(x) for x in level], "valid_bins": valid}
    early_bins = [j for j in (0, 1) if j in valid]
    x = np.array(valid, float); xc = x - x.mean()
    store = {}
    for name, ps in pairs.items():
        gaps = np.array([binned[a] - binned[b] for a, b in ps])                 # [pairs, bins]
        early = gaps[:, early_bins].mean(axis=1) if early_bins else np.full(len(ps), np.nan)
        slope = (gaps[:, valid] @ xc) / (xc @ xc) if len(valid) >= 3 else np.full(len(ps), np.nan)
        tot = np.array([total[a] - total[b] for a, b in ps])
        store[name] = (early, slope)
        out[name] = dict(n=len(ps), gap_by_bin=[float(g) for g in gaps.mean(axis=0)],
                         early=float(early.mean()), early_p=e180.signflip(early, rng),
                         early_pos=int((early > 0).sum()),
                         slope=float(slope.mean()), slope_p=e180.signflip(slope, rng),
                         slope_neg=int((slope < 0).sum()),
                         total=float(tot.mean()), total_p=e180.signflip(tot, rng))
    for key, idx in (("spec_early", 0), ("spec_slope", 1)):
        v, c = store["vertical"][idx], store["polar"][idx]
        obs_d = v.mean() - c.mean()
        both = np.concatenate([v, c]); k = len(v)
        X = rng.permuted(np.tile(both, (N_PERM, 1)), axis=1)
        perm = X[:, :k].mean(axis=1) - X[:, k:].mean(axis=1)
        out[key] = dict(diff=float(obs_d), p=float((np.sum(np.abs(perm) >= abs(obs_d) - 1e-15) + 1) / (N_PERM + 1)))
    return out


def verdict(out):
    v = out["vertical"]
    lead = v["early"] > 0 and v["early_p"] < ALPHA
    if not lead:
        return "NO_LEAD"
    return "SPENT" if (v["slope"] < 0 and v["slope_p"] < ALPHA) else "LEAD_NOT_SPENT"


def report(out):
    lab = [f"{1 + BIN * j}-{BIN * (j + 1)}" for j in range(N_BINS)]
    print("  tokens after the word:      " + " ".join(f"{s:>7s}" for s in lab))
    print("  mean KL over all words:     " + " ".join(f"{x:7.4f}" for x in out["bin_level"])
          + f"   (bins used: {[lab[j] for j in out['valid_bins']]})")
    for name in ("vertical", "polar", "valence"):
        print(f"  gap, {name:9s}             " + " ".join(f"{g:+7.3f}" for g in out[name]["gap_by_bin"]))
    for name in ("vertical", "polar", "valence"):
        r = out[name]
        print(f"  {name:8s} early lead (tokens 1-10) {r['early']:+.3f} p {r['early_p']:.4f} ({r['early_pos']}/{r['n']} pairs) | "
              f"slope per bin {r['slope']:+.4f} p {r['slope_p']:.4f} ({r['slope_neg']}/{r['n']} falling) | "
              f"whole text {r['total']:+.3f} p {r['total_p']:.4f}")
    print(f"  vertical minus polar: early lead {out['spec_early']['diff']:+.3f} (p {out['spec_early']['p']:.4f}); "
          f"slope {out['spec_slope']['diff']:+.4f} (p {out['spec_slope']['p']:.4f})")


def self_test():
    print("SYNTHETIC SELF-TEST (real verdict path)")
    ok = True
    want = {"spent": "SPENT", "lead_kept": "LEAD_NOT_SPENT", "no_lead": "NO_LEAD", "floor_only": "LEAD_NOT_SPENT"}
    for kind, expect in want.items():
        rng = np.random.default_rng(5)
        obs, pairs = {}, {}
        n = 0
        ks = np.arange(T_GEN + 1)
        for name, cnt in (("vertical", 31), ("polar", 30), ("valence", 20)):
            ps = []
            for _ in range(cnt):
                a, b = f"w{n}", f"w{n + 1}"; n += 2
                decay = 0.25 if kind == "floor_only" else 0.05
                base = np.log(0.5) - decay * ks + rng.normal(0, 0.2)
                lead = {"spent": 0.5 * np.exp(-ks / 10.0), "lead_kept": 0.3 * np.ones_like(ks, float),
                        "no_lead": np.zeros_like(ks, float), "floor_only": 0.3 * np.ones_like(ks, float)}[kind]
                def draw(extra):
                    v = np.exp(base + extra + rng.normal(0, 0.15, (24, T_GEN + 1)))
                    return np.maximum(v, 2e-6) if kind == "floor_only" else v      # both members hit a floor
                obs[a] = draw(lead); obs[b] = draw(0.0)
                ps.append([a, b])
            pairs[name] = ps
        out = analyse(obs, pairs)
        got = verdict(out)
        good = got == expect
        ok &= good
        print(f"  world {kind}: {got} (want {expect}); bins used {out['valid_bins']} {'ok' if good else 'FAIL'}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return ok


def main(name):
    run = e180.Runner2(name)
    torch = run.torch
    model = run.model
    pairs = STIM["pairs"]
    words = sorted({w for ps in pairs.values() for p in ps for w in p})
    frames = STIM["frames"]
    wid = {w: run.tk.encode(" " + w, add_special_tokens=False) for w in words}
    assert all(len(v) == 1 for v in wid.values())

    # ---- sample the model's own continuations (batched per frame; saved)
    gen_path = f"{ROOT}/exp183_generations_{name}.npz"
    if os.path.exists(gen_path):
        z = np.load(gen_path, allow_pickle=True)
        seqs = {int(k): z[k] for k in z.files}
        print("  sampled texts loaded from disk")
    else:
        seqs = {}
        for fi, fr in enumerate(frames):
            ws = fr.split(" "); slot = ws.index("{w}")
            head = [t for w_ in ws[:slot] for t in run.tk.encode(" " + w_, add_special_tokens=False)]
            rows = [[run.bos] + head + wid[w] for w in words for _ in range(N_SAMPLES)]
            x = torch.tensor(rows, device=run.dev)
            torch.manual_seed(183000 + fi)
            with torch.no_grad():
                g = model.generate(x, max_new_tokens=T_GEN, do_sample=True, temperature=1.0, top_k=None, top_p=None,
                                   stop_at_eos=False, prepend_bos=False, use_past_kv_cache=True,
                                   return_type="tokens", verbose=False)
            assert g.shape == (len(rows), x.shape[1] + T_GEN)
            seqs[fi] = g.cpu().numpy()
            print(f"  sampled prompt {fi + 1}/{len(frames)}: {len(rows)} texts of {g.shape[1]} tokens", flush=True)
        np.savez(gen_path, **{str(k): v for k, v in seqs.items()})
    ex = seqs[0][words.index("up") * N_SAMPLES]
    print("  example (after 'up'):", repr(run.tk.decode(ex[1:])))

    # ---- layer means over all clean runs
    mu_path = f"{ROOT}/exp183_cache_mu_{name}.npz"
    if os.path.exists(mu_path):
        mu = np.load(mu_path)["mu"]
    else:
        tot = np.zeros((N_LAYERS, model.cfg.d_model)); cnt = 0
        for fi in range(len(frames)):
            S = seqs[fi]
            for s in range(0, len(S), 32):
                x = torch.tensor(S[s: s + 32], device=run.dev)
                with torch.no_grad():
                    _, cache = model.run_with_cache(x, names_filter=run.hooks_all)
                for L in range(N_LAYERS):
                    tot[L] += cache[run.hooks_all[L]][:, 1:, :].float().sum(dim=(0, 1)).cpu().numpy().astype(np.float64)
                cnt += x.shape[0] * (x.shape[1] - 1)
                del cache
        mu = tot / cnt
        np.savez(mu_path, mu=mu, n_tokens=cnt)
        print(f"  layer averages over {cnt} tokens (saved)")
    mu_t = [torch.tensor(mu[L], dtype=torch.float32, device=run.dev) for L in range(N_LAYERS)]

    # ---- mechanics: exp175's checks, then this script's batched drop against exp175's drop_kl
    def frame_text(w, fr):
        ws = fr.split(" "); slot = ws.index("{w}"); ws[slot] = w
        g = run.groups(ws)
        return [t for x in g for t in x], sum(len(x) for x in g[: slot + 1])
    if not e175.mechanics_checks(run, [frame_text(w, frames[i]) for i, w in enumerate(["the", "and", "with", "that"])], mu_t):
        print("STOP: the drop mechanics are wrong."); sys.exit(2)

    def batched(S_rows, p):
        """S_rows: [B, T1] token array, word at position p in every row.
        Returns KL by position summed over DROP_LAYERS: [B, T1]."""
        B = len(S_rows); nL = len(DROP_LAYERS)
        x = torch.tensor(S_rows, device=run.dev)
        with torch.no_grad():
            clean_logp = torch.log_softmax(model(x).float(), dim=-1)
        big = x.repeat(nL, 1)
        fwd = []
        for j, L in enumerate(DROP_LAYERS):
            rows = torch.arange(j * B, (j + 1) * B, device=run.dev)
            def hook(resid, hook, rows=rows, L=L):
                resid[rows, p, :] = mu_t[L].to(resid.dtype)
                return resid
            fwd.append((run.hooks_all[L], hook))
        with torch.no_grad():
            logp = torch.log_softmax(model.run_with_hooks(big, fwd_hooks=fwd).float(), dim=-1)
        cp = clean_logp.exp().repeat(nL, 1, 1)
        kl = (cp * (clean_logp.repeat(nL, 1, 1) - logp)).sum(dim=-1).reshape(nL, B, -1).sum(dim=0)
        return kl.float().cpu().numpy().astype(np.float64)

    S0 = seqs[0]; p0 = S0.shape[1] - T_GEN - 1
    mine = batched(S0[:2], p0)
    toks = torch.tensor(S0[:1], device=run.dev)
    with torch.no_grad():
        lp = torch.log_softmax(model(toks)[0].float(), dim=-1)
    ref = run.drop_kl(toks, [(p0, L) for L in DROP_LAYERS], mu_t, lp).sum(axis=0)
    agree = float(np.abs(mine[0] - ref).max())
    print(f"  batched drop vs exp175's drop_kl on one text: max difference {agree:.2e} -> {'ok' if agree < 1e-3 else 'FAIL'}")
    if agree >= 1e-3:
        print("STOP: batched drop disagrees with the reference."); sys.exit(2)

    # ---- the holes along the model's own text
    rows_path = f"{ROOT}/exp183_rows_{name}.jsonl"
    done = set()
    if os.path.exists(rows_path):
        done = {(json.loads(l)["frame"], json.loads(l)["row"]) for l in open(rows_path)}
        print(f"  resuming: {len(done)} texts on disk")
    BATCH = 4
    with open(rows_path, "a") as fh:
        for fi in range(len(frames)):
            S = seqs[fi]; p = S.shape[1] - T_GEN - 1
            for s in range(0, len(S), BATCH):
                idx = [r for r in range(s, min(s + BATCH, len(S))) if (fi, r) not in done]
                if not idx:
                    continue
                kl = batched(S[idx], p)
                for j, r in enumerate(idx):
                    fh.write(json.dumps({"frame": fi, "row": r, "word": words[r // N_SAMPLES],
                                         "by_k": [float(v) for v in kl[j, p: p + T_GEN + 1]]}) + "\n")
                fh.flush()
            print(f"  prompt {fi + 1}/{len(frames)} measured (saved as it goes)", flush=True)
    obs = {w: [] for w in words}
    for line in open(rows_path):
        r = json.loads(line); obs[r["word"]].append(r["by_k"])
    assert all(len(v) == len(frames) * N_SAMPLES for v in obs.values())
    print("\n" + "=" * 72 + f"\nRESULTS — {name}\n" + "=" * 72)
    out = analyse({w: np.array(v) for w, v in obs.items()}, pairs)
    report(out)
    v = verdict(out)
    print(f"\n  >>> {name}: {v} <<<")
    json.dump({"model": name, "analysis": out, "verdict": v}, open(f"{ROOT}/exp183_results_{name}.json", "w"), indent=1)
    print(f"results saved to exp183_results_{name}.json")


if __name__ == "__main__":
    print("exp183 — free generation (prereg frozen at befa567)")
    if not self_test():
        sys.exit(1)
    if len(sys.argv) < 2 or sys.argv[1] == "--selftest-only":
        sys.exit(0)
    main(sys.argv[1])
