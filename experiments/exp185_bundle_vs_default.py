"""exp185_bundle_vs_default.py — the UP bundle, or the default member of any
pair? Frozen rules: PREREG_exp185.md (commit 0b2c8ad, 2026-10-02 19:55 IST).
Written AFTER the freeze. Measure, frames and statistics are exp180's,
imported. Old pairs use exp180's saved values; new words are run in exp180's
frames with exp180's saved layer averages.

Usage:  ./lakoff/bin/python3 exp185_bundle_vs_default.py pythia-410m
        ./lakoff/bin/python3 exp185_bundle_vs_default.py gpt2-medium
"""
import json, os, re, sys
import numpy as np
ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)
import exp175_holding_up as e175
import exp180_paired_retry as e180

ALPHA, N_PERM, N_BOOT, N_LAYERS = 0.05, 10000, 2000, 24
P = lambda s: [p.split("/") for p in s.split()]
BUNDLE_OLD = P("big/small large/little long/short wide/narrow fast/slow hot/cold thick/thin strong/weak full/empty rich/poor loud/quiet bright/dark many/few more/less major/minor plus/minus")
NOLINK_OLD = P("near/far front/back forward/backward inside/outside before/after in/out early/late open/closed wet/dry")
BUNDLE_NEW = P("huge/tiny maximum/minimum increase/decrease gain/loss grow/shrink add/subtract winner/loser superior/inferior senior/junior awake/asleep alive/dead healthy/sick active/passive master/slave majority/minority abundant/scarce")
NOLINK_NEW = P("here/there this/that now/then come/go give/take push/pull buy/sell left/right north/south east/west male/female day/night land/sea start/finish begin/end send/receive read/write question/answer cause/effect input/output")
PREREG = " ".join(open(f"{ROOT}/PREREG_exp185.md").read().split())
for lst in (BUNDLE_OLD, NOLINK_OLD, BUNDLE_NEW, NOLINK_NEW):
    assert " ".join("/".join(p) for p in lst) in PREREG, "pair list drifted from prereg"
assert (len(BUNDLE_OLD), len(NOLINK_OLD), len(BUNDLE_NEW), len(NOLINK_NEW)) == (16, 9, 16, 20)
assert e180.ALPHA == ALPHA and e180.N_PERM == N_PERM and e180.N_BOOT == N_BOOT


def holds(r):
    return r["mean"] > 0 and r["p"] < ALPHA and r["intercept_ci"][0] > 0


def main(name):
    from wordfreq import zipf_frequency
    old = json.load(open(f"{ROOT}/exp180_results_{name}.json"))
    value, zipf = dict(old["values"]), dict(old["zipf"])
    new_words = sorted({w for p in BUNDLE_NEW + NOLINK_NEW for w in p})
    assert not (set(new_words) & set(value))
    run = e180.Runner2(name)
    torch = run.torch
    frames = e180.STIM["frames"]
    assert all(len(run.tk.encode(" " + w, add_special_tokens=False)) == 1 for w in new_words)
    mu = np.load(f"{ROOT}/exp180_cache_mu_{name}.npz")["mu"]
    mu_list = [torch.tensor(mu[L], dtype=torch.float32, device=run.dev) for L in range(N_LAYERS)]

    def frame_text(w, fr):
        ws = fr.split(" "); slot = ws.index("{w}"); ws[slot] = w
        g = run.groups(ws)
        return [t for x in g for t in x], sum(len(x) for x in g[: slot + 1])
    if not e175.mechanics_checks(run, [frame_text(w, frames[i]) for i, w in enumerate(["the", "and", "with", "that"])], mu_list):
        print("STOP: the drop mechanics are wrong."); sys.exit(2)
    # sanity: one old word must reproduce its exp180 value under the same layer averages
    chk = []
    for fr in frames:
        ids, p = frame_text("big", fr)
        toks, resid, logp = run.clean(ids)
        chk.append(np.log(run.holes(toks, logp, mu_list, [p])[1][0]))
    drift = abs(float(np.mean(chk)) - value["big"]["reach"])
    print(f"  'big' recomputed against exp180's saved value: difference {drift:.2e} -> {'ok' if drift < 1e-3 else 'FAIL'}")
    if drift >= 1e-3:
        print("STOP: not on the same footing as exp180."); sys.exit(2)
    path = f"{ROOT}/exp185_rows_{name}.jsonl"
    rows = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line); rows[(r["word"], r["frame"])] = r
    with open(path, "a") as fh:
        for w in new_words:
            for fi, fr in enumerate(frames):
                if (w, fi) in rows:
                    continue
                ids, p = frame_text(w, fr)
                toks, resid, logp = run.clean(ids)
                e_next, e_reach = run.holes(toks, logp, mu_list, [p])
                r = {"word": w, "frame": fi, "E_next": float(e_next[0]), "E_reach": float(e_reach[0])}
                fh.write(json.dumps(r) + "\n"); fh.flush(); rows[(w, fi)] = r
    nf = len(frames)
    for w in new_words:
        value[w] = {"reach": float(np.mean([np.log(rows[(w, f)]["E_reach"]) for f in range(nf)])),
                    "next": float(np.mean([np.log(rows[(w, f)]["E_next"]) for f in range(nf)]))}
        zipf[w] = zipf_frequency(w, "en")
    print("\n" + "=" * 72 + f"\nRESULTS — {name}\n" + "=" * 72)
    out = e180.analyse(value, zipf, {"vertical": BUNDLE_OLD + BUNDLE_NEW, "polar": NOLINK_OLD + NOLINK_NEW}, "reach", seed=185)
    B, N, S = out["vertical"], out["polar"], out["spec"]
    for label, r in (("BUNDLE ", B), ("NO-LINK", N)):
        print(f"  {label} {r['n']} pairs: first minus second {r['mean']:+.3f}  d_z {r['d_z']:+.2f}  p {r['p']:.4f}  "
              f"({r['n_positive']}/{r['n']} positive) | at equal frequency {r['intercept']:+.3f} "
              f"CI[{r['intercept_ci'][0]:+.3f},{r['intercept_ci'][1]:+.3f}] -> {'HOLDS' if holds(r) else 'does not hold'}")
    print(f"  BUNDLE minus NO-LINK: {S['diff']:+.3f}  p {S['p']:.4f}")
    sub = e180.analyse(value, zipf, {"vertical": BUNDLE_OLD, "polar": NOLINK_OLD, "b_new": BUNDLE_NEW, "n_new": NOLINK_NEW}, "reach", seed=1850)
    print("  by source (descriptive): " + " | ".join(f"{k} {sub[key]['mean']:+.3f} (p {sub[key]['p']:.3f}, {sub[key]['n_positive']}/{sub[key]['n']})"
          for k, key in (("BUNDLE old", "vertical"), ("NO-LINK old", "polar"), ("BUNDLE new", "b_new"), ("NO-LINK new", "n_new"))))
    if holds(B) and holds(N):
        v = "ANY_DEFAULT"
    elif holds(B) and S["diff"] > 0 and S["p"] < ALPHA:
        v = "UP_BUNDLE"
    else:
        v = "UNCLEAR"
    print(f"\n  >>> {name}: {v} <<<")
    pp = lambda ps: {"/".join(p): round(value[p[0]]["reach"] - value[p[1]]["reach"], 3) for p in ps}
    json.dump({"model": name, "bundle": B, "nolink": N, "spec": S, "verdict": v,
               "pair_diffs": {"bundle": pp(BUNDLE_OLD + BUNDLE_NEW), "nolink": pp(NOLINK_OLD + NOLINK_NEW)}},
              open(f"{ROOT}/exp185_results_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
