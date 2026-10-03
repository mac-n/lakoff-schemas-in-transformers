"""
exp191_suffix_clean.py — does the suffix finding survive a clean test?
Frozen rules: PREREG_exp191.md (commit 85b306c, 2026-10-02 22:01 IST).
Written AFTER the freeze. CPU (the GPU is busy with exp187 / exp190).

Pythia 410M: word states inside exp180's sentences, start token, single-token
pairs only, fragment-free axes, random-word null. GloVe: the same words,
pairs and lists with exp171's static strip (all-but-the-top, 3 components).
"""
import json
import re
import sys

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)
import exp180_paired_retry as e180            # frames (frozen stimuli)
import exp185_bundle_vs_default as e185       # extra single-token pool words
from exp178_what_produces_coupling import POS_EXT, NEG_EXT

MIN_PAIRS = 8
N_NULL = 300
SEED_NULL = 191
LAYERS = [4, 8, 12, 16, 20]
PRIMARY = 12
INFL = ["ER_comparative", "EST_superlative", "ING_progressive", "ED_past", "S_plural"]
DERIV = ["UN_negation", "RE_repetition"]
COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]
PREREG = open(f"{ROOT}/PREREG_exp191.md").read()
for _n, _v in [("MIN_PAIRS", MIN_PAIRS), ("N_NULL", N_NULL), ("SEED_NULL", SEED_NULL)]:
    _m = re.search(rf"\b{_n} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_v), f"{_n} drifted from prereg"


def analyse(vec, pairs, axes_def, pool, strip, label, layers_note=""):
    """vec: {word: vector}. Returns sink table and null for one substrate / layer."""
    mean_of = lambda ws: np.mean([vec[w] for w in ws if w in vec], axis=0)

    def axis(a, b):
        raw = mean_of(a) - mean_of(b)
        return strip(raw / np.linalg.norm(raw))

    axes = {n: axis(a, b) for n, (a, b) in axes_def.items()}
    sdir, npairs = {}, {}
    for suf, ps in pairs.items():
        diffs = [vec[i] / np.linalg.norm(vec[i]) - vec[b] / np.linalg.norm(vec[b]) for b, i in ps if b in vec and i in vec]
        npairs[suf] = len(diffs)
        if diffs:
            raw = np.mean(diffs, axis=0)
            sdir[suf] = strip(raw / np.linalg.norm(raw))
    table = {suf: {n: float(sdir[suf] @ ax) for n, ax in axes.items()} for suf in sdir}
    binding = [s for s in INFL if npairs.get(s, 0) >= MIN_PAIRS]
    rng = np.random.default_rng(SEED_NULL)
    pw = [w for w in pool if w in vec]
    null_mean, null_by = [], {s: [] for s in sdir}
    for _ in range(N_NULL):
        ws = list(rng.choice(pw, 49, replace=False))
        ax = axis(ws[:27], ws[27:])
        for s in sdir:
            null_by[s].append(float(sdir[s] @ ax))
        null_mean.append(float(np.mean([sdir[s] @ ax for s in binding])))
    null_mean = np.array(null_mean)
    sink = float(np.mean([table[s]["BALANCE"] for s in binding]))
    p5, p95 = np.percentile(null_mean, [5, 95])
    allneg = all(table[s]["BALANCE"] < 0 for s in binding)
    if sink < p5:
        v = "SINK_SURVIVES" if allneg else "SINK_WEAK"
    elif sink > p95:
        v = "SINK_REVERSED"
    else:
        v = "SINK_GONE"
    pct = lambda s, n: float((np.array(null_by[s]) < table[s][n]).mean() * 100)
    return dict(label=label, table=table, npairs=npairs, binding=binding, sink=sink, null_p5=float(p5),
                null_p95=float(p95), null_median=float(np.median(null_mean)),
                sink_pct=float((null_mean < sink).mean() * 100), verdict=v,
                ed_fb=table.get("ED_past", {}).get("FORWARD-BACK"), ed_fb_pct=pct("ED_past", "FORWARD-BACK") if "ED_past" in sdir else None,
                bal_pct={s: pct(s, "BALANCE") for s in sdir}, in_range=bool(p5 <= sink <= p95))


def show(r):
    print(f"\n--- {r['label']} ---")
    names = list(next(iter(r["table"].values())).keys())
    print(f"  {'suffix':16s} {'pairs':>5}  " + " ".join(f"{n[:9]:>9}" for n in names) + "   BALANCE percentile among random axes")
    for s in INFL + DERIV:
        if s in r["table"]:
            print(f"  {s:16s} {r['npairs'][s]:>5}  " + " ".join(f"{r['table'][s][n]:>+9.3f}" for n in names)
                  + f"   {r['bal_pct'][s]:.0f}th{'' if (s in r['binding'] or s in DERIV) else '  (non-binding)'}")
    print(f"  sink = mean of the inflectional suffixes on BALANCE: {r['sink']:+.3f} | random-word axes: median {r['null_median']:+.3f} "
          f"[5% {r['null_p5']:+.3f}, 95% {r['null_p95']:+.3f}]; at the {r['sink_pct']:.0f}th percentile -> {r['verdict']}")
    if r["ed_fb"] is not None:
        print(f"  -ed on FORWARD-BACK: {r['ed_fb']:+.3f} ({r['ed_fb_pct']:.0f}th percentile among random axes)")


def main():
    from transformers import AutoTokenizer
    print("exp191 — the suffix finding, clean (prereg frozen at 85b306c)")
    tk = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    one = lambda w: len(tk.encode(" " + w, add_special_tokens=False)) == 1
    src = open(f"{ROOT}/exp154_norm_confound_control.py").read()
    SP = eval(re.search(r"SUFFIX_PAIRS = (\{.*?\n\})", src, re.S).group(1))
    held = json.load(open(f"{ROOT}/exp171_pairs_heldout.json"))
    W = json.load(open(f"{ROOT}/exp175_stimuli.json"))["words"]
    axes_def = {"BALANCE": (POS_EXT, NEG_EXT), "UP-DOWN": (W["UP"], W["DOWN"])}
    for n, d in W["schemas"].items():
        if n != "BALANCE":
            axes_def[n] = (d["pos"], d["neg"])
    axis_words = {w for a, b in axes_def.values() for w in a + b}
    pairs = {}
    for suf in INFL + DERIV:
        ps = [tuple(p) for p in SP[suf]] + [tuple(p) for p in held.get(suf, [])]
        keep = []
        for b, i in ps:
            if one(b) and one(i) and b not in axis_words and i not in axis_words and (b, i) not in keep \
                    and b not in COMMON + RARE and i not in COMMON + RARE:
                keep.append((b, i))
        pairs[suf] = keep
    print("  single-token pairs after exclusions: " + ", ".join(f"{s.split('_')[0]} {len(v)}" for s, v in pairs.items()))
    pair_words = {w for ps in pairs.values() for p in ps for w in p}
    pool_src = set(W["VALPOS"] + W["VALNEG"]) | axis_words | {w for ps in list(e180.STIM["pairs"].values()) +
                                                               [e185.BUNDLE_NEW, e185.NOLINK_NEW] for p in ps for w in p}
    pool = sorted(w for w in pool_src if one(w) and w not in POS_EXT + NEG_EXT and w not in pair_words)
    words = sorted(pair_words | axis_words | set(pool) | set(COMMON) | set(RARE))
    print(f"  words: {len(words)}; null pool {len(pool)}")

    # ---- Pythia 410M states inside the sentences (prefix up to the word is all a causal model sees)
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("pythia-410m", device="cpu"); model.eval()
    BOS = int(tk.bos_token_id)
    hooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    heads = []
    for fr in e180.STIM["frames"]:
        ws = fr.split(" "); head = ws[: ws.index("{w}")]
        heads.append([BOS] + [t for x in head for t in tk.encode(" " + x, add_special_tokens=False)])
    S = {L: {} for L in LAYERS}
    for k, w in enumerate(words):
        wid = tk.encode(" " + w, add_special_tokens=False)
        acc = {L: 0.0 for L in LAYERS}
        for h in heads:
            with torch.no_grad():
                _, cache = model.run_with_cache(torch.tensor([h + wid]), names_filter=hooks)
            for L in LAYERS:
                acc[L] = acc[L] + cache[f"blocks.{L}.hook_resid_post"][0, -1, :].numpy().astype(np.float64)
        for L in LAYERS:
            S[L][w] = acc[L] / len(heads)
        if (k + 1) % 150 == 0:
            print(f"  Pythia word states {k + 1}/{len(words)}", flush=True)
    del model
    results = {}
    for L in LAYERS:
        vec = S[L]
        M = np.stack(list(vec.values())); aniso = M.mean(axis=0); aniso /= np.linalg.norm(aniso)
        fq = np.mean([vec[w] for w in COMMON], axis=0) - np.mean([vec[w] for w in RARE], axis=0); fq /= np.linalg.norm(fq)
        fo = fq - (fq @ aniso) * aniso; fo /= np.linalg.norm(fo)

        def strip(d, aniso=aniso, fo=fo):
            d = d - (d @ aniso) * aniso; d = d - (d @ fo) * fo
            return d / np.linalg.norm(d)
        results[f"pythia_L{L}"] = analyse(vec, pairs, axes_def, pool, strip, f"Pythia 410M, layer {L}{' (PRIMARY)' if L == PRIMARY else ''}")
        show(results[f"pythia_L{L}"])

    # ---- GloVe, same words, exp171's static strip (all-but-the-top, 3 components)
    import gensim.downloader as api
    print("\nLoading glove-wiki-gigaword-300 ...", flush=True)
    wv = api.load("glove-wiki-gigaword-300")
    gv = {w: np.asarray(wv[w], np.float64) for w in words if w in wv.key_to_index}
    print(f"  vocabulary hit {len(gv)}/{len(words)}")
    Mg = np.stack(list(gv.values())); mu = Mg.mean(axis=0)
    _, _, Vt = np.linalg.svd(Mg - mu, full_matrices=False); pcs = Vt[:3]; mun = mu / np.linalg.norm(mu)

    def strip_g(d):
        d = d - (d @ mun) * mun
        for pc in pcs:
            d = d - (d @ pc) * pc
        return d / np.linalg.norm(d)
    results["glove"] = analyse(gv, pairs, axes_def, pool, strip_g, "GloVe (static)")
    show(results["glove"])

    p = results[f"pythia_L{PRIMARY}"]; g = results["glove"]
    diss = "DISSOCIATION_SURVIVES" if (p["verdict"] == "SINK_SURVIVES" and g["in_range"]) else "DISSOCIATION_NOT_SUPPORTED"
    print("\n" + "=" * 76)
    print(f"  >>> Pythia layer {PRIMARY}: {p['verdict']} (sink {p['sink']:+.3f}, {p['sink_pct']:.0f}th percentile of random axes) <<<")
    print(f"  >>> GloVe: sink {g['sink']:+.3f}, {g['sink_pct']:.0f}th percentile of its random axes ({g['verdict']}) <<<")
    print(f"  >>> {diss} <<<")
    print("  Pythia by layer: " + "  ".join(f"L{L} {results[f'pythia_L{L}']['sink']:+.3f} ({results[f'pythia_L{L}']['sink_pct']:.0f}th, {results[f'pythia_L{L}']['verdict']})" for L in LAYERS))
    json.dump({"verdict": p["verdict"], "dissociation": diss, "results": results}, open(f"{ROOT}/exp191_results.json", "w"), indent=1)
    print("results saved to exp191_results.json")


if __name__ == "__main__":
    main()
