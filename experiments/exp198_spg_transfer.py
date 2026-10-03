"""
exp198_spg_transfer.py — source/goal role transfer, cue-disjoint sets, two
models. Frozen rules: PREREG_exp198.md (commit 5326a4c, 3 Oct 2026 02:26 IST).
Written AFTER the freeze. CPU. Item = (sentence, source noun, goal noun);
in every set, even-indexed items name the source first, odd-indexed the goal
first (order balanced 16/16).

Usage: ./lakoff/bin/python3 exp198_spg_transfer.py pythia-410m
       ./lakoff/bin/python3 exp198_spg_transfer.py gpt2-medium
"""
import json
import re
import sys

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)
from exp197_spg_pilot import LITERAL as PILOT_LIT, METAPHOR as PILOT_MET

N_SHUF = 300
SEED = 198
PRIMARY_LAYER = 8
LAYERS = [4, 8, 12, 16, 20]
PREREG = open(f"{ROOT}/PREREG_exp198.md").read()
for _n, _v in [("N_SHUF", N_SHUF), ("SEED", SEED), ("PRIMARY_LAYER", PRIMARY_LAYER)]:
    _m = re.search(rf"\b{_n} = ([0-9]+)", PREREG)
    assert _m and int(_m.group(1)) == _v, f"{_n} drifted from prereg"

# ---- LITERAL, cue-free (no from/to). Spatial verbs only. Even: source first; odd: goal first.
LITERAL_CF = [
    ("The walk began at the barn and ended at the river.", "barn", "river"),
    ("They reached the harbour, having set out at the farm.", "farm", "harbour"),
    ("The route starts at the station and finishes at the port.", "station", "port"),
    ("Their destination was the lake; their starting point was the cabin.", "cabin", "lake"),
    ("The race began at the park and ended at the library.", "park", "library"),
    ("The ship's destination was the dock, its origin the bay.", "bay", "dock"),
    ("The journey started at the oasis and finished at the fort.", "oasis", "fort"),
    ("The summit was where they ended; the village was where they began.", "village", "summit"),
    ("She left the mill behind and arrived at the church.", "mill", "church"),
    ("He arrived at the island after departing the mainland.", "mainland", "island"),
    ("The bus departed the market and pulled in at the hospital.", "market", "hospital"),
    ("The ridge lay ahead of them; the valley lay behind.", "valley", "ridge"),
    ("The stream rises on the mountain and empties into the sea.", "mountain", "sea"),
    ("The plane landed at the capital after taking off at the airfield.", "airfield", "capital"),
    ("The caravan departed the well at dawn and reached the fort by dusk.", "well", "fort"),
    ("He pulled up at the bridge, having ridden out of the inn.", "inn", "bridge"),
    ("The children set off at the school and arrived at the beach.", "school", "beach"),
    ("The ferry docked at the pier after casting off at the quay.", "quay", "pier"),
    ("The road leaves the mine and ends at the town.", "mine", "town"),
    ("We arrived at the hotel, having started out at the cathedral.", "cathedral", "hotel"),
    ("The pilgrims set out at the gate and knelt at the shrine.", "gate", "shrine"),
    ("The dog ended up in the garden, having bolted out of the kitchen.", "kitchen", "garden"),
    ("The path leaves the meadow and climbs to the cliff.", "meadow", "cliff"),
    ("The lighthouse was their destination; the cove was their origin.", "cove", "lighthouse"),
    ("The parade set off at the square and finished at the stadium.", "square", "stadium"),
    ("They came to rest at the chapel after leaving the barracks.", "barracks", "chapel"),
    ("The tunnel begins at the quarry and comes out at the depot.", "quarry", "depot"),
    ("The riders arrived at the ranch, having left the canyon at noon.", "canyon", "ranch"),
    ("The trail starts at the lodge and ends at the waterfall.", "lodge", "waterfall"),
    ("The pier was where the boat arrived; the marsh was where it set out.", "marsh", "pier"),
    ("The convoy departed the airport and halted at the embassy.", "airport", "embassy"),
    ("The tower was where the march ended; the prison was where it started.", "prison", "tower"),
]
# ---- ABSTRACT, cue-free, no cue word shared with LITERAL_CF. Half the goals negative. Even: source first.
ABSTRACT_CF = [
    ("His poverty slowly turned into wealth.", "poverty", "wealth"),
    ("Confidence grew out of her old doubt.", "doubt", "confidence"),
    ("Their friendship soured into rivalry.", "friendship", "rivalry"),
    ("Grief replaced the joy of the early years.", "joy", "grief"),
    ("The country's peace gave way to war.", "peace", "war"),
    ("Despair emerged where his hope had been.", "hope", "despair"),
    ("The company's loss became profit within a year.", "loss", "profit"),
    ("Calm eventually replaced her anger.", "anger", "calm"),
    ("Their suspicion deepened into hatred.", "suspicion", "hatred"),
    ("Recovery followed a long illness.", "illness", "recovery"),
    ("His boredom hardened into resentment.", "boredom", "resentment"),
    ("Success grew out of the chaos of the first year.", "chaos", "success"),
    ("The town's prosperity decayed into ruin.", "prosperity", "ruin"),
    ("Growth emerged out of the recession.", "recession", "growth"),
    ("Her innocence matured into wisdom.", "innocence", "wisdom"),
    ("Fluency came, in the end, out of years of silence.", "silence", "fluency"),
    ("The logic of the debate degenerated into insult.", "logic", "insult"),
    ("Order was built out of the disorder of the war years.", "disorder", "order"),
    ("Her health declined into frailty.", "health", "frailty"),
    ("Fame was what his obscurity eventually became.", "obscurity", "fame"),
    ("The trust between them curdled into betrayal.", "trust", "betrayal"),
    ("Dominance was what the firm's bankruptcy turned into.", "bankruptcy", "dominance"),
    ("Theory gradually gave way to practice.", "theory", "practice"),
    ("Victory was what grew out of that defeat.", "defeat", "victory"),
    ("His courage crumbled into cowardice.", "courage", "cowardice"),
    ("Skill was what emerged out of all that clumsiness.", "clumsiness", "skill"),
    ("The abundance of the harvest years dwindled into famine.", "abundance", "famine"),
    ("Clarity eventually came out of the confusion.", "confusion", "clarity"),
    ("Their freedom eroded into servitude.", "freedom", "servitude"),
    ("Comfort was what her hardship slowly became.", "hardship", "comfort"),
    ("The honesty of his youth decayed into deceit.", "honesty", "deceit"),
    ("Mastery grew, over years, out of ignorance.", "ignorance", "mastery"),
]
assert len(LITERAL_CF) == 32 and len(ABSTRACT_CF) == 32
_lit_words = set(" ".join(s for s, _, _ in LITERAL_CF).lower().replace(".", "").replace(";", "").replace(",", "").split())
_abs_words = set(" ".join(s for s, _, _ in ABSTRACT_CF).lower().replace(".", "").replace(";", "").replace(",", "").split())
SHARED = sorted((_lit_words & _abs_words) - {"the", "a", "of", "and", "was", "were", "at", "in", "into", "out", "his", "her", "their", "its",
                                              "they", "he", "she", "it", "what", "where", "had", "been", "that", "all", "years", "year", "first", "early", "old",
                                              "long", "slowly", "eventually", "gradually", "end", "ended", "between", "them", "over", "which", "by", "on", "to"})


def swapped(items):
    out = []
    for s, a, b in items:
        def rep(m, a=a, b=b):
            w = m.group(0); new = b if w.lower() == a else a
            return new.capitalize() if w[0].isupper() else new
        t = re.sub(rf"\b({a}|{b})\b", rep, s, flags=re.I)
        out.append((t, b, a))
    return out


def no_journey(items):
    out = []
    for k, (_, s, g) in enumerate(items):
        a, b = (s, g) if k % 2 == 0 else (g, s)
        out.append((f"The {a} and the {b} were both discussed at length.", s, g))
    return out


def main(name):
    from transformer_lens import HookedTransformer
    print(f"exp198 — source/goal role transfer, cue-disjoint ({name}; prereg frozen at 5326a4c)")
    print(f"  cue words shared between LITERAL-CF and ABSTRACT-CF after stop-words: {SHARED}")
    model = HookedTransformer.from_pretrained(name, device="cpu"); model.eval()
    tk = model.tokenizer; BOS = int(tk.bos_token_id)
    hooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    rng = np.random.default_rng(SEED)

    def states(items):
        X = {L: [] for L in LAYERS}; y = []
        for sent, s, g in items:
            ids = [BOS] + tk.encode(sent, add_special_tokens=False)
            with torch.no_grad():
                _, cache = model.run_with_cache(torch.tensor([ids]), names_filter=hooks)
            for noun, lab in ((s, 0), (g, 1)):
                pos = re.search(rf'\b{re.escape(noun)}\b', sent, re.I).start()
                p = len(tk.encode(sent[: pos + len(noun)], add_special_tokens=False))
                assert tk.decode(ids[p]).strip().lower() in noun.lower(), (sent, noun, tk.decode(ids[p]))
                for L in LAYERS:
                    X[L].append(cache[f"blocks.{L}.hook_resid_post"][0, p, :].numpy().astype(np.float64))
                y.append(lab)
        return {L: np.stack(X[L]) for L in LAYERS}, np.array(y)

    sets = {"LITERAL-CF": LITERAL_CF, "ABSTRACT-CF": ABSTRACT_CF, "SWAPPED": swapped(ABSTRACT_CF), "NO-JOURNEY": no_journey(ABSTRACT_CF),
            "LITERAL-FT": PILOT_LIT[:24], "ABSTRACT-FT": PILOT_MET[:24]}
    D = {k: states(v) for k, v in sets.items()}

    def fit(X, y, lam=10.0):
        mu = X.mean(0); sd = X.std(0) + 1e-6; Z = (X - mu) / sd
        w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - 0.5))
        return lambda Xn: (((Xn - mu) / sd) @ w > 0).astype(int)

    acc = lambda pr, k, L: float((pr(D[k][0][L]) == D[k][1]).mean())
    Xl, yl = D["LITERAL-CF"]
    res = {}
    print(f"\n  {'layer':>5} {'LOO lit':>8} {'ABSTRACT-CF':>12} {'SWAPPED':>8} {'NO-JOURNEY':>11} {'shuf 5-95%':>13} {'FT->FT':>7} {'CF->FT':>7}")
    for L in LAYERS:
        pr = fit(Xl[L], yl)
        loo = []
        for i in range(0, len(yl), 2):
            keep = np.ones(len(yl), bool); keep[i:i + 2] = False
            loo.append(float((fit(Xl[L][keep], yl[keep])(Xl[L][~keep]) == yl[~keep]).mean()))
        shuf = [float((fit(Xl[L], rng.permutation(yl))(D["ABSTRACT-CF"][0][L]) == D["ABSTRACT-CF"][1]).mean()) for _ in range(N_SHUF)]
        p5, p95 = np.percentile(shuf, [5, 95])
        prft = fit(D["LITERAL-FT"][0][L], D["LITERAL-FT"][1])
        r = dict(loo=float(np.mean(loo)), abstract=acc(pr, "ABSTRACT-CF", L), swapped=acc(pr, "SWAPPED", L), nojourney=acc(pr, "NO-JOURNEY", L),
                 p5=float(p5), p95=float(p95), ft_ft=acc(prft, "ABSTRACT-FT", L), cf_ft=acc(pr, "ABSTRACT-FT", L))
        res[L] = r
        print(f"  {L:>5} {r['loo']:>8.2f} {r['abstract']:>12.2f} {r['swapped']:>8.2f} {r['nojourney']:>11.2f} {p5:>6.2f}-{p95:<6.2f} {r['ft_ft']:>7.2f} {r['cf_ft']:>7.2f}"
              f"{'   <- primary' if L == PRIMARY_LAYER else ''}", flush=True)
    r = res[PRIMARY_LAYER]
    if r["abstract"] <= r["p95"]:
        v = "NO_TRANSFER"
    elif r["swapped"] > r["p95"] and r["p5"] <= r["nojourney"] <= r["p95"]:
        v = "TRANSFERS"
    else:
        v = "CUE_OR_NOUN"
    peak = max(res, key=lambda L: res[L]["abstract"])
    print(f"\n  >>> {name}, layer {PRIMARY_LAYER}: {v} (abstract {r['abstract']:.2f}, swapped {r['swapped']:.2f}, no-journey {r['nojourney']:.2f}, "
          f"shuffle 95th {r['p95']:.2f}); peak layer {peak} <<<")
    json.dump({"model": name, "verdict": v, "peak_layer": peak, "shared_cue_words": SHARED, "layers": {str(k): v_ for k, v_ in res.items()}},
              open(f"{ROOT}/exp198_results_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
