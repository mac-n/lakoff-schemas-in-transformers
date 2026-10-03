"""
exp197_spg_pilot.py — does a SOURCE/GOAL role probe trained on literal journeys
transfer to metaphorical ones? PILOT; rule in PILOT_exp197_source_path_goal.md
(commit 68e29db). Pythia 410M on CPU. Each item: (sentence, source noun, goal noun).
Order balanced: in every set, half the sentences name the goal first.
"""
import json
import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
LAYERS = [4, 8, 12, 16, 20]
PRIMARY = 12
N_SHUF = 300
SEED = 197

LITERAL = [
    ("She walked from the barn to the river.", "barn", "river"),
    ("He drove to the coast from the city.", "city", "coast"),
    ("The hikers left the village and reached the summit.", "village", "summit"),
    ("They reached the harbour after leaving the farm.", "farm", "harbour"),
    ("The train ran from the station to the port.", "station", "port"),
    ("We headed to the lake, leaving the cabin behind.", "cabin", "lake"),
    ("The letter travelled from the castle to the abbey.", "castle", "abbey"),
    ("She arrived at the church having set out from the mill.", "mill", "church"),
    ("The boys ran from the school to the beach.", "school", "beach"),
    ("He sailed to the island from the mainland.", "mainland", "island"),
    ("The bus went from the market to the hospital.", "market", "hospital"),
    ("They climbed to the ridge from the valley.", "valley", "ridge"),
    ("The river flows from the mountain to the sea.", "mountain", "sea"),
    ("She flew to the capital from the airfield.", "airfield", "capital"),
    ("The caravan crossed from the oasis to the fort.", "oasis", "fort"),
    ("He rode to the bridge from the inn.", "inn", "bridge"),
    ("The children walked from the park to the library.", "park", "library"),
    ("The ship steamed to the dock from the bay.", "bay", "dock"),
    ("The road leads from the mine to the town.", "mine", "town"),
    ("We cycled to the beach from the hotel.", "hotel", "beach"),
    ("The pilgrims went from the gate to the shrine.", "gate", "shrine"),
    ("The dog ran to the garden from the kitchen.", "kitchen", "garden"),
    ("The path winds from the meadow to the cliff.", "meadow", "cliff"),
    ("She moved to the city from the farm.", "farm", "city"),
    # cue-free (no from/to)
    ("The walk began at the barn and ended at the river.", "barn", "river"),
    ("They finished at the harbour, having started at the farm.", "farm", "harbour"),
    ("The route starts at the station and finishes at the port.", "station", "port"),
    ("Their destination was the lake; their starting point was the cabin.", "cabin", "lake"),
    ("The race began at the park and ended at the library.", "park", "library"),
    ("The ship's destination was the dock, its origin the bay.", "bay", "dock"),
    ("The journey started at the oasis and finished at the fort.", "oasis", "fort"),
    ("The summit was where they ended; the village was where they began.", "village", "summit"),
]
METAPHOR = [
    ("He went from poverty to wealth.", "poverty", "wealth"),
    ("She moved to confidence from doubt.", "doubt", "confidence"),
    ("The argument runs from the premise to the conclusion.", "premise", "conclusion"),
    ("They came to agreement from conflict.", "conflict", "agreement"),
    ("The country went from war to peace.", "war", "peace"),
    ("He drifted to despair from hope.", "hope", "despair"),
    ("The company moved from loss to profit.", "loss", "profit"),
    ("She turned to calm from anger.", "anger", "calm"),
    ("The talks moved from suspicion to trust.", "suspicion", "trust"),
    ("The patient went to recovery from illness.", "illness", "recovery"),
    ("His mood shifted from boredom to excitement.", "boredom", "excitement"),
    ("The plan moved to success from chaos.", "chaos", "success"),
    ("The discussion wandered from politics to religion.", "politics", "religion"),
    ("The economy moved to growth from recession.", "recession", "growth"),
    ("The story moves from innocence to experience.", "innocence", "experience"),
    ("They progressed to fluency from silence.", "silence", "fluency"),
    ("The debate went from logic to insult.", "logic", "insult"),
    ("The town moved to prosperity from ruin.", "ruin", "prosperity"),
    ("She went from student to teacher.", "student", "teacher"),
    ("He rose to fame from obscurity.", "obscurity", "fame"),
    ("The mood went from joy to grief.", "joy", "grief"),
    ("The firm climbed to dominance from bankruptcy.", "bankruptcy", "dominance"),
    ("His thinking moved from theory to practice.", "theory", "practice"),
    ("The team went to victory from defeat.", "defeat", "victory"),
    # cue-free
    ("Her career began in obscurity and ended in fame.", "obscurity", "fame"),
    ("The war ended in peace, though it began in anger.", "anger", "peace"),
    ("The lesson started with confusion and finished with clarity.", "confusion", "clarity"),
    ("The treaty brought trust where there had been suspicion.", "suspicion", "trust"),
    ("The talks began with hostility and ended with friendship.", "hostility", "friendship"),
    ("Recovery eventually followed the illness.", "illness", "recovery"),
    ("Years of practice turned clumsiness into skill.", "clumsiness", "skill"),
    ("Poverty slowly gave way to comfort.", "poverty", "comfort"),
]
CUEFREE_FROM = 24


def control_items(items, rng):
    out = []
    for k, (_, s, g) in enumerate(items):
        a, b = (s, g) if k % 2 == 0 else (g, s)
        out.append((f"The {a} and the {b} were both mentioned in the report.", s, g))
    return out


def main():
    from transformer_lens import HookedTransformer
    print("exp197 PILOT — source/goal role transfer (rule frozen at 68e29db)")
    model = HookedTransformer.from_pretrained("pythia-410m", device="cpu"); model.eval()
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
                # last token of the noun's first occurrence in the sentence (lowercase match)
                low = sent.lower(); pos = low.index(noun.lower())
                prefix = sent[:pos + len(noun)]
                p = len(tk.encode(prefix, add_special_tokens=False))      # index of noun's last token (+1 for BOS, -1 for 0-index)
                for L in LAYERS:
                    X[L].append(cache[f"blocks.{L}.hook_resid_post"][0, p, :].numpy().astype(np.float64))
                y.append(lab)
        return {L: np.stack(X[L]) for L in LAYERS}, np.array(y)

    Xl, yl = states(LITERAL); Xm, ym = states(METAPHOR)
    Xc, yc = states(control_items(LITERAL + METAPHOR, rng))
    cue_free = np.array([i >= CUEFREE_FROM for i in range(len(METAPHOR)) for _ in (0, 1)])

    def fit(X, y, lam=10.0):
        mu = X.mean(0); sd = X.std(0) + 1e-6; Z = (X - mu) / sd
        w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - 0.5))
        return lambda Xn: (((Xn - mu) / sd) @ w > 0).astype(int)

    results = {}
    print(f"\n  {'layer':>5}  {'train fit':>9}  {'metaphor':>8}  {'cue-free':>8}  {'noun ctrl':>9}  {'shuffle 95%':>11}")
    for L in LAYERS:
        pr = fit(Xl[L], yl)
        acc_train = float((pr(Xl[L]) == yl).mean()); pm = pr(Xm[L])
        acc_m = float((pm == ym).mean()); acc_cf = float((pm[cue_free] == ym[cue_free]).mean())
        acc_c = float((pr(Xc[L]) == yc).mean())
        shuf = []
        for _ in range(N_SHUF):
            ys = rng.permutation(yl); shuf.append(float((fit(Xl[L], ys)(Xm[L]) == ym).mean()))
        p95 = float(np.percentile(shuf, 95))
        results[L] = dict(train=acc_train, metaphor=acc_m, cue_free=acc_cf, control=acc_c, shuffle_p95=p95, shuffle_median=float(np.median(shuf)))
        print(f"  {L:>5}  {acc_train:>9.2f}  {acc_m:>8.2f}  {acc_cf:>8.2f}  {acc_c:>9.2f}  {p95:>11.2f}{'   <- primary' if L == PRIMARY else ''}")
    r = results[PRIMARY]
    if r["metaphor"] > r["shuffle_p95"]:
        v = "NOUN_LEAK" if r["control"] > r["shuffle_p95"] else "TRANSFERS"
    else:
        v = "NO_TRANSFER"
    print(f"\n  >>> layer {PRIMARY}: {v} (metaphor {r['metaphor']:.2f}, control {r['control']:.2f}, shuffle 95th {r['shuffle_p95']:.2f}) <<<")
    json.dump({"verdict": v, "results": {str(k): v_ for k, v_ in results.items()}}, open(f"{ROOT}/exp197_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
