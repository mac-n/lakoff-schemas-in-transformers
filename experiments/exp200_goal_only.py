"""
exp200_goal_only.py — is "goal" a role the model gives to what an instruction
asks for, with no source in the sentence? Frozen rules: PREREG_exp200.md
(commit d8bad61, 3 Oct 2026). Written AFTER the freeze. Llama base and
instruct, CPU. Probe trained as exp199 (cue-free literal journeys, 29).
"""
import json
import re
import sys

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)
from exp199_instructions_spg import LITERAL, INSTRUCT as E199_INSTRUCT

N_SHUF = 300
SEED = 200
PRIMARY_LAYER = 6
LAYERS = [3, 6, 8, 10, 13]
PREREG = open(f"{ROOT}/PREREG_exp200.md").read()
for _n, _v in [("N_SHUF", N_SHUF), ("SEED", SEED), ("PRIMARY_LAYER", PRIMARY_LAYER)]:
    _m = re.search(rf"\b{_n} = ([0-9]+)", PREREG)
    assert _m and int(_m.group(1)) == _v, f"{_n} drifted from prereg"

GOAL_ONLY = [
    ("Write a haiku about rain.", "haiku"), ("Draw me a map of the village.", "map"), ("Compose a lullaby for a baby.", "lullaby"),
    ("Give me a riddle about time.", "riddle"), ("Write a limerick about a cat.", "limerick"), ("Build me a budget for the trip.", "budget"),
    ("Suggest a title for the book.", "title"), ("Write a toast for the wedding.", "toast"), ("Design a logo for the bakery.", "logo"),
    ("Invent a password for me.", "password"), ("Write a sonnet about autumn.", "sonnet"), ("Plan an itinerary for Rome.", "itinerary"),
    ("Give me a nickname for my dog.", "nickname"), ("Write a eulogy for my uncle.", "eulogy"), ("Prepare a schedule for the week.", "schedule"),
    ("Write a joke about plumbers.", "joke"), ("Make me a playlist for running.", "playlist"), ("Write an apology to my neighbour.", "apology"),
    ("Draft a contract for the sale.", "contract"), ("Write a motto for the team.", "motto"), ("Create a crossword about birds.", "crossword"),
    ("Write a fable about a fox.", "fable"), ("Give me a mnemonic for the planets.", "mnemonic"), ("Write a prayer for the harvest.", "prayer"),
    ("Propose a name for the startup.", "name"), ("Write a ballad about the sea.", "ballad"), ("Make a checklist for moving house.", "checklist"),
    ("Write a caption for this photo.", "caption"), ("Compose an anthem for the school.", "anthem"), ("Write a verdict on the case.", "verdict"),
    ("Give me a diagnosis for these symptoms.", "diagnosis"), ("Write a prophecy for the new year.", "prophecy"),
]
SOURCE_ONLY = [
    ("Read this letter carefully.", "letter"), ("Study the painting for a minute.", "painting"), ("Listen to the recording twice.", "recording"),
    ("Look at the photograph closely.", "photograph"), ("Examine the fossil under the lamp.", "fossil"), ("Consider the evidence before you.", "evidence"),
    ("Inspect the engine before we leave.", "engine"), ("Watch the footage from last night.", "footage"), ("Review the ledger from March.", "ledger"),
    ("Check the inventory this afternoon.", "inventory"), ("Skim the brochure on the desk.", "brochure"), ("Go over the testimony again.", "testimony"),
    ("Scan the receipts from the trip.", "receipts"), ("Observe the specimen for an hour.", "specimen"), ("Read the contract clause by clause.", "clause"),
    ("Analyse the sample in the fridge.", "sample"), ("Taste the soup before serving.", "soup"), ("Look over the floorplan on the wall.", "floorplan"),
    ("Browse the catalogue at your leisure.", "catalogue"), ("Study the scoreboard carefully.", "scoreboard"), ("Read the warning on the bottle.", "warning"),
    ("Examine the wound before bandaging it.", "wound"), ("Check the gauge every ten minutes.", "gauge"), ("Watch the tide for an hour.", "tide"),
    ("Consider the proposal overnight.", "proposal"), ("Listen to the witness without interrupting.", "witness"), ("Inspect the hull for cracks.", "hull"),
    ("Review the footnotes before the meeting.", "footnotes"), ("Study the timetable on page nine.", "timetable"), ("Read the inscription on the stone.", "inscription"),
    ("Look at the X-ray on the screen.", "X-ray"), ("Examine the signature closely.", "signature"),
]
assert len(GOAL_ONLY) == 32 and len(SOURCE_ONLY) == 32
_train_nouns = {w for _, a, b in LITERAL for w in (a, b)} | {w for _, a, b in E199_INSTRUCT for w in (a, b)}
for _, n in GOAL_ONLY + SOURCE_ONLY:
    assert n not in _train_nouns, n
NO_TASK = lambda items: [(f"The {n} was on the table yesterday.", n) for _, n in items]


def main(name):
    from transformer_lens import HookedTransformer
    tag = name.split("/")[-1]
    print(f"exp200 — is goal a role given to what an instruction asks for? ({tag}; prereg frozen at d8bad61)")
    if "Instruct" in name:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct", torch_dtype=torch.float32)
        model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B", hf_model=hf, device="cpu", dtype=torch.float32,
                                                  tokenizer=AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct"))
    else:
        model = HookedTransformer.from_pretrained(name, device="cpu", dtype=torch.float32)
    model.eval(); tk = model.tokenizer; BOS = int(tk.bos_token_id)
    hooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    rng = np.random.default_rng(SEED)

    def noun_states(items):
        """items: (sentence, noun) -> {L: [n, d]}"""
        X = {L: [] for L in LAYERS}
        for sent, noun in items:
            ids = [BOS] + tk.encode(sent, add_special_tokens=False)
            with torch.no_grad():
                _, cache = model.run_with_cache(torch.tensor([ids]), names_filter=hooks)
            pos = re.search(rf'\b{re.escape(noun)}\b', sent, re.I).start(); p = len(tk.encode(sent[: pos + len(noun)], add_special_tokens=False))
            assert tk.decode(ids[p]).strip().lower() in noun.lower(), (sent, noun, tk.decode(ids[p]))
            for L in LAYERS:
                X[L].append(cache[f"blocks.{L}.hook_resid_post"][0, p, :].float().numpy().astype(np.float64))
        return {L: np.stack(X[L]) for L in LAYERS}

    # training states: both nouns of each literal sentence
    pairs = [(s, a, 0) for s, a, b in LITERAL] + [(s, b, 1) for s, a, b in LITERAL]
    Xtr = noun_states([(s, n) for s, n, _ in pairs]); ytr = np.array([lab for _, _, lab in pairs])
    G, S = noun_states(GOAL_ONLY), noun_states(SOURCE_ONLY)
    Gc, Sc = noun_states(NO_TASK(GOAL_ONLY)), noun_states(NO_TASK(SOURCE_ONLY))

    def fit(X, y, lam=10.0):
        mu = X.mean(0); sd = X.std(0) + 1e-6; Z = (X - mu) / sd
        w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - 0.5))
        sc = lambda Xn: ((Xn - mu) / sd) @ w
        scale = sc(X).std() + 1e-9
        return lambda Xn: sc(Xn) / scale          # standardised projection; positive = goal side

    res = {}
    print(f"\n  {'layer':>5} {'goal-only shift':>16} {'frac>0':>7} {'source-only shift':>18} {'frac<0':>7} {'shuffle 5-95%':>15}")
    for L in LAYERS:
        sc = fit(Xtr[L], ytr)
        gs = sc(G[L]) - sc(Gc[L]); ss = sc(S[L]) - sc(Sc[L])
        shuf = []
        for _ in range(N_SHUF):
            scs = fit(Xtr[L], rng.permutation(ytr)); shuf.append(float((scs(G[L]) - scs(Gc[L])).mean()))
        p5, p95 = np.percentile(shuf, [5, 95])
        r = dict(goal_shift=float(gs.mean()), goal_frac=float((gs > 0).mean()), source_shift=float(ss.mean()), source_frac=float((ss < 0).mean()),
                 p5=float(p5), p95=float(p95))
        res[L] = r
        print(f"  {L:>5} {r['goal_shift']:>+16.3f} {r['goal_frac']:>7.2f} {r['source_shift']:>+18.3f} {r['source_frac']:>7.2f} {p5:>+7.3f}/{p95:<+7.3f}"
              f"{'   <- primary' if L == PRIMARY_LAYER else ''}", flush=True)
    r = res[PRIMARY_LAYER]
    g_ok = r["goal_shift"] > 0 and r["goal_shift"] > r["p95"]; s_ok = r["source_shift"] < r["p5"]
    v = "GOAL_IS_A_ROLE" if (g_ok and s_ok) else "GOAL_ONLY_HALF" if g_ok else "SOURCE_ONLY_HALF" if s_ok else "NOT_A_ROLE"
    print(f"\n  >>> {tag}, layer {PRIMARY_LAYER}: {v} (goal-only shift {r['goal_shift']:+.3f}, source-only shift {r['source_shift']:+.3f}, "
          f"shuffle [{r['p5']:+.3f}, {r['p95']:+.3f}]) <<<")
    json.dump({"model": name, "verdict": v, "layers": {str(k): v_ for k, v_ in res.items()}}, open(f"{ROOT}/exp200_results_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
