"""
exp199_instructions_spg.py — do instructions use the source/goal roles?
Frozen rules: PREREG_exp199.md (commit 47e3112, 3 Oct 2026). Written AFTER
the freeze. Llama-3.2-1B base and instruct, CPU. Probe trained on exp198's
cue-free literal journeys (minus the three with into / out of), tested on
instructions. Even-indexed items name the source first, odd the goal first.
"""
import json
import re
import sys

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)
from exp198_spg_transfer import LITERAL_CF, swapped

N_SHUF = 300
SEED = 199
PRIMARY_LAYER = 6
LAYERS = [3, 6, 8, 10, 13]
PREREG = open(f"{ROOT}/PREREG_exp199.md").read()
for _n, _v in [("N_SHUF", N_SHUF), ("SEED", SEED), ("PRIMARY_LAYER", PRIMARY_LAYER)]:
    _m = re.search(rf"\b{_n} = ([0-9]+)", PREREG)
    assert _m and int(_m.group(1)) == _v, f"{_n} drifted from prereg"
LITERAL = [it for it in LITERAL_CF if " into " not in it[0] and " out of " not in it[0]]
assert len(LITERAL) == 29, len(LITERAL)

INSTRUCT = [
    ("Turn this draft into a summary.", "draft", "summary"),
    ("Write a report based on these notes.", "notes", "report"),
    ("Rewrite the outline as an essay.", "outline", "essay"),
    ("Produce a chart out of this table.", "table", "chart"),
    ("Convert the transcript into minutes.", "transcript", "minutes"),
    ("Give me a poem made out of this prose.", "prose", "poem"),
    ("Turn the list into a paragraph.", "list", "paragraph"),
    ("Create a story out of this script.", "script", "story"),
    ("Rewrite the code as pseudocode.", "code", "pseudocode"),
    ("Make a headline out of this article.", "article", "headline"),
    ("Turn these bullets into a speech.", "bullets", "speech"),
    ("Write a tweet summarising this abstract.", "abstract", "tweet"),
    ("Turn this recipe into a timeline.", "recipe", "timeline"),
    ("Produce a diagram based on the description.", "description", "diagram"),
    ("Rewrite the dialogue as a narrative.", "dialogue", "narrative"),
    ("Write lyrics based on this melody.", "melody", "lyrics"),
    ("Convert the spreadsheet into a database.", "spreadsheet", "database"),
    ("Draft an email out of these reminders.", "reminders", "email"),
    ("Turn the lecture into flashcards.", "lecture", "flashcards"),
    ("Produce a syllabus out of this textbook.", "textbook", "syllabus"),
    ("Rewrite the manual as a tutorial.", "manual", "tutorial"),
    ("Make a quiz out of this chapter.", "chapter", "quiz"),
    ("Turn the complaint into a request.", "complaint", "request"),
    ("Write a review based on these screenshots.", "screenshots", "review"),
    ("Convert the sketch into a blueprint.", "sketch", "blueprint"),
    ("Produce a forecast out of the data.", "data", "forecast"),
    ("Turn the interview into a profile.", "interview", "profile"),
    ("Write a slogan based on this manifesto.", "manifesto", "slogan"),
    ("Rewrite the theorem as an algorithm.", "theorem", "algorithm"),
    ("Make a poster out of this announcement.", "announcement", "poster"),
    ("Turn the memo into an agenda.", "memo", "agenda"),
    ("Produce a glossary out of the thesis.", "thesis", "glossary"),
]
assert len(INSTRUCT) == 32
NO_TASK = [(f"The {(s, g)[k % 2]} and the {(g, s)[k % 2]} are both attached.", s, g) for k, (_, s, g) in enumerate(INSTRUCT)]
CHAT_PREFIX = "<|start_header_id|>user<|end_header_id|>\n\n"
CHAT_SUFFIX = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def main(name):
    from transformer_lens import HookedTransformer
    tag = name.split("/")[-1]
    print(f"exp199 — do instructions use the source/goal roles? ({tag}; prereg frozen at 47e3112)")
    if "Instruct" in name:
        # the instruct weights are cached under unsloth/; the architecture is Llama-3.2-1B's, so TransformerLens
        # takes the base config and the instruct weights (same procedure exp181 used through HF transformers)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct", torch_dtype=torch.float32)
        model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B", hf_model=hf, device="cpu", dtype=torch.float32,
                                                  tokenizer=AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct"))
    else:
        model = HookedTransformer.from_pretrained(name, device="cpu", dtype=torch.float32)
    model.eval()
    tk = model.tokenizer; BOS = int(tk.bos_token_id)
    hooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    rng = np.random.default_rng(SEED)

    def states(items, prefix="", suffix=""):
        X = {L: [] for L in LAYERS}; y = []
        for sent, s, g in items:
            text = prefix + sent + suffix
            ids = [BOS] + tk.encode(text, add_special_tokens=False)
            with torch.no_grad():
                _, cache = model.run_with_cache(torch.tensor([ids]), names_filter=hooks)
            for noun, lab in ((s, 0), (g, 1)):
                pos = re.search(rf'\b{re.escape(noun)}\b', text, re.I).start()
                p = len(tk.encode(text[: pos + len(noun)], add_special_tokens=False))
                assert tk.decode(ids[p]).strip().lower() in noun.lower(), (sent, noun, tk.decode(ids[p]))
                for L in LAYERS:
                    X[L].append(cache[f"blocks.{L}.hook_resid_post"][0, p, :].float().numpy().astype(np.float64))
                y.append(lab)
        return {L: np.stack(X[L]) for L in LAYERS}, np.array(y)

    D = {"LITERAL": states(LITERAL), "INSTRUCT": states(INSTRUCT), "SWAPPED": states(swapped(INSTRUCT)), "NO-TASK": states(NO_TASK)}
    is_instruct = "Instruct" in name
    if is_instruct:
        D["INSTRUCT-CHAT"] = states(INSTRUCT, CHAT_PREFIX, CHAT_SUFFIX)

    def fit(X, y, lam=10.0):
        mu = X.mean(0); sd = X.std(0) + 1e-6; Z = (X - mu) / sd
        w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - 0.5))
        return lambda Xn: (((Xn - mu) / sd) @ w > 0).astype(int)

    acc = lambda pr, k, L: float((pr(D[k][0][L]) == D[k][1]).mean())
    Xl, yl = D["LITERAL"]; res = {}
    print(f"\n  {'layer':>5} {'LOO lit':>8} {'INSTRUCT':>9} {'SWAPPED':>8} {'NO-TASK':>8} {'shuf 5-95%':>13} {'chat':>6}")
    for L in LAYERS:
        pr = fit(Xl[L], yl)
        loo = []
        for i in range(0, len(yl), 2):
            keep = np.ones(len(yl), bool); keep[i:i + 2] = False
            loo.append(float((fit(Xl[L][keep], yl[keep])(Xl[L][~keep]) == yl[~keep]).mean()))
        shuf = [float((fit(Xl[L], rng.permutation(yl))(D["INSTRUCT"][0][L]) == D["INSTRUCT"][1]).mean()) for _ in range(N_SHUF)]
        p5, p95 = np.percentile(shuf, [5, 95])
        r = dict(loo=float(np.mean(loo)), instruct=acc(pr, "INSTRUCT", L), swapped=acc(pr, "SWAPPED", L), notask=acc(pr, "NO-TASK", L),
                 p5=float(p5), p95=float(p95), chat=acc(pr, "INSTRUCT-CHAT", L) if is_instruct else None)
        res[L] = r
        print(f"  {L:>5} {r['loo']:>8.2f} {r['instruct']:>9.2f} {r['swapped']:>8.2f} {r['notask']:>8.2f} {p5:>6.2f}-{p95:<6.2f} "
              f"{(f'{r[chr(99)+chr(104)+chr(97)+chr(116)]:.2f}' if is_instruct else '   -'):>6}{'   <- primary' if L == PRIMARY_LAYER else ''}", flush=True)
    r = res[PRIMARY_LAYER]
    if r["instruct"] <= r["p95"]:
        v = "NO_TRANSFER"
    elif r["swapped"] > r["p95"] and r["p5"] <= r["notask"] <= r["p95"]:
        v = "TRANSFERS"
    else:
        v = "CUE_OR_NOUN"
    peak = max(res, key=lambda L: res[L]["instruct"])
    print(f"\n  >>> {tag}, layer {PRIMARY_LAYER}: {v} (instruct {r['instruct']:.2f}, swapped {r['swapped']:.2f}, no-task {r['notask']:.2f}, "
          f"shuffle 95th {r['p95']:.2f}); peak layer {peak} <<<")
    json.dump({"model": name, "verdict": v, "peak_layer": peak, "layers": {str(k): v_ for k, v_ in res.items()}},
              open(f"{ROOT}/exp199_results_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
