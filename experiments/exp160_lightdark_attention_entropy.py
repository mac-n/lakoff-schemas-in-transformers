"""
exp160_lightdark_attention_entropy.py — SECOND-EMBODIMENT candidate,
correlational stage (A). Does a token's position on the LIGHT-DARK axis
predict its attention entropy (peaked vs diffuse attention)?

Carrier story (the thing DIFFICULTY->load never had): exp4 found DARK
steering DEGRADES computation (arithmetic breaks, outputs collapse into
"maze of the world of the world..."). Candidate mechanism: darkness =
diffuse attention. exp160 tests it correlationally first; if it holds and
is concept-specific, exp161 pushes it causally (DARK-steer -> entropy up).

Physiology: attention entropy at a query position = mean over heads of
the Shannon entropy of its attention distribution over keys, NORMALISED
by log(n_keys) so it lives in [0,1] regardless of position. (Attention
is inherently sequence-level — the word-level d_norm protocol cannot
measure it; hence prompts, not isolated words.)

Concept: LIGHT-DARK schema axis, built from canonical word pairs in
residual space (aniso+freq stripped, exp138/154 protocol), then prompt
token UNIT residuals are projected onto it.

Lessons baked in:
  - CONFOUND CONTROL (the week's recurring lesson): attention entropy is
    dominated by query POSITION and residual NORM. Report raw corr AND
    PARTIAL corr(LD_proj, entropy | position, norm). The partial is the
    real test.
  - SPECIFICITY: partial corr for ALL 8 schema axes vs entropy. A real
    coupling is LIGHT-DARK-specific, not "every schema predicts entropy".
  - Generic diverse prompts (not stuffed with light/dark vocab) so the
    LIGHT-DARK projection varies naturally — avoids circularity.
  - Verdict logic pure + synthetic-tested before the run.
  - Cross-model: Pythia (primary) / GPT-2 / Llama.

PRE-REGISTRATION (2026-06-11, before running; this Claude):
  Committed prediction — deflationary, per the week's hard-won prior
  (5 mechanism stories dead). I predict NO robust coupling after the
  position+norm control: |partial r(LD, entropy)| < 0.15 at all decision
  layers, in all three models. Reasoning: attention entropy is governed
  by syntactic/positional structure, not the semantic LIGHT-DARK content
  of a token. BUT exp4's degradation is a real carrier, so if it survives
  against this prior it is strong evidence — which is why it's worth the
  committed bet.
  Point predictions:
    P1 raw corr may be nonzero (norm/position leakage) but PARTIAL
       |r| < 0.15 at decision layers.
    P2 if any partial coupling appears, it will NOT be LIGHT-DARK-
       specific (some other schema ranks top) — i.e. a generic
       norm/position residual, not embodiment.
  Decision rule (per model; Pythia primary; decision layers mid-depth):
    L1 partial |r| >= 0.20 at all decision layers AND LIGHT-DARK ranks
       top among 8 schemas at a majority: REAL correlational coupling.
       Build exp161 (causal DARK-steer -> entropy). My prior falsified.
    L2 partial |r| < 0.15 at all decision layers: no coupling. Attention
       entropy is the wrong physiology, or exp4's degradation is
       norm-mediated not attention-mediated (check: corr(DARK, norm)).
    L3 otherwise (intermediate, or non-specific): confounded/weak;
       refine before any causal follow-up.
"""

import gc
import os

import numpy as np
import torch
from huggingface_hub import get_token

os.environ["HF_TOKEN"] = get_token() or ""

from transformer_lens import HookedTransformer

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML
from markedness_norm_protocol import SCHEMA_NAMES, COMMON, RARE, corrf

MODELS = {
    "pythia-410m":   dict(repo="pythia-410m",            layers=[3, 8, 12, 16, 20], decision=[8, 12, 16]),
    "gpt2-medium":   dict(repo="gpt2-medium",            layers=[3, 8, 12, 16, 20], decision=[8, 12, 16]),
    "Llama-3.2-1B":  dict(repo="meta-llama/Llama-3.2-1B", layers=[2, 5, 8, 11, 13], decision=[5, 8, 11]),
}

PROMPTS = [
    "The room was quiet and the afternoon light fell across the floor.",
    "She could not understand the instructions no matter how she tried.",
    "He solved the equation quickly and moved on to the next page.",
    "A thick fog rolled in and the path ahead became impossible to see.",
    "The argument was clear, each step following plainly from the last.",
    "Everything felt muddled, the words blurring together on the screen.",
    "The child laughed and ran across the bright open field.",
    "Static filled the radio and the message was lost in the noise.",
    "We agreed on the plan and signed the papers before lunch.",
    "The meaning kept slipping away every time she reached for it.",
    "Sunlight streamed through the window and warmed the wooden desk.",
    "His thoughts were tangled and he could not find a place to begin.",
    "The recipe listed the steps in a simple and orderly way.",
    "Smoke obscured the exit and people stumbled in the dark hallway.",
    "The lecture made the whole subject suddenly obvious to everyone.",
    "Doubt crept in and the once-firm decision dissolved into confusion.",
    "The river was calm and the stones beneath were perfectly visible.",
    "A migraine pressed behind her eyes and the page swam out of focus.",
    "They mapped the route carefully and knew exactly where to turn.",
    "The contract was vague and no one could say what it required.",
    "Dawn broke and the valley emerged crisp and sharp below them.",
    "He mumbled an answer that nobody in the room could make sense of.",
    "The diagram explained the engine far better than the manual had.",
    "Shadows lengthened and the woods grew indistinct and threatening.",
    "She read the proof twice and saw at once why it was true.",
    "The crowd surged and the speaker's point was swallowed by the din.",
    "Cold rain fell steadily on the grey and featureless plain.",
    "A single clear note rang out and the whole melody fell into place.",
    "The data was inconsistent and the conclusion remained uncertain.",
    "Morning was clean and quiet and the coffee was hot on the table.",
    "His vision blurred at the edges and the street signs went unreadable.",
    "The teacher drew one line and the difficult idea became simple.",
    "Confusion spread through the office as the rumours multiplied.",
    "The lake mirrored the mountains in still and faultless detail.",
    "Everything he said contradicted what he had said the hour before.",
    "The lantern steadied and the cave walls came sharply into view.",
    "She lost the thread of the story somewhere in the third chapter.",
    "The answer was plain once the extra clutter had been cleared away.",
    "Murk and weed choked the pond until nothing could be seen below.",
    "The map was precise and every landmark stood out at a glance.",
]


def schema_words():
    s = set(COMMON + RARE)
    for sn in SCHEMA_NAMES:
        for p, n in LAKOFF_SCHEMAS_MML[sn]:
            s.add(p); s.add(n)
    return sorted(s)


def attn_entropy_per_query(pattern):
    # pattern: [n_heads, q, k]; normalised Shannon entropy per query (q>=1)
    nH, Q, _ = pattern.shape
    out = np.full(Q, np.nan)
    for q in range(1, Q):
        p = pattern[:, q, :q + 1]                  # [heads, keys]
        p = np.clip(p, 1e-12, 1.0)
        H = -(p * np.log(p)).sum(axis=1)           # per head
        out[q] = float(H.mean() / np.log(q + 1))   # normalise by log(n_keys)
    return out


def partial_corr(x, y, covars):
    """corr(x, y | covars). covars: list of arrays."""
    A = np.vstack(covars + [np.ones_like(x)]).T
    def resid(v):
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ coef
    return corrf(resid(np.asarray(x, float)), resid(np.asarray(y, float)))


# ---------------- verdict (pure, selftest-able) ----------------

def verdict160(summary, decision_layers, hi=0.20, lo=0.15):
    pr = [abs(summary[L]["partial_LD"]) for L in decision_layers]
    spec = sum(1 for L in decision_layers if summary[L]["ld_rank"] == 1)
    q_couple = all(p >= hi for p in pr)
    q_spec = spec > len(decision_layers) / 2
    q_null = all(p < lo for p in pr)
    if q_couple and q_spec:
        return "L1"
    if q_null:
        return "L2"
    return "L3"


def selftest_verdict160(dl=(8, 12, 16)):
    def fake(pr, rank):
        return {L: {"partial_LD": pr, "ld_rank": rank} for L in dl}
    assert verdict160(fake(0.30, 1), dl) == "L1"
    assert verdict160(fake(0.05, 3), dl) == "L2"
    assert verdict160(fake(0.30, 4), dl) == "L3"   # strong but not specific
    assert verdict160(fake(0.17, 1), dl) == "L3"   # intermediate
    print("selftest_verdict160: all branches fire correctly.")


print("Running verdict-logic selftest first (synthetic-test convention)...")
selftest_verdict160()

vocab = schema_words()
all_results = {}

for tag, cfg in MODELS.items():
    print(f"\n{'='*72}\n{tag}\n{'='*72}")
    model = HookedTransformer.from_pretrained(cfg["repo"], device="mps")
    model.eval()
    LAYERS = cfg["layers"]
    rhooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    phooks = [f"blocks.{L}.attn.hook_pattern" for L in LAYERS]

    # --- isolated-word residuals -> aniso/freq/schema directions ---
    wres = {}
    for w in vocab:
        with torch.no_grad():
            _, c = model.run_with_cache(model.to_tokens(w), names_filter=rhooks)
        wres[w] = {L: c[f"blocks.{L}.hook_resid_post"][0, -1, :].float().cpu().numpy()
                   for L in LAYERS}
    dirs = {}
    for L in LAYERS:
        arr = np.stack([wres[w][L] for w in vocab])
        aniso = arr.mean(0); aniso /= np.linalg.norm(aniso)
        fr = (np.mean([wres[w][L] for w in COMMON], 0)
              - np.mean([wres[w][L] for w in RARE], 0))
        fr /= np.linalg.norm(fr)
        fro = fr - (fr @ aniso) * aniso; fro /= np.linalg.norm(fro)

        def strip(d):
            d = d - (d @ aniso) * aniso
            d = d - (d @ fro) * fro
            return d / np.linalg.norm(d)

        sd = {}
        for sn in SCHEMA_NAMES:
            pairs = LAKOFF_SCHEMAS_MML[sn]
            pos = sorted(set(p[0] for p in pairs)); neg = sorted(set(p[1] for p in pairs))
            raw = np.mean([wres[w][L] for w in pos], 0) - np.mean([wres[w][L] for w in neg], 0)
            sd[sn] = strip(raw / np.linalg.norm(raw))
        dirs[L] = sd

    # --- prompts: collect per-token (schema projs, entropy, position, norm) ---
    acc = {L: dict(proj={sn: [] for sn in SCHEMA_NAMES}, ent=[], pos=[], norm=[])
           for L in LAYERS}
    for prompt in PROMPTS:
        toks = model.to_tokens(prompt)
        with torch.no_grad():
            _, c = model.run_with_cache(toks, names_filter=rhooks + phooks)
        for L in LAYERS:
            resid = c[f"blocks.{L}.hook_resid_post"][0].float().cpu().numpy()  # [pos, d]
            pat = c[f"blocks.{L}.attn.hook_pattern"][0].float().cpu().numpy()  # [head, q, k]
            ent = attn_entropy_per_query(pat)
            for q in range(1, resid.shape[0]):
                if np.isnan(ent[q]):
                    continue
                r = resid[q]; nrm = np.linalg.norm(r); u = r / nrm
                for sn in SCHEMA_NAMES:
                    acc[L]["proj"][sn].append(float(u @ dirs[L][sn]))
                acc[L]["ent"].append(ent[q]); acc[L]["pos"].append(q); acc[L]["norm"].append(nrm)

    # --- per-layer correlations ---
    summary = {}
    print(f"  {'L':>3} {'raw r(LD,ent)':>14} {'partial r(LD|pos,norm)':>24} "
          f"{'LD rank/8':>10} {'n_tok':>7}")
    for L in LAYERS:
        a = acc[L]
        pos = np.array(a["pos"], float); norm = np.array(a["norm"], float)
        ent = np.array(a["ent"], float)
        ld = np.array(a["proj"]["LIGHT-DARK"], float)
        raw = corrf(ld, ent)
        partials = {sn: partial_corr(np.array(a["proj"][sn], float), ent, [pos, norm])
                    for sn in SCHEMA_NAMES}
        pld = partials["LIGHT-DARK"]
        rank = 1 + sum(1 for sn in SCHEMA_NAMES if sn != "LIGHT-DARK"
                       and abs(partials[sn]) > abs(pld))
        summary[L] = dict(raw=raw, partial_LD=pld, ld_rank=rank, partials=partials,
                          n=len(ent))
        print(f"  {L:>3} {raw:>+14.3f} {pld:>+24.3f} {rank:>10} {len(ent):>7}")
    # specificity detail at decision layers
    for L in cfg["decision"]:
        ps = summary[L]["partials"]
        ranked = sorted(SCHEMA_NAMES, key=lambda s: -abs(ps[s]))
        print(f"  L{L} partial-r by schema: " +
              ", ".join(f"{s.split('-')[0][:5]} {ps[s]:+.2f}" for s in ranked))
    all_results[tag] = (summary, cfg["decision"])
    del model, wres, acc
    gc.collect(); torch.mps.empty_cache()

# ---------------- verdicts ----------------
print("\n" + "=" * 72)
print("VERDICT vs pre-registered decision rule (per model)")
print("=" * 72)
for tag, (summary, decision) in all_results.items():
    code = verdict160(summary, decision)
    pr = ", ".join(f"L{L}:{summary[L]['partial_LD']:+.2f}(rk{summary[L]['ld_rank']})"
                   for L in decision)
    print(f"  {tag:<14} {code}   [{pr}]")
print("\n  L1=real coupling (build causal exp161)  L2=no coupling  L3=confounded/weak")
print("  Primary model = pythia-410m.")
