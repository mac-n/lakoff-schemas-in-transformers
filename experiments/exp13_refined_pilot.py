"""
exp13_refined_pilot.py - refined capability pilot for asymmetric grounding.

exp12 lesson: comparing all-schema vs neutral mostly measures register, because
schema sentences (mixed across children) just look more literary than neutral
ones. The Lakoffian-distinctive test is WITHIN-SCHEMA across children: do
features fire on UP_LITERAL AND UP_MORE AND UP_HAPPY but NOT on neutral?

Asymmetric grounding score per feature:
    score = min(UP_LITERAL_mean, UP_MORE_mean, UP_HAPPY_mean) - NEUTRAL_mean

A feature with high positive score: fires on ALL THREE UP-children (the worst
of the three is still high) AND not on neutral. That's the signature of a
feature that's tracking UP-schema across cross-domain extensions, rather than
firing on a register confound or a single-domain feature.

Aggregation: MAX activation across token positions per sentence (not mean).
Peak signal is what we want; the schema features only fire on the schema-
relevant tokens, and averaging across all positions dilutes that.
"""

import json
import re
import time

import numpy as np
import requests
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

# ---- The four groups ----
UP_LITERAL = [
    "The hot air balloon ascended slowly into the morning sky.",
    "She climbed the ladder to reach the rooftop antenna.",
    "The kite hovered just below the gymnasium ceiling.",
    "Hikers reached the summit of the mountain just before sunset.",
    "The aeroplane climbed steadily after takeoff.",
    "A bird perched on the highest spire of the tower.",
    "He hoisted the heavy box up the stairs to the attic.",
    "The escalator carried them to the top floor of the building.",
    "They ascended the mountaintop trail in two hours.",
    "The cathedral's tower rose above all the surrounding rooftops.",
]

UP_MORE = [
    "The company's revenue grew by twenty percent last quarter.",
    "Interest accrued steadily on the savings account.",
    "Their inventory expanded after the new warehouse opened.",
    "Sales increased throughout the holiday season.",
    "The population of the city has multiplied since 1980.",
    "He augmented his income by taking a second contract.",
    "Production capacity increased to meet rising demand.",
    "Audience numbers grew with every new season.",
    "Subscription figures have been steadily increasing.",
    "Stock holdings accrued substantial value during the rally.",
]

UP_HAPPY = [
    "She was beaming after receiving the acceptance letter.",
    "The whole family felt ecstatic about the announcement.",
    "He returned from the trip looking cheerful and rested.",
    "Crowds were jubilant when their team won the championship.",
    "Her smile was radiant as she stepped onto the stage.",
    "The children were delighted by the surprise puppet show.",
    "He felt elated knowing his work had been recognized.",
    "Their faces were gleeful as they unwrapped the gifts.",
    "She was exultant after finishing her first marathon.",
    "The chef seemed joyful preparing the celebration meal.",
]

NEUTRAL = [
    "The conference room is on the third level of the office.",
    "Tuesday's agenda includes three main discussion topics.",
    "He bought fresh sourdough at the bakery this morning.",
    "Their new office has many large windows facing east.",
    "The film starts at eight o'clock tonight downtown.",
    "She enjoys quiet gardening on her weekends.",
    "The package arrived on a Friday afternoon last week.",
    "He plays violin in the local community orchestra.",
    "The bookstore on Main Street stays open until midnight.",
    "Yesterday's weather was unusually warm for autumn.",
    "Their cat has soft black and white fur.",
    "She studies cultural anthropology at the university.",
    "The restaurant serves Mediterranean cuisine on Tuesdays.",
    "His sister works as a graphic designer in the city.",
    "The concert took place in the central park yesterday.",
]

GROUPS = {
    "UP_LITERAL": UP_LITERAL,
    "UP_MORE": UP_MORE,
    "UP_HAPPY": UP_HAPPY,
    "NEUTRAL": NEUTRAL,
}

UP_GROUPS = ["UP_LITERAL", "UP_MORE", "UP_HAPPY"]

print(f"Sentence counts: {[(g, len(s)) for g, s in GROUPS.items()]}")

# ---- Device ----
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Load model ----
print("\nLoading Pythia 70m-deduped...")
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
model.eval()
n_layers = model.cfg.n_layers
print(f"  {n_layers} layers")


def collect_acts(text, hook):
    tokens = model.to_tokens(text)
    if tokens.shape[1] > 64:
        tokens = tokens[:, :64]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook)
    return cache[hook][0].cpu().float()  # (seq_len, d_model)


# ---- Per-layer analysis ----
results = {}
for layer in range(n_layers):
    hook = f"blocks.{layer}.hook_resid_post"
    print(f"\n--- Layer {layer} ---")

    sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=hook, device="cpu")
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res

    # Encode each sentence: take MAX activation per feature across token positions
    def encode_group(sentences):
        max_per_sentence = []  # list of (n_features,) MAX activations
        for s in sentences:
            acts = collect_acts(s, hook)  # (seq_len, d_model)
            with torch.no_grad():
                feat_acts = sae.encode(acts)  # (seq_len, n_features)
            max_per_sentence.append(feat_acts.max(0).values.numpy().astype(np.float64))
        return np.stack(max_per_sentence)

    group_acts = {g: encode_group(s) for g, s in GROUPS.items()}  # group -> (n_sent, n_feat)
    n_features = group_acts["NEUTRAL"].shape[1]

    # Per-feature mean (across sentences) per group
    group_means = {g: a.mean(0) for g, a in group_acts.items()}  # (n_features,) per group

    # Asymmetric grounding score: min over UP-children means, minus neutral mean
    up_stack = np.stack([group_means[g] for g in UP_GROUPS])  # (3, n_features)
    min_up = up_stack.min(0)
    asym_score = min_up - group_means["NEUTRAL"]

    # Also: mean-over-UP minus neutral (less stringent)
    mean_up_minus_neutral = up_stack.mean(0) - group_means["NEUTRAL"]

    # Top by asymmetric grounding score
    top_idx_asym = np.argsort(-asym_score)[:20]

    results[layer] = {
        "n_features": n_features,
        "group_means": group_means,
        "asym_score": asym_score,
        "mean_up_minus_neutral": mean_up_minus_neutral,
        "top_features_asym": [
            (int(i),
             float(asym_score[i]),
             float(group_means["UP_LITERAL"][i]),
             float(group_means["UP_MORE"][i]),
             float(group_means["UP_HAPPY"][i]),
             float(group_means["NEUTRAL"][i]))
            for i in top_idx_asym
        ],
        # How many features have positive asymmetric score (fire on all three UP > neutral)?
        "n_positive_asym": int((asym_score > 0).sum()),
        # Stronger threshold
        "n_strong_asym": int((asym_score > 0.5).sum()),
    }

    print(f"  features with positive asym score: {results[layer]['n_positive_asym']} / {n_features}")
    print(f"  features with strong (>0.5) asym score: {results[layer]['n_strong_asym']}")
    feat_idx, score, ul, um, uh, n = results[layer]["top_features_asym"][0]
    print(f"  TOP feature: idx={feat_idx}, asym_score={score:.3f}")
    print(f"    UP_LITERAL={ul:.3f}, UP_MORE={um:.3f}, UP_HAPPY={uh:.3f}, NEUTRAL={n:.3f}")

    del sae

del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---- Neuronpedia lookups ----
print("\n\nFetching Neuronpedia descriptions...")
to_fetch = set()
for layer in range(n_layers):
    np_sae = f"{layer}-res-sm"
    for feat_idx, *_ in results[layer]["top_features_asym"][:15]:
        to_fetch.add((np_sae, feat_idx))

print(f"Total features to lookup: {len(to_fetch)}")


def fetch_feature(np_sae, feat_idx):
    url = f"https://www.neuronpedia.org/api/feature/pythia-70m-deduped/{np_sae}/{feat_idx}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except json.JSONDecodeError:
            clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', r.text)
            try:
                return json.loads(clean, strict=False)
            except Exception:
                return None
    except Exception as e:
        return {"_error": str(e)}


feature_cache = {}
for i, (np_sae, feat_idx) in enumerate(sorted(to_fetch)):
    if i % 20 == 0:
        print(f"  {i}/{len(to_fetch)}")
    feature_cache[(np_sae, feat_idx)] = fetch_feature(np_sae, feat_idx)
    time.sleep(0.3)


def feature_desc(d):
    if d is None or "_error" in d:
        return "(no data)"
    expls = d.get("explanations") or []
    if expls:
        return expls[0].get("description", "").strip()[:180]
    return "(no auto-interp description)"


# ---- Write report ----
report_path = "/Users/macn/Documents/embeddingexp/results_exp13_refined_pilot.md"
with open(report_path, "w") as f:
    def out(s=""):
        print(s)
        f.write(s + "\n")

    out("# exp13 — Refined capability pilot: asymmetric grounding test")
    out()
    out(f"Sentences: 10 UP_LITERAL + 10 UP_MORE + 10 UP_HAPPY + 15 NEUTRAL.")
    out("Asymmetric grounding score = min(UP_LITERAL_mean, UP_MORE_mean, UP_HAPPY_mean) − NEUTRAL_mean.")
    out("Aggregation: MAX activation per feature across token positions per sentence.")
    out()
    out("Per layer: count of features with positive asym score, top 15 by score with descriptions.")
    out()

    out("## Summary across layers")
    out()
    out(f"  {'layer':>5}  {'n_pos_asym':>11}  {'n_strong_asym':>14}  {'top_asym_score':>15}")
    for layer in range(n_layers):
        r = results[layer]
        top_score = r["top_features_asym"][0][1]
        out(f"  {layer:>5d}  {r['n_positive_asym']:>11d}  {r['n_strong_asym']:>14d}  {top_score:>15.3f}")
    out()

    for layer in range(n_layers):
        r = results[layer]
        np_sae = f"{layer}-res-sm"
        out("-" * 100)
        out(f"## Layer {layer}  (positive asym features: {r['n_positive_asym']}, strong: {r['n_strong_asym']})")
        out("-" * 100)
        out()
        out("Top 15 features by asymmetric grounding score:")
        out(f"  {'feat':>6} {'score':>8} {'UP_LIT':>8} {'UP_MORE':>8} {'UP_HAPPY':>9} {'NEUTRAL':>8}  description")
        for feat_idx, score, ul, um, uh, n in r["top_features_asym"][:15]:
            d = feature_cache.get((np_sae, feat_idx))
            desc = feature_desc(d)
            out(f"  {feat_idx:>6d} {score:>+8.3f} {ul:>8.3f} {um:>8.3f} {uh:>9.3f} {n:>8.3f}  {desc}")
        out()

print(f"\nReport: {report_path}")

torch.save({
    "results": results,
    "groups": GROUPS,
    "feature_cache": {f"{k[0]}_{k[1]}": v for k, v in feature_cache.items()},
}, "/Users/macn/Documents/embeddingexp/exp13_results.pt")
