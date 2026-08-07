"""
exp12_capability_pilot.py - Pythia 70m capability sanity check.

Before investing in a 240-sentence corpus + the full deployment-pattern
analysis, verify the model can at least show DIFFERENTIAL activation
between schema-invoking sentences and neutral controls.

If yes → proceed with full corpus. If no → reconsider model size or
methodology before committing more sentence-writing time.

Method:
  - 15 sentences with clear schema invocation (spans UP and IN children
    for breadth — UP_LITERAL, UP_HAPPY, UP_MORE, UP_STATUS, IN_LITERAL, IN_MIND)
  - 15 neutral controls, matched roughly for length and grammar
  - Run through Pythia 70m, encode through res-sm SAE at EVERY layer
  - Per sentence: mean SAE activation across all token positions
    (65536-dim vector)
  - Per feature, compute mean(schema) - mean(neutral) = differential
  - Top features by |differential| at each layer, lookup via Neuronpedia
  - Verdict on capability: do top differential features have
    plausibly schema-relevant descriptions?
"""

import json
import re
import time

import numpy as np
import requests
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

# ---- The two groups ----
SCHEMA_SENTENCES = [
    # UP_LITERAL
    "The hot air balloon ascended high above the trees.",
    "She climbed the ladder to reach the top shelf.",
    "Smoke from the chimney soared into the morning sky.",
    # UP_HAPPY
    "Her mood was jubilant after she heard the good news.",
    "He felt ecstatic when the results finally came in.",
    "The whole team was elated by the unexpected victory.",
    # UP_MORE
    "The temperature increased steadily throughout the afternoon.",
    "Sales grew significantly during the third quarter.",
    "The company's revenue augmented year over year.",
    # UP_STATUS
    "She was promoted to senior partner last month.",
    "The professor's reputation became prominent in the field.",
    # IN_LITERAL
    "The wine was sealed in the bottle for years.",
    "The puppy was safely contained in its crate.",
    # IN_MIND
    "He pondered the difficult question for hours.",
    "She contemplated her future in silence.",
]

NEUTRAL_SENTENCES = [
    "The conference room is on the third floor.",
    "Tuesday's agenda includes three main topics.",
    "He bought fresh bread at the bakery this morning.",
    "Their new office has large windows facing east.",
    "The film starts at eight o'clock tonight.",
    "She enjoys gardening on her weekends.",
    "The package arrived on Friday afternoon.",
    "He plays violin in the local orchestra.",
    "The bookstore on Main Street stays open late.",
    "Yesterday's weather was unusually warm for autumn.",
    "Their cat has black and white fur.",
    "She studies anthropology at the university.",
    "The restaurant serves Mediterranean food on Tuesdays.",
    "His sister works as a graphic designer downtown.",
    "The concert was held in the city park yesterday.",
]

print(f"Schema sentences: {len(SCHEMA_SENTENCES)}")
print(f"Neutral sentences: {len(NEUTRAL_SENTENCES)}")

# ---- Device ----
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Load model ----
print("\nLoading Pythia 70m-deduped...")
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
model.eval()
n_layers = model.cfg.n_layers
print(f"  {n_layers} layers")


def collect_acts_for_sentence(text, hook_name):
    """Return residual stream activations at all token positions for one sentence."""
    tokens = model.to_tokens(text)
    if tokens.shape[1] > 64:
        tokens = tokens[:, :64]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
    return cache[hook_name][0].cpu().float()  # (seq_len, d_model)


# ---- Process each layer ----
results = {}
for layer in range(n_layers):
    hook = f"blocks.{layer}.hook_resid_post"
    sae_id = hook
    print(f"\n--- Layer {layer} ---")

    # Load SAE
    sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=sae_id, device="cpu")
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res

    # Collect activations + SAE-encode per sentence
    def encode_group(sentences):
        """Return list of (n_features,) per-sentence mean SAE activations."""
        per_sentence_mean_acts = []
        for s in sentences:
            acts = collect_acts_for_sentence(s, hook)  # (seq_len, d_model)
            with torch.no_grad():
                feat_acts = sae.encode(acts)  # (seq_len, n_features)
            per_sentence_mean_acts.append(feat_acts.mean(0).numpy().astype(np.float64))
        return np.stack(per_sentence_mean_acts)

    schema_acts = encode_group(SCHEMA_SENTENCES)        # (n_schema, n_features)
    neutral_acts = encode_group(NEUTRAL_SENTENCES)      # (n_neutral, n_features)

    n_features = schema_acts.shape[1]

    # Differential per feature
    schema_mean = schema_acts.mean(0)
    neutral_mean = neutral_acts.mean(0)
    diff = schema_mean - neutral_mean
    abs_diff = np.abs(diff)

    # Also compute t-stat-ish: diff / pooled-std (signal-to-noise)
    pooled_std = np.sqrt((schema_acts.var(0) + neutral_acts.var(0)) / 2 + 1e-8)
    tstat = diff / pooled_std

    # Top by absolute differential
    top_idx_diff = np.argsort(-abs_diff)[:15]
    top_idx_t = np.argsort(-np.abs(tstat))[:15]

    results[layer] = {
        "schema_mean_overall": float(schema_mean.mean()),
        "neutral_mean_overall": float(neutral_mean.mean()),
        "top_diff_features": [(int(i), float(diff[i]), float(abs_diff[i]), float(tstat[i])) for i in top_idx_diff],
        "top_tstat_features": [(int(i), float(diff[i]), float(tstat[i])) for i in top_idx_t],
        "n_features": n_features,
        "frac_features_schema_higher": float((diff > 0).mean()),
        "frac_features_neutral_higher": float((diff < 0).mean()),
    }
    print(f"  schema_mean={schema_mean.mean():.4f}, neutral_mean={neutral_mean.mean():.4f}")
    print(f"  fraction features schema>neutral: {(diff > 0).mean():.3f}")
    print(f"  top |diff| feature: idx={top_idx_diff[0]}, diff={diff[top_idx_diff[0]]:+.4f}, tstat={tstat[top_idx_diff[0]]:+.3f}")
    print(f"  top |tstat| feature: idx={top_idx_t[0]}, diff={diff[top_idx_t[0]]:+.4f}, tstat={tstat[top_idx_t[0]]:+.3f}")

    del sae

del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---- Lookup top features per layer on Neuronpedia ----
print("\n\nFetching Neuronpedia descriptions for top differential features...")


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


to_fetch = set()
for layer, r in results.items():
    np_sae = f"{layer}-res-sm"
    for feat_idx, _, _, _ in r["top_diff_features"][:10]:
        to_fetch.add((np_sae, feat_idx))
    for feat_idx, _, _ in r["top_tstat_features"][:10]:
        to_fetch.add((np_sae, feat_idx))

print(f"Total unique features to lookup: {len(to_fetch)}")
feature_cache = {}
for i, (np_sae, feat_idx) in enumerate(sorted(to_fetch)):
    if i % 20 == 0:
        print(f"  {i}/{len(to_fetch)}")
    feature_cache[(np_sae, feat_idx)] = fetch_feature(np_sae, feat_idx)
    time.sleep(0.3)


def feature_summary(d):
    if d is None or "_error" in d:
        return "(no data)"
    expls = d.get("explanations") or []
    if expls:
        return expls[0].get("description", "").strip()[:150]
    return "(no description)"


# ---- Write report ----
report_path = "/Users/macn/Documents/embeddingexp/results_exp12_capability_pilot.md"
with open(report_path, "w") as f:
    def out(s=""):
        print(s)
        f.write(s + "\n")

    out("# exp12 — Pythia 70m capability sanity check")
    out()
    out(f"Schema sentences: {len(SCHEMA_SENTENCES)}, neutral controls: {len(NEUTRAL_SENTENCES)}")
    out("Per layer, computing mean SAE-feature activation across token positions per sentence,")
    out("then per-feature mean differences between schema and neutral groups.")
    out()

    for layer in range(n_layers):
        r = results[layer]
        out(f"## Layer {layer} (res-sm)")
        out(f"- schema-group mean activation across all features: {r['schema_mean_overall']:.4f}")
        out(f"- neutral-group mean activation: {r['neutral_mean_overall']:.4f}")
        out(f"- fraction of features schema>neutral: {r['frac_features_schema_higher']:.3f}")
        out()
        out("### Top features by |mean activation difference|")
        for feat_idx, d, ad, t in r["top_diff_features"][:10]:
            np_sae = f"{layer}-res-sm"
            desc = feature_summary(feature_cache.get((np_sae, feat_idx)))
            arrow = "→ schema" if d > 0 else "→ neutral"
            out(f"- feat {feat_idx}: diff={d:+.4f} (tstat={t:+.2f}) {arrow}")
            out(f"  *{desc}*")
        out()
        out("### Top features by |t-statistic| (effect size adjusted)")
        for feat_idx, d, t in r["top_tstat_features"][:10]:
            np_sae = f"{layer}-res-sm"
            desc = feature_summary(feature_cache.get((np_sae, feat_idx)))
            arrow = "→ schema" if d > 0 else "→ neutral"
            out(f"- feat {feat_idx}: diff={d:+.4f} (tstat={t:+.2f}) {arrow}")
            out(f"  *{desc}*")
        out()

print(f"\nReport: {report_path}")

# Save raw results
torch.save({
    "results_per_layer": results,
    "schema_sentences": SCHEMA_SENTENCES,
    "neutral_sentences": NEUTRAL_SENTENCES,
    "feature_cache": {f"{k[0]}_{k[1]}": v for k, v in feature_cache.items()},
}, "/Users/macn/Documents/embeddingexp/exp12_results.pt")
print("Raw results saved to exp12_results.pt")
