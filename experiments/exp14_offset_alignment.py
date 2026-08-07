"""
exp14_offset_alignment.py - schemas as DIRECTIONS, not features.

The Lakoffian claim, properly stated for SAE feature space: image schemas are
INVARIANT RELATIONAL TRANSFORMATIONS across domains. UP isn't a feature; UP
is the offset that takes a baseline state to its upward-shifted counterpart,
and that offset is approximately the SAME VECTOR across temperature, mood,
prices, status, etc. (As king-man+woman=queen tells us "royal" is a direction
not a feature, the claim here is that "rising/lifting/increasing/promoting"
share a direction in SAE feature space.)

Niamh's key insight (2026-05-25): a schema is less of a feature than a
relationship between features. The right analysis is not which features fire
but whether the OFFSETS between paired states cluster in direction.

This connects to Li et al.'s 2024 "crystals" finding (analogy parallelograms
in SAE feature space) — they showed this for relational concepts like
country→capital and gender pairs but explicitly did NOT test image schemas.
That's the open lane.

Pipeline:
  1. Matched paired sentences per (schema, domain): each pair differs minimally
     in the schema-transformation (baseline ↔ upward-shifted).
  2. For each pair at each layer: compute SAE-feature-space offset Δ.
  3. Compute pairwise cosine alignment between Δs:
     - Within-schema cross-domain: UP-temp vs UP-mood vs UP-price etc.
       (the load-bearing Lakoffian test)
     - Within-schema within-domain: should be very high (sanity)
     - Cross-schema: UP vs DOWN should be antialigned (the inverse of UP)
     - Schema vs null: should be ~0 (the discriminative test)
     - Schema vs BEVERAGE-sham: should be much higher than BEVERAGE-sham vs
       itself if schemas are embodied transformations and beverage is just
       taxonomic
  4. PCA on stacked offsets per schema: does a dominant direction exist?
  5. Per-layer comparison: where in the model do schemas-as-directions live?
"""

import json
import time
import re
from collections import defaultdict

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

# ---- Matched pair corpus ----
# Each pair: (baseline_state_sentence, schema_transformed_sentence)
# Pairs are word-difference-minimal so the offset isolates the schema content.

PAIRS = {
    "UP": {
        # 5 domains, 5 pairs each = 25 UP pairs
        "temperature": [
            ("The temperature was constant throughout the day.",
             "The temperature rose throughout the day."),
            ("The thermometer reading held steady overnight.",
             "The thermometer reading climbed overnight."),
            ("Room temperature stayed the same all afternoon.",
             "Room temperature soared all afternoon."),
            ("The water temperature was even before boiling.",
             "The water temperature ascended past boiling."),
            ("The forecast showed stable temperatures this week.",
             "The forecast showed increasing temperatures this week."),
        ],
        "mood": [
            ("Her mood was neutral after the meeting.",
             "Her mood was elated after the meeting."),
            ("He felt nothing about the news.",
             "He felt jubilant about the news."),
            ("Their spirits were average that morning.",
             "Their spirits were radiant that morning."),
            ("She seemed unchanged by the praise.",
             "She seemed ecstatic from the praise."),
            ("The audience reacted plainly to the song.",
             "The audience reacted joyfully to the song."),
        ],
        "quantity": [
            ("The company's revenue was stable last quarter.",
             "The company's revenue grew last quarter."),
            ("Sales held steady through summer.",
             "Sales increased through summer."),
            ("The population stayed flat over the decade.",
             "The population multiplied over the decade."),
            ("Inventory remained constant this month.",
             "Inventory expanded this month."),
            ("Subscribers stayed the same all year.",
             "Subscribers accrued all year."),
        ],
        "status": [
            ("Her position was unchanged at the firm.",
             "Her position was promoted at the firm."),
            ("His reputation remained the same in the field.",
             "His reputation became prominent in the field."),
            ("She held the same rank for years.",
             "She held a distinguished rank for years."),
            ("His standing was ordinary among peers.",
             "His standing was esteemed among peers."),
            ("The professor's recognition was middling.",
             "The professor's recognition was prestigious."),
        ],
        "health": [
            ("His condition was stable yesterday.",
             "His condition was thriving yesterday."),
            ("Her vitality stayed even after the surgery.",
             "Her vitality returned vigorously after the surgery."),
            ("The patient remained unchanged.",
             "The patient was recuperating."),
            ("The plants looked the same in the garden.",
             "The plants looked robust in the garden."),
            ("His energy was average that week.",
             "His energy was vital that week."),
        ],
    },
    "DOWN": {
        "temperature": [
            ("The temperature was constant throughout the day.",
             "The temperature plunged throughout the day."),
            ("The thermometer reading held steady overnight.",
             "The thermometer reading plummeted overnight."),
            ("Room temperature stayed the same all afternoon.",
             "Room temperature sank all afternoon."),
            ("The water temperature was even before boiling.",
             "The water temperature descended past freezing."),
            ("The forecast showed stable temperatures this week.",
             "The forecast showed tumbling temperatures this week."),
        ],
        "mood": [
            ("Her mood was neutral after the meeting.",
             "Her mood was dejected after the meeting."),
            ("He felt nothing about the news.",
             "He felt morose about the news."),
            ("Their spirits were average that morning.",
             "Their spirits were forlorn that morning."),
            ("She seemed unchanged by the praise.",
             "She seemed glum despite the praise."),
            ("The audience reacted plainly to the song.",
             "The audience reacted somberly to the song."),
        ],
        "quantity": [
            ("The company's revenue was stable last quarter.",
             "The company's revenue declined last quarter."),
            ("Sales held steady through summer.",
             "Sales decreased through summer."),
            ("The population stayed flat over the decade.",
             "The population dwindled over the decade."),
            ("Inventory remained constant this month.",
             "Inventory shrank this month."),
            ("Subscribers stayed the same all year.",
             "Subscribers waned all year."),
        ],
        "status": [
            ("Her position was unchanged at the firm.",
             "Her position was demoted at the firm."),
            ("His reputation remained the same in the field.",
             "His reputation was disgraced in the field."),
            ("She held the same rank for years.",
             "She was ousted from her rank."),
            ("His standing was ordinary among peers.",
             "His standing was discredited among peers."),
            ("The professor's recognition was middling.",
             "The professor's recognition was dethroned."),
        ],
        "health": [
            ("His condition was stable yesterday.",
             "His condition was ailing yesterday."),
            ("Her vitality stayed even after the surgery.",
             "Her vitality deteriorated after the surgery."),
            ("The patient remained unchanged.",
             "The patient was languishing."),
            ("The plants looked the same in the garden.",
             "The plants looked sickly in the garden."),
            ("His energy was average that week.",
             "His energy was feeble that week."),
        ],
    },
    "BEVERAGE_sham": {
        # Taxonomic specifications: category → instance. Should NOT have shared
        # direction across "domains" (different beverage categories).
        "coffee_pair": [
            ("She ordered coffee at the cafe.",
             "She ordered espresso at the cafe."),
            ("They served coffee at the meeting.",
             "They served cappuccino at the meeting."),
            ("He drank coffee in the morning.",
             "He drank latte in the morning."),
            ("The waiter brought coffee.",
             "The waiter brought mocha."),
            ("She prefers coffee after dinner.",
             "She prefers americano after dinner."),
        ],
        "alcohol_pair": [
            ("He drank beer at the pub.",
             "He drank lager at the pub."),
            ("They opened beer for the guests.",
             "They opened bourbon for the guests."),
            ("She brought beer to the barbecue.",
             "She brought champagne to the barbecue."),
            ("The bartender poured beer.",
             "The bartender poured whiskey."),
            ("He prefers beer with dinner.",
             "He prefers wine with dinner."),
        ],
        "tea_pair": [
            ("They served tea at the meeting.",
             "They served matcha at the meeting."),
            ("She drank tea in the afternoon.",
             "She drank chai in the afternoon."),
            ("He ordered tea at the cafe.",
             "He ordered oolong at the cafe."),
            ("The host offered tea.",
             "The host offered kombucha."),
            ("She prefers tea before bed.",
             "She prefers rooibos before bed."),
        ],
    },
    "NULL": {
        # Random transformations: changes that aren't schema-shaped at all.
        # Should produce uncorrelated offsets.
        "random_pair": [
            ("The cat is gray.", "The dog is large."),
            ("He bought bread today.", "She studied physics today."),
            ("The film starts at eight.", "Their cat has soft fur."),
            ("She works downtown on Mondays.", "He plays violin on Mondays."),
            ("The bookstore opens late.", "The package arrived Friday."),
            ("Their office has windows.", "The restaurant serves food."),
            ("She studies anthropology.", "His sister works in design."),
            ("The concert was yesterday.", "The agenda includes topics."),
            ("Tuesday's meeting is short.", "Yesterday's weather was warm."),
            ("The bakery opens early.", "The library closes late."),
        ],
    },
}

# Flatten pairs with labels
flat_pairs = []  # list of (schema, domain, baseline, transformed)
for schema, domains in PAIRS.items():
    for domain, pairs in domains.items():
        for baseline, transformed in pairs:
            flat_pairs.append((schema, domain, baseline, transformed))

print(f"Total pairs: {len(flat_pairs)}")
print(f"  UP: {sum(1 for p in flat_pairs if p[0]=='UP')}")
print(f"  DOWN: {sum(1 for p in flat_pairs if p[0]=='DOWN')}")
print(f"  BEVERAGE: {sum(1 for p in flat_pairs if p[0]=='BEVERAGE_sham')}")
print(f"  NULL: {sum(1 for p in flat_pairs if p[0]=='NULL')}")

# ---- Device ----
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Load model ----
print("\nLoading Pythia 70m-deduped...")
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
model.eval()
n_layers = model.cfg.n_layers


def collect_max_acts_at_hook(text, hook):
    """Return SAE-encoded MAX activation across token positions, (n_features,)."""
    tokens = model.to_tokens(text)
    if tokens.shape[1] > 64:
        tokens = tokens[:, :64]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook)
    return cache[hook][0].cpu().float()  # (seq_len, d_model)


# ---- Per layer: encode all pairs, compute offset vectors ----
results = {}
for layer in range(n_layers):
    hook = f"blocks.{layer}.hook_resid_post"
    print(f"\n--- Layer {layer} ---")

    sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=hook, device="cpu")
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res

    offsets = []  # list of (schema, domain, pair_idx, Δ-vector)
    for pair_idx, (schema, domain, baseline, transformed) in enumerate(flat_pairs):
        b_acts = collect_max_acts_at_hook(baseline, hook)        # (seq, d_model)
        t_acts = collect_max_acts_at_hook(transformed, hook)
        with torch.no_grad():
            b_feat = sae.encode(b_acts).max(0).values.numpy().astype(np.float64)
            t_feat = sae.encode(t_acts).max(0).values.numpy().astype(np.float64)
        delta = t_feat - b_feat
        offsets.append((schema, domain, pair_idx, delta))

    n_features = offsets[0][3].shape[0]

    # Compute pairwise cosine similarities
    # Group offsets by schema
    by_schema = defaultdict(list)  # schema -> list of (domain, idx, delta)
    for s, d, i, v in offsets:
        by_schema[s].append((d, i, v))

    def cos(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # Within-schema pairwise alignment (all pairs of Δs within a schema)
    within_schema = {}
    for s, items in by_schema.items():
        sims = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                sims.append(cos(items[i][2], items[j][2]))
        within_schema[s] = {
            "mean": float(np.mean(sims)) if sims else 0.0,
            "median": float(np.median(sims)) if sims else 0.0,
            "n": len(sims),
            "all": sims,
        }

    # Cross-schema alignment: UP vs DOWN, UP vs NULL, UP vs BEVERAGE
    def cross_schema_sims(a_schema, b_schema):
        a_items = by_schema.get(a_schema, [])
        b_items = by_schema.get(b_schema, [])
        sims = []
        for _, _, av in a_items:
            for _, _, bv in b_items:
                sims.append(cos(av, bv))
        return sims

    cross = {}
    for sa, sb in [("UP", "DOWN"), ("UP", "NULL"), ("DOWN", "NULL"),
                   ("UP", "BEVERAGE_sham"), ("DOWN", "BEVERAGE_sham"),
                   ("BEVERAGE_sham", "NULL")]:
        sims = cross_schema_sims(sa, sb)
        cross[f"{sa}_vs_{sb}"] = {
            "mean": float(np.mean(sims)) if sims else 0.0,
            "median": float(np.median(sims)) if sims else 0.0,
        }

    # Within-domain alignment for UP only (sanity check: should be highest)
    up_by_domain = defaultdict(list)
    for d, i, v in by_schema["UP"]:
        up_by_domain[d].append(v)
    within_domain_up = []
    for d, vs in up_by_domain.items():
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                within_domain_up.append(cos(vs[i], vs[j]))
    within_domain_up_mean = float(np.mean(within_domain_up)) if within_domain_up else 0.0

    # Cross-domain UP alignment (pairs of Δs from DIFFERENT domains within UP)
    cross_domain_up = []
    domains = list(up_by_domain.keys())
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            for vi in up_by_domain[domains[i]]:
                for vj in up_by_domain[domains[j]]:
                    cross_domain_up.append(cos(vi, vj))
    cross_domain_up_mean = float(np.mean(cross_domain_up)) if cross_domain_up else 0.0

    # PCA on UP offsets: does a dominant direction exist?
    up_stack = np.stack([v for _, _, v in by_schema["UP"]])
    up_centered = up_stack - up_stack.mean(0, keepdims=True)
    _, S, _ = np.linalg.svd(up_centered, full_matrices=False)
    var_ratio = (S ** 2) / (S ** 2).sum()
    pc1_var_ratio = float(var_ratio[0])
    pc12_var_ratio = float(var_ratio[:2].sum())

    # Same for NULL pairs (control)
    null_stack = np.stack([v for _, _, v in by_schema["NULL"]])
    null_centered = null_stack - null_stack.mean(0, keepdims=True)
    _, S_null, _ = np.linalg.svd(null_centered, full_matrices=False)
    null_var_ratio = (S_null ** 2) / (S_null ** 2).sum()
    null_pc1_var = float(null_var_ratio[0])

    # Same for BEVERAGE
    bev_stack = np.stack([v for _, _, v in by_schema["BEVERAGE_sham"]])
    bev_centered = bev_stack - bev_stack.mean(0, keepdims=True)
    _, S_bev, _ = np.linalg.svd(bev_centered, full_matrices=False)
    bev_var_ratio = (S_bev ** 2) / (S_bev ** 2).sum()
    bev_pc1_var = float(bev_var_ratio[0])

    results[layer] = {
        "within_schema": within_schema,
        "cross_schema": cross,
        "within_domain_up_mean": within_domain_up_mean,
        "cross_domain_up_mean": cross_domain_up_mean,
        "pc1_var_up": pc1_var_ratio,
        "pc12_var_up": pc12_var_ratio,
        "pc1_var_null": null_pc1_var,
        "pc1_var_bev": bev_pc1_var,
        "n_features": n_features,
    }

    print(f"  Within-schema mean cosine alignment:")
    for s, stats in within_schema.items():
        print(f"    {s}: mean={stats['mean']:.4f}, median={stats['median']:.4f}, n={stats['n']}")
    print(f"  Within-domain UP mean cos: {within_domain_up_mean:.4f}")
    print(f"  Cross-domain UP mean cos:  {cross_domain_up_mean:.4f}")
    print(f"  Cross-schema:")
    for k, v in cross.items():
        print(f"    {k}: mean={v['mean']:+.4f}")
    print(f"  PC1 var of UP offsets:   {pc1_var_ratio:.4f}")
    print(f"  PC1 var of NULL offsets: {null_pc1_var:.4f}")
    print(f"  PC1 var of BEV offsets:  {bev_pc1_var:.4f}")
    print(f"  → If UP-PC1 >> NULL-PC1 and >> BEV-PC1, UP-as-direction exists")

    del sae

del model

# ---- Summary report ----
report_path = "/Users/macn/Documents/embeddingexp/results_exp14_offset_alignment.md"
with open(report_path, "w") as f:
    def out(s=""):
        print(s)
        f.write(s + "\n")

    out("# exp14 — Schemas as directions: offset-vector alignment in SAE feature space")
    out()
    out("**The Lakoffian claim restated:** image schemas are invariant relational")
    out("transformations across domains. UP isn't a feature; UP is the offset that")
    out("takes a baseline state to its upward-shifted counterpart, and that offset")
    out("is approximately the same vector across temperature, mood, prices, status,")
    out("health. (Like king−man+woman=queen: 'royal' is a direction, not a feature.)")
    out()
    out(f"Pairs: 25 UP (5 domains × 5 pairs), 25 DOWN (5 domains × 5 pairs),")
    out(f"15 BEVERAGE sham (3 categories × 5 pairs, taxonomic), 10 NULL.")
    out("Aggregation: MAX SAE activation per feature across token positions.")
    out("Offset Δ = activation(transformed) − activation(baseline) per pair.")
    out()
    out("## Summary across layers")
    out()
    out(f"  {'layer':>5} {'UP_within':>10} {'DOWN_within':>11} {'BEV_within':>10} {'NULL_within':>11} "
        f"{'UP_cross_domain':>15} {'UP_PC1':>7} {'NULL_PC1':>9} {'BEV_PC1':>8} {'UP_vs_DOWN':>11}")
    for layer in range(n_layers):
        r = results[layer]
        out(f"  {layer:>5d} "
            f"{r['within_schema']['UP']['mean']:>10.4f} "
            f"{r['within_schema']['DOWN']['mean']:>11.4f} "
            f"{r['within_schema']['BEVERAGE_sham']['mean']:>10.4f} "
            f"{r['within_schema']['NULL']['mean']:>11.4f} "
            f"{r['cross_domain_up_mean']:>15.4f} "
            f"{r['pc1_var_up']:>7.3f} "
            f"{r['pc1_var_null']:>9.3f} "
            f"{r['pc1_var_bev']:>8.3f} "
            f"{r['cross_schema']['UP_vs_DOWN']['mean']:>+11.4f}")
    out()
    out("## Interpretation guide")
    out()
    out("- **UP_within > NULL_within**: UP pairs align more than random pairs → schema-as-direction exists.")
    out("- **UP_within > BEV_within**: UP (embodied schema) aligns more than BEVERAGE (taxonomy) → schemas ≠ taxonomic categories.")
    out("- **UP_cross_domain ≈ UP_within**: alignment is *cross-domain* (the Lakoffian invariance), not just within-domain.")
    out("- **UP_PC1 >> NULL_PC1**: UP offsets have a dominant principal component (a 'UP direction') that random offsets lack.")
    out("- **UP_vs_DOWN < 0**: UP and DOWN are antialigned (DOWN ≈ −UP in feature space).")
    out("- **Layer pattern**: where do schemas-as-directions live?")

print(f"\nReport: {report_path}")

torch.save({
    "results": results,
    "pairs": flat_pairs,
}, "/Users/macn/Documents/embeddingexp/exp14_results.pt")
