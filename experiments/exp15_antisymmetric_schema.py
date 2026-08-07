"""
exp15_antisymmetric_schema.py - isolate the antisymmetric (UP − DOWN) component.

Niamh's insight after exp14: UP and DOWN offsets are positively correlated
(+0.06) because both contain a large shared "state-change happened here"
component that drowns out the actual schema polarity. If we decompose:

    UP_offset   = A · schema_axis + B · common_axis + noise
    DOWN_offset = −A · schema_axis + B · common_axis + noise

then UP − DOWN cancels the common_axis and isolates the schema:

    UP − DOWN = 2A · schema_axis + noise

This is exactly the contrast that should reveal schemas-as-directions if
they're there. exp14's matched-pair design (same baseline for UP and DOWN
per domain/pair_idx) makes this decomposition straightforward.

Same sentences as exp14. Per (domain, pair_idx), compute schema_offset.
Test:
  - Pairwise cosine alignment of schema_offsets ACROSS domains.
    If schemas are domain-invariant, this should be high. Much higher
    than raw UP-alignment or DOWN-alignment.
  - PC1 of stacked schema_offsets: dominant direction?
  - Compare to common_offsets (UP + DOWN): what does the SHARED axis look
    like? Is it interpretable as valence/change?
  - Compare to NULL antisymmetric: should be ~0 (random pairs don't have
    schema structure even after antisymmetrization)
"""

import json
import re
from collections import defaultdict

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

# ---- Pairs (same structure as exp14: matched baseline shared by UP and DOWN per domain/idx) ----
# Each domain has 5 (baseline, UP, DOWN) triples.
DOMAINS = {
    "temperature": [
        ("The temperature was constant throughout the day.",
         "The temperature rose throughout the day.",
         "The temperature plunged throughout the day."),
        ("The thermometer reading held steady overnight.",
         "The thermometer reading climbed overnight.",
         "The thermometer reading plummeted overnight."),
        ("Room temperature stayed the same all afternoon.",
         "Room temperature soared all afternoon.",
         "Room temperature sank all afternoon."),
        ("The water temperature was even before boiling.",
         "The water temperature ascended past boiling.",
         "The water temperature descended past freezing."),
        ("The forecast showed stable temperatures this week.",
         "The forecast showed increasing temperatures this week.",
         "The forecast showed tumbling temperatures this week."),
    ],
    "mood": [
        ("Her mood was neutral after the meeting.",
         "Her mood was elated after the meeting.",
         "Her mood was dejected after the meeting."),
        ("He felt nothing about the news.",
         "He felt jubilant about the news.",
         "He felt morose about the news."),
        ("Their spirits were average that morning.",
         "Their spirits were radiant that morning.",
         "Their spirits were forlorn that morning."),
        ("She seemed unchanged by the praise.",
         "She seemed ecstatic from the praise.",
         "She seemed glum despite the praise."),
        ("The audience reacted plainly to the song.",
         "The audience reacted joyfully to the song.",
         "The audience reacted somberly to the song."),
    ],
    "quantity": [
        ("The company's revenue was stable last quarter.",
         "The company's revenue grew last quarter.",
         "The company's revenue declined last quarter."),
        ("Sales held steady through summer.",
         "Sales increased through summer.",
         "Sales decreased through summer."),
        ("The population stayed flat over the decade.",
         "The population multiplied over the decade.",
         "The population dwindled over the decade."),
        ("Inventory remained constant this month.",
         "Inventory expanded this month.",
         "Inventory shrank this month."),
        ("Subscribers stayed the same all year.",
         "Subscribers accrued all year.",
         "Subscribers waned all year."),
    ],
    "status": [
        ("Her position was unchanged at the firm.",
         "Her position was promoted at the firm.",
         "Her position was demoted at the firm."),
        ("His reputation remained the same in the field.",
         "His reputation became prominent in the field.",
         "His reputation was disgraced in the field."),
        ("She held the same rank for years.",
         "She held a distinguished rank for years.",
         "She was ousted from her rank."),
        ("His standing was ordinary among peers.",
         "His standing was esteemed among peers.",
         "His standing was discredited among peers."),
        ("The professor's recognition was middling.",
         "The professor's recognition was prestigious.",
         "The professor's recognition was dethroned."),
    ],
    "health": [
        ("His condition was stable yesterday.",
         "His condition was thriving yesterday.",
         "His condition was ailing yesterday."),
        ("Her vitality stayed even after the surgery.",
         "Her vitality returned vigorously after the surgery.",
         "Her vitality deteriorated after the surgery."),
        ("The patient remained unchanged.",
         "The patient was recuperating.",
         "The patient was languishing."),
        ("The plants looked the same in the garden.",
         "The plants looked robust in the garden.",
         "The plants looked sickly in the garden."),
        ("His energy was average that week.",
         "His energy was vital that week.",
         "His energy was feeble that week."),
    ],
}

# Null triples: random sentence triples where the "UP" and "DOWN" are arbitrary
# changes, not schema-shaped. Antisymmetrizing these should give random noise.
NULL_TRIPLES = [
    ("The cat is on the chair.",
     "The dog is on the table.",
     "The bird is on the branch."),
    ("She bought bread today.",
     "She bought milk today.",
     "She bought eggs today."),
    ("The film starts at eight.",
     "The film starts at nine.",
     "The film starts at ten."),
    ("He plays violin in the orchestra.",
     "He plays cello in the orchestra.",
     "He plays viola in the orchestra."),
    ("The bookstore opens at nine.",
     "The bookstore opens at ten.",
     "The bookstore opens at eight."),
    ("Tuesday's meeting is short.",
     "Wednesday's meeting is short.",
     "Thursday's meeting is short."),
    ("The bakery sells croissants.",
     "The bakery sells baguettes.",
     "The bakery sells brioche."),
]

# Flatten
all_triples = []  # list of (group, domain, pair_idx, baseline, A, B)
for domain, triples in DOMAINS.items():
    for i, (b, u, d) in enumerate(triples):
        all_triples.append(("SCHEMA", domain, i, b, u, d))
for i, (b, x, y) in enumerate(NULL_TRIPLES):
    all_triples.append(("NULL", "null", i, b, x, y))

print(f"Schema triples: {sum(1 for t in all_triples if t[0]=='SCHEMA')}")
print(f"Null triples: {sum(1 for t in all_triples if t[0]=='NULL')}")

# ---- Device ----
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Load model ----
print("\nLoading Pythia 70m-deduped...")
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
model.eval()
n_layers = model.cfg.n_layers


def collect_acts(text, hook):
    tokens = model.to_tokens(text)
    if tokens.shape[1] > 64:
        tokens = tokens[:, :64]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook)
    return cache[hook][0].cpu().float()  # (seq_len, d_model)


def cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---- Per layer analysis ----
results = {}
for layer in range(n_layers):
    hook = f"blocks.{layer}.hook_resid_post"
    print(f"\n--- Layer {layer} ---")

    sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=hook, device="cpu")
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res

    # Encode each sentence to SAE feature space (max across token positions)
    def enc(text):
        acts = collect_acts(text, hook)
        with torch.no_grad():
            return sae.encode(acts).max(0).values.numpy().astype(np.float64)

    # Build offset triples per (group, domain, pair_idx)
    up_offsets = defaultdict(list)        # group -> list of (domain, idx, vector)
    down_offsets = defaultdict(list)
    schema_offsets = defaultdict(list)    # UP - DOWN
    common_offsets = defaultdict(list)    # UP + DOWN

    for group, domain, idx, baseline, a, b in all_triples:
        b_vec = enc(baseline)
        a_vec = enc(a)
        b_vec_diff = a_vec - b_vec               # A side offset
        c_vec = enc(b)
        c_vec_diff = c_vec - b_vec               # B side offset
        up_offsets[group].append((domain, idx, b_vec_diff))
        down_offsets[group].append((domain, idx, c_vec_diff))
        schema_offsets[group].append((domain, idx, b_vec_diff - c_vec_diff))
        common_offsets[group].append((domain, idx, b_vec_diff + c_vec_diff))

    # Compute pairwise alignment across domains for SCHEMA group
    def pairwise_cross_domain(items):
        """Cosine sims between items from DIFFERENT domains."""
        sims = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][0] != items[j][0]:  # different domain
                    sims.append(cos(items[i][2], items[j][2]))
        return sims

    def pairwise_same_domain(items):
        """Cosine sims between items from THE SAME domain."""
        sims = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][0] == items[j][0]:
                    sims.append(cos(items[i][2], items[j][2]))
        return sims

    # For SCHEMA group
    up_cross = pairwise_cross_domain(up_offsets["SCHEMA"])
    down_cross = pairwise_cross_domain(down_offsets["SCHEMA"])
    schema_cross = pairwise_cross_domain(schema_offsets["SCHEMA"])
    common_cross = pairwise_cross_domain(common_offsets["SCHEMA"])
    schema_same_domain = pairwise_same_domain(schema_offsets["SCHEMA"])

    # NULL group baselines
    null_up_cross = []
    null_down_cross = []
    null_schema_cross = []
    null_common_cross = []
    null_items = up_offsets["NULL"]
    null_a = up_offsets["NULL"]
    null_b = down_offsets["NULL"]
    null_s = schema_offsets["NULL"]
    null_c = common_offsets["NULL"]
    for i in range(len(null_a)):
        for j in range(i + 1, len(null_a)):
            null_up_cross.append(cos(null_a[i][2], null_a[j][2]))
            null_down_cross.append(cos(null_b[i][2], null_b[j][2]))
            null_schema_cross.append(cos(null_s[i][2], null_s[j][2]))
            null_common_cross.append(cos(null_c[i][2], null_c[j][2]))

    # PCA: top eigenvalue ratio for schema offsets
    def pca_top_var(stack):
        if stack.shape[0] < 2:
            return 0.0
        centered = stack - stack.mean(0, keepdims=True)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        return float((S[0] ** 2) / (S ** 2).sum())

    up_stack = np.stack([v for _, _, v in up_offsets["SCHEMA"]])
    down_stack = np.stack([v for _, _, v in down_offsets["SCHEMA"]])
    schema_stack = np.stack([v for _, _, v in schema_offsets["SCHEMA"]])
    common_stack = np.stack([v for _, _, v in common_offsets["SCHEMA"]])
    null_schema_stack = np.stack([v for _, _, v in schema_offsets["NULL"]])

    up_pc1 = pca_top_var(up_stack)
    down_pc1 = pca_top_var(down_stack)
    schema_pc1 = pca_top_var(schema_stack)
    common_pc1 = pca_top_var(common_stack)
    null_schema_pc1 = pca_top_var(null_schema_stack)

    results[layer] = {
        "up_cross_mean": float(np.mean(up_cross)),
        "down_cross_mean": float(np.mean(down_cross)),
        "schema_cross_mean": float(np.mean(schema_cross)),
        "common_cross_mean": float(np.mean(common_cross)),
        "schema_same_domain_mean": float(np.mean(schema_same_domain)),
        "null_up_cross_mean": float(np.mean(null_up_cross)),
        "null_schema_cross_mean": float(np.mean(null_schema_cross)),
        "null_common_cross_mean": float(np.mean(null_common_cross)),
        "up_pc1": up_pc1,
        "down_pc1": down_pc1,
        "schema_pc1": schema_pc1,
        "common_pc1": common_pc1,
        "null_schema_pc1": null_schema_pc1,
    }

    print(f"  Cross-domain alignment of offsets (mean cosine):")
    print(f"    UP only:         {results[layer]['up_cross_mean']:+.4f}  (exp14 found ~0.04)")
    print(f"    DOWN only:       {results[layer]['down_cross_mean']:+.4f}  (exp14 found ~0.07)")
    print(f"    UP − DOWN:       {results[layer]['schema_cross_mean']:+.4f}  ← isolates schema axis")
    print(f"    UP + DOWN:       {results[layer]['common_cross_mean']:+.4f}  ← isolates common confound")
    print(f"    NULL (UP − B):   {results[layer]['null_schema_cross_mean']:+.4f}  ← noise floor for antisym")
    print(f"  PC1 variance ratio:")
    print(f"    UP only: {up_pc1:.3f}, DOWN only: {down_pc1:.3f}, SCHEMA: {schema_pc1:.3f}, COMMON: {common_pc1:.3f}")

    del sae

del model

# ---- Summary report ----
report_path = "/Users/macn/Documents/embeddingexp/results_exp15_antisymmetric_schema.md"
with open(report_path, "w") as f:
    def out(s=""):
        print(s)
        f.write(s + "\n")

    out("# exp15 — Antisymmetric decomposition: isolating the schema axis")
    out()
    out("**Niamh's insight:** exp14 found UP and DOWN positively correlated (+0.06).")
    out("If `UP = A·schema_axis + B·common_axis` and `DOWN = −A·schema_axis + B·common_axis`,")
    out("then `UP − DOWN = 2A·schema_axis` (the antisymmetric part isolates the schema)")
    out("and `UP + DOWN = 2B·common_axis` (the symmetric part isolates the confound).")
    out()
    out("Same matched-pair design as exp14 but with shared baseline per (domain, pair_idx).")
    out()
    out("## Summary across layers")
    out()
    out(f"  {'layer':>5} {'UP_xdomain':>11} {'DOWN_xdomain':>13} {'SCHEMA_xdomain':>15} {'COMMON_xdomain':>15} {'NULL_SCHEMA_x':>14}")
    for layer in range(n_layers):
        r = results[layer]
        out(f"  {layer:>5d} {r['up_cross_mean']:>+11.4f} {r['down_cross_mean']:>+13.4f} "
            f"{r['schema_cross_mean']:>+15.4f} {r['common_cross_mean']:>+15.4f} {r['null_schema_cross_mean']:>+14.4f}")
    out()
    out("PC1 variance ratio (a single dominant direction would be ~0.5+):")
    out(f"  {'layer':>5} {'UP_PC1':>8} {'DOWN_PC1':>10} {'SCHEMA_PC1':>11} {'COMMON_PC1':>11} {'NULL_SCHEMA_PC1':>16}")
    for layer in range(n_layers):
        r = results[layer]
        out(f"  {layer:>5d} {r['up_pc1']:>8.3f} {r['down_pc1']:>10.3f} {r['schema_pc1']:>11.3f} "
            f"{r['common_pc1']:>11.3f} {r['null_schema_pc1']:>16.3f}")
    out()
    out("## Verdicts")
    out()
    out("- **SCHEMA_xdomain >> UP_xdomain and >> DOWN_xdomain**: the schema axis exists, was just hidden by common-axis confound. THE LOAD-BEARING TEST.")
    out("- **SCHEMA_xdomain ≈ UP_xdomain ≈ DOWN_xdomain**: no improvement from decomposition; schemas are weak everywhere.")
    out("- **COMMON_xdomain > SCHEMA_xdomain**: the dominant shared structure across UP/DOWN pairs is the common axis (valence/change), not the schema.")
    out("- **SCHEMA_PC1 >> NULL_SCHEMA_PC1**: a single dominant schema direction exists.")
    out("- **NULL_SCHEMA_xdomain ≈ 0**: antisymmetrization of arbitrary triples gives noise (sanity check).")

print(f"\nReport: {report_path}")

torch.save({
    "results": results,
    "triples": all_triples,
}, "/Users/macn/Documents/embeddingexp/exp15_results.pt")
