"""
exp16_antisymmetric_410m.py - same test as exp15, on Pythia 410m.

If schemas-as-polar-opposites EMERGE with model scale, this should show:
  - SCHEMA_xdomain (UP-DOWN) significantly larger than UP or DOWN alone
  - COMMON axis still present but no longer dominant
  - Indicating the model has "figured out" that UP and DOWN are opposite
    poles of a shared vertical axis

If 410m looks identical to 70m, the lack-of-polar-structure persists at scale.

Substrate differences from exp15:
  - Pythia 70m → Pythia 410m (d_model 512 → 1024, n_layers 6 → 24)
  - SAELens residual-stream SAE → eai-sparsify MLP-output SAE
  - 32k features → 65k features
  - These are real substrate changes; for a true apples-to-apples
    comparison we'd want residual-stream SAEs at 410m too, but those only
    cover 4 layers. We'll use the full-layer-coverage MLP version first.
"""

import gc
from collections import defaultdict

import numpy as np
import torch
from sparsify import Sae
from transformer_lens import HookedTransformer

# Same triples as exp15
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
all_triples = []
for domain, triples in DOMAINS.items():
    for i, (b, u, d) in enumerate(triples):
        all_triples.append(("SCHEMA", domain, i, b, u, d))
for i, (b, x, y) in enumerate(NULL_TRIPLES):
    all_triples.append(("NULL", "null", i, b, x, y))

print(f"Triples: {len(all_triples)} (SCHEMA={sum(1 for t in all_triples if t[0]=='SCHEMA')}, NULL={sum(1 for t in all_triples if t[0]=='NULL')})")

# ---- Device ----
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Load Pythia 410m via TransformerLens ----
print("\nLoading Pythia 410m via TransformerLens...")
model = HookedTransformer.from_pretrained("EleutherAI/pythia-410m", device=device)
model.eval()
n_layers = model.cfg.n_layers
print(f"  {n_layers} layers, d_model={model.cfg.d_model}")

# ---- Load all SAEs ----
print("\nLoading EleutherAI/sae-pythia-410m-65k (24 SAEs)...")
saes = Sae.load_many("EleutherAI/sae-pythia-410m-65k")
print(f"  {len(saes)} SAEs loaded. Keys: {sorted(saes.keys())[:3]} ... {sorted(saes.keys())[-3:]}")


def collect_mlp_out(text, layer):
    """Get MLP output at given layer, MAX across token positions, SAE-encoded."""
    hook = f"blocks.{layer}.hook_mlp_out"
    tokens = model.to_tokens(text)
    if tokens.shape[1] > 64:
        tokens = tokens[:, :64]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook)
    mlp_acts = cache[hook][0]  # (seq_len, d_model)
    # Encode through SAE
    sae_key = f"layers.{layer}.mlp"
    sae = saes[sae_key]
    with torch.no_grad():
        # sparsify's Sae.encode returns latents
        latents = sae.encode(mlp_acts.cpu().float())
        # Handle case where encode returns a namedtuple/object with .top_indices and .top_acts
        if hasattr(latents, 'top_indices') and hasattr(latents, 'top_acts'):
            # TopK SAE: reconstruct sparse vector
            n_features = sae.num_latents
            seq_len = mlp_acts.shape[0]
            full = torch.zeros(seq_len, n_features)
            for s in range(seq_len):
                full[s, latents.top_indices[s]] = latents.top_acts[s].float()
            return full.max(0).values.numpy().astype(np.float64)
        else:
            # Dense activations
            return latents.max(0).values.cpu().numpy().astype(np.float64)


def cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---- Run analysis at each layer ----
results = {}
for layer in range(n_layers):
    print(f"\n--- Layer {layer} ---")
    sae_key = f"layers.{layer}.mlp"
    if sae_key not in saes:
        print(f"  No SAE for {sae_key}, skipping")
        continue

    up_offsets = defaultdict(list)
    down_offsets = defaultdict(list)
    schema_offsets = defaultdict(list)
    common_offsets = defaultdict(list)

    for group, domain, idx, baseline, a, b in all_triples:
        b_vec = collect_mlp_out(baseline, layer)
        a_vec = collect_mlp_out(a, layer)
        c_vec = collect_mlp_out(b, layer)
        up_offsets[group].append((domain, idx, a_vec - b_vec))
        down_offsets[group].append((domain, idx, c_vec - b_vec))
        schema_offsets[group].append((domain, idx, (a_vec - b_vec) - (c_vec - b_vec)))
        common_offsets[group].append((domain, idx, (a_vec - b_vec) + (c_vec - b_vec)))

    def pairwise_cross_domain(items):
        sims = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][0] != items[j][0]:
                    sims.append(cos(items[i][2], items[j][2]))
        return sims

    up_cross = pairwise_cross_domain(up_offsets["SCHEMA"])
    down_cross = pairwise_cross_domain(down_offsets["SCHEMA"])
    schema_cross = pairwise_cross_domain(schema_offsets["SCHEMA"])
    common_cross = pairwise_cross_domain(common_offsets["SCHEMA"])

    null_a = up_offsets["NULL"]
    null_b = down_offsets["NULL"]
    null_s = schema_offsets["NULL"]
    null_c = common_offsets["NULL"]
    null_up_cross = []
    null_schema_cross = []
    null_common_cross = []
    for i in range(len(null_a)):
        for j in range(i + 1, len(null_a)):
            null_up_cross.append(cos(null_a[i][2], null_a[j][2]))
            null_schema_cross.append(cos(null_s[i][2], null_s[j][2]))
            null_common_cross.append(cos(null_c[i][2], null_c[j][2]))

    def pca_top_var(stack):
        if stack.shape[0] < 2:
            return 0.0
        centered = stack - stack.mean(0, keepdims=True)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        return float((S[0] ** 2) / (S ** 2).sum())

    up_stack = np.stack([v for _, _, v in up_offsets["SCHEMA"]])
    schema_stack = np.stack([v for _, _, v in schema_offsets["SCHEMA"]])
    common_stack = np.stack([v for _, _, v in common_offsets["SCHEMA"]])

    results[layer] = {
        "up_cross_mean": float(np.mean(up_cross)),
        "down_cross_mean": float(np.mean(down_cross)),
        "schema_cross_mean": float(np.mean(schema_cross)),
        "common_cross_mean": float(np.mean(common_cross)),
        "null_up_cross_mean": float(np.mean(null_up_cross)),
        "null_schema_cross_mean": float(np.mean(null_schema_cross)),
        "null_common_cross_mean": float(np.mean(null_common_cross)),
        "up_pc1": pca_top_var(up_stack),
        "schema_pc1": pca_top_var(schema_stack),
        "common_pc1": pca_top_var(common_stack),
    }

    r = results[layer]
    print(f"  UP_xdomain={r['up_cross_mean']:+.4f}  DOWN_xdomain={r['down_cross_mean']:+.4f}  "
          f"SCHEMA_xdomain={r['schema_cross_mean']:+.4f}  COMMON_xdomain={r['common_cross_mean']:+.4f}")
    print(f"  PC1: UP={r['up_pc1']:.3f}  SCHEMA={r['schema_pc1']:.3f}  COMMON={r['common_pc1']:.3f}")

# ---- Report ----
report_path = "/Users/macn/Documents/embeddingexp/results_exp16_antisymmetric_410m.md"
with open(report_path, "w") as f:
    def out(s=""):
        print(s)
        f.write(s + "\n")

    out("# exp16 — Antisymmetric decomposition on Pythia 410m")
    out()
    out("Same test as exp15 (which ran on 70m) on Pythia 410m's MLP-output SAEs.")
    out("If schemas-as-polar-opposites emerge with scale, SCHEMA_xdomain should be larger")
    out("relative to UP/DOWN/COMMON than at 70m.")
    out()
    out("**For comparison, exp15 at Pythia 70m (residual-stream SAE):**")
    out("  UP_xdomain ≈ 0.04, DOWN_xdomain ≈ 0.06, SCHEMA_xdomain ≈ 0.02 (SMALLER), COMMON_xdomain ≈ 0.06 (LARGEST)")
    out("  i.e. antisymmetric decomposition made things WORSE at 70m. UP and DOWN treated as separate change-dirs.")
    out()
    out("## Summary across layers (Pythia 410m)")
    out()
    out(f"  {'layer':>5} {'UP_xdom':>9} {'DOWN_xdom':>10} {'SCHEMA_xdom':>12} {'COMMON_xdom':>12} {'SCHEMA_PC1':>11} {'COMMON_PC1':>11}")
    for layer in sorted(results.keys()):
        r = results[layer]
        out(f"  {layer:>5d} {r['up_cross_mean']:>+9.4f} {r['down_cross_mean']:>+10.4f} "
            f"{r['schema_cross_mean']:>+12.4f} {r['common_cross_mean']:>+12.4f} "
            f"{r['schema_pc1']:>11.3f} {r['common_pc1']:>11.3f}")
    out()
    out("## Verdicts")
    out()
    out("- If at any layer **SCHEMA_xdomain > UP_xdomain and > DOWN_xdomain**: schemas-as-polar-opposites EMERGED with scale. This is a real positive result for the scale hypothesis.")
    out("- If SCHEMA_xdomain still < UP/DOWN/COMMON at every layer: 410m doesn't have the polar structure either. Pattern persists.")
    out("- Watch the per-layer pattern: maybe schemas emerge at specific depths.")

print(f"\nReport: {report_path}")

torch.save({
    "results": results,
    "triples": all_triples,
}, "/Users/macn/Documents/embeddingexp/exp16_results.pt")
