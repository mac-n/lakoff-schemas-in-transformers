"""
exp18_antisymmetric_2_8B.py - same antisymmetric test on Pythia 2.8B.

Third scale point. If the SCHEMA < UP/DOWN/COMMON pattern holds at 2.8B too,
"Pythia models at any scale don't represent UP/DOWN as polar opposites" is a
robust three-point cross-scale finding.

If 2.8B suddenly shows SCHEMA > UP/DOWN at some layer, schemas-as-polar emerge
with scale and we have a clean scale story.

Substrate: jacobdunefsky/pythia-2.8B-saes — 32 residual-stream-PRE SAEs
(one per layer of Pythia 2.8B), 60k features each, TopK with k=60, stored as
safetensors. Loaded one-at-a-time to fit in 16GB RAM.

Memory strategy:
  1. Load Pythia 2.8B in fp16 (~5.6GB)
  2. One forward pass per sentence; cache all 32 hook_resid_pre activations
  3. Free model? Probably keep it loaded; ~6GB total used so far
  4. For each layer 0-31: download (huggingface_hub caches) + load SAE
     (629MB), encode all sentences, compute stats, FREE SAE
"""

import gc
import json
import os
from collections import defaultdict

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformer_lens import HookedTransformer

# ---- Same triples as exp15/16 ----
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

all_triples = []
for domain, triples in DOMAINS.items():
    for i, (b, u, d) in enumerate(triples):
        all_triples.append(("SCHEMA", domain, i, b, u, d))
for i, (b, x, y) in enumerate(NULL_TRIPLES):
    all_triples.append(("NULL", "null", i, b, x, y))

# All sentences (flat list, for activation collection)
all_sentences = []
for _, _, _, b, u, d in all_triples:
    all_sentences.extend([b, u, d])
print(f"Total sentences to encode: {len(all_sentences)}")

# ---- Device ----
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Load model (fp16 for memory) ----
print("\nLoading Pythia 2.8B in fp16...")
model = HookedTransformer.from_pretrained(
    "EleutherAI/pythia-2.8b",
    device=device,
    dtype=torch.float16,
)
model.eval()
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
print(f"  {n_layers} layers, d_model={d_model}")


# ---- Collect activations at all hook_resid_pre for all sentences ----
print(f"\nCollecting hook_resid_pre activations for {len(all_sentences)} sentences across all {n_layers} layers...")
all_hooks = [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]
# Store as {layer: list of (seq_len, d_model) tensors per sentence}
activations_per_layer = {i: [] for i in range(n_layers)}

for idx, text in enumerate(all_sentences):
    if idx % 20 == 0 and idx > 0:
        print(f"  {idx}/{len(all_sentences)}")
    tokens = model.to_tokens(text)
    if tokens.shape[1] > 64:
        tokens = tokens[:, :64]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in all_hooks)
    for i in range(n_layers):
        # store in fp32 on CPU to save MPS memory
        activations_per_layer[i].append(cache[all_hooks[i]][0].cpu().float())

print(f"\nActivation collection done. Memory in activations: ~{sum(sum(a.numel() * 4 for a in v) for v in activations_per_layer.values()) / 1e9:.2f} GB")

# Free model — we don't need it anymore
del model
gc.collect()
if device == "mps":
    torch.mps.empty_cache()
elif device == "cuda":
    torch.cuda.empty_cache()
print("Model freed.")


def encode_topk_sae(x, w_enc, b_enc, w_dec, b_dec, k):
    """TopK SAE encode. x is (..., d_in); returns sparse (..., n_features)."""
    pre = (x - b_dec) @ w_enc + b_enc  # (..., n_features)
    topk_vals, topk_idx = torch.topk(pre, k, dim=-1)
    sparse = torch.zeros_like(pre)
    sparse.scatter_(-1, topk_idx, topk_vals)
    return sparse


def cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pca_top_var(stack):
    if stack.shape[0] < 2:
        return 0.0
    centered = stack - stack.mean(0, keepdims=True)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    return float((S[0] ** 2) / (S ** 2).sum())


# ---- Per layer: load SAE, encode, analyze, free ----
print("\nProcessing each layer (download + load SAE + encode + analyze + free)...")
results = {}

for layer in range(n_layers):
    print(f"\n--- Layer {layer} ---")

    # Download SAE files (cached on disk)
    print("  Downloading SAE...")
    try:
        sae_path = hf_hub_download(
            repo_id="jacobdunefsky/pythia-2.8B-saes",
            filename=f"pythia-2.8B-dun-resid-sae{layer}/sae.safetensors",
        )
        config_path = hf_hub_download(
            repo_id="jacobdunefsky/pythia-2.8B-saes",
            filename=f"pythia-2.8B-dun-resid-sae{layer}/sae.json",
        )
    except Exception as e:
        print(f"  SAE download FAILED: {e}")
        continue

    with open(config_path) as f:
        cfg = json.load(f)
    k = cfg["top_k"]

    # Load SAE tensors
    sae_w = {}
    with safe_open(sae_path, framework="pt") as f:
        for key in f.keys():
            sae_w[key] = f.get_tensor(key).float()
    print(f"  SAE loaded: {sae_w['W_enc'].shape[0]} → {sae_w['W_enc'].shape[1]} features, k={k}")

    # Encode each sentence's activations through this SAE; take MAX across token positions
    feat_per_sentence = []  # list of (n_features,) feature vectors per sentence
    for sentence_acts in activations_per_layer[layer]:
        # sentence_acts is (seq_len, d_model) on CPU fp32
        with torch.no_grad():
            sparse = encode_topk_sae(sentence_acts, sae_w["W_enc"], sae_w["b_enc"],
                                     sae_w["W_dec"], sae_w["b_dec"], k)
            feat_per_sentence.append(sparse.max(0).values.numpy().astype(np.float64))

    # Index back into triples
    up_offsets = defaultdict(list)
    down_offsets = defaultdict(list)
    schema_offsets = defaultdict(list)
    common_offsets = defaultdict(list)

    sentence_idx = 0
    for group, domain, idx, baseline, a, b in all_triples:
        b_vec = feat_per_sentence[sentence_idx]
        a_vec = feat_per_sentence[sentence_idx + 1]
        c_vec = feat_per_sentence[sentence_idx + 2]
        sentence_idx += 3
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

    schema_stack = np.stack([v for _, _, v in schema_offsets["SCHEMA"]])
    common_stack = np.stack([v for _, _, v in common_offsets["SCHEMA"]])

    results[layer] = {
        "up_cross_mean": float(np.mean(up_cross)),
        "down_cross_mean": float(np.mean(down_cross)),
        "schema_cross_mean": float(np.mean(schema_cross)),
        "common_cross_mean": float(np.mean(common_cross)),
        "schema_pc1": pca_top_var(schema_stack),
        "common_pc1": pca_top_var(common_stack),
    }

    r = results[layer]
    print(f"  UP_xd={r['up_cross_mean']:+.4f}  DOWN_xd={r['down_cross_mean']:+.4f}  "
          f"SCHEMA_xd={r['schema_cross_mean']:+.4f}  COMMON_xd={r['common_cross_mean']:+.4f}")

    # Free SAE memory
    del sae_w, feat_per_sentence
    gc.collect()


# ---- Report ----
report_path = "/Users/macn/Documents/embeddingexp/results_exp18_antisymmetric_2_8B.md"
with open(report_path, "w") as f:
    def out(s=""):
        print(s)
        f.write(s + "\n")

    out("# exp18 — Antisymmetric decomposition on Pythia 2.8B")
    out()
    out("Third scale point in the antisymmetric-decomposition test. Same matched-triple")
    out("design as exp15 (70m) and exp16 (410m). 32-layer residual-stream-pre SAEs from")
    out("jacobdunefsky/pythia-2.8B-saes (60k features, TopK k=60).")
    out()
    out("**Reference patterns from smaller models:**")
    out("- 70m: SCHEMA_xdomain ≈ 0.02, UP/DOWN/COMMON_xdomain ≈ 0.04-0.06")
    out("- 410m: SCHEMA_xdomain ≈ 0.02-0.05, UP/DOWN/COMMON_xdomain ≈ 0.03-0.11")
    out()
    out("Both showed: SCHEMA < UP < DOWN < COMMON. Hypothesis: same at 2.8B.")
    out("Counter-hypothesis: scale changes the picture and SCHEMA emerges above UP/DOWN.")
    out()
    out("## Summary across all 32 layers")
    out()
    out(f"  {'layer':>5} {'UP_xdom':>9} {'DOWN_xdom':>10} {'SCHEMA_xdom':>12} {'COMMON_xdom':>12} {'SCHEMA_PC1':>11}")
    for layer in sorted(results.keys()):
        r = results[layer]
        out(f"  {layer:>5d} {r['up_cross_mean']:>+9.4f} {r['down_cross_mean']:>+10.4f} "
            f"{r['schema_cross_mean']:>+12.4f} {r['common_cross_mean']:>+12.4f} {r['schema_pc1']:>11.3f}")

print(f"\nReport: {report_path}")

torch.save({
    "results": results,
    "triples": all_triples,
}, "/Users/macn/Documents/embeddingexp/exp18_results.pt")
