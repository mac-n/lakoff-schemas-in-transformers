"""
exp6_distractor_projection.py — Phase 1 step 4.

Anisotropy correction for the contrast vectors built in exp5.

Motivation: exp5 found |cos(LIGHT, BEVERAGE)| ≈ 0.2 at layer 12 (and ≈ 0.99 at
layer 23), meaning our contrast vectors at all-but-the-first layer are
substantially driven by anisotropy of the MLP-output activation space rather
than schema semantics. Li et al. (Geometry of Concepts, 2024) show this same
phenomenon for Gemma residual streams and propose projecting out
high-variance "distractor" directions before doing concept geometry. This
script implements the LDA-style distractor projection adapted for our setup.

Pipeline:
  1. Load Pythia 410m + register MLP-output hooks
  2. Sample ~1000 sentences from wikitext (neutral corpus). Collect last-token
     MLP outputs at every layer.
  3. PCA per layer (via SVD). Top-K principal directions per layer = the
     distractor subspace.
  4. Load existing contrast vectors from exp5.
  5. Project each contrast vector onto the orthogonal complement of the
     distractor subspace at each layer. Renormalise.
  6. Re-compute cosine sims. Compare BEFORE vs AFTER at multiple layers.
  7. Save corrected vectors + the distractor-subspace bases (so we can apply
     the same projection to SAE decoder rows later).

We default to K=3 distractors initially; will adjust based on the scree at
each layer.

Output: contrast_vectors_410m_corrected.pt
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools

K_DISTRACTORS = 3  # top-K principal directions to project out per layer
N_CORPUS_SENTENCES = 1000  # how many neutral sentences for covariance estimate

# --- Device ---
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load model + register MLP-output hooks ---
print("\nLoading Pythia 410m...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m").to(device)
model.eval()
n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
print(f"Model: {n_layers} layers, d_model={d_model}")

mlp_outputs = {}
def make_hook(i):
    def hook(module, input, output):
        mlp_outputs[i] = output.detach()
    return hook

hooks = [model.gpt_neox.layers[i].mlp.register_forward_hook(make_hook(i)) for i in range(n_layers)]


# --- Load neutral corpus for covariance estimation ---
print("\nLoading neutral corpus (wikitext-2)...")
try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
    texts = [t.strip() for t in ds["text"] if len(t.strip()) > 50]
    texts = texts[:N_CORPUS_SENTENCES]
    print(f"  Loaded {len(texts)} sentences from wikitext-2.")
except Exception as e:
    print(f"  wikitext load failed: {e}")
    print("  Falling back to a small hardcoded diverse corpus.")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "She bought a basket of apples and walked home along the river.",
        "The economy contracted by three percent in the fourth quarter.",
        "Quantum mechanics describes the behavior of matter at small scales.",
        "He played the violin softly while she read by candlelight.",
        "The mountain rose sharply from the surrounding plains.",
        "After the storm passed, the air was clear and cool.",
        "Trade negotiations between the two countries lasted for weeks.",
        "The novel explores themes of memory, loss, and reconciliation.",
        "Children laughed as the dog chased its tail around the yard.",
    ] * 50  # 500 sentences via repetition — not ideal but a fallback


# --- Collect MLP-output activations ---
print(f"\nCollecting last-token MLP outputs from {len(texts)} sentences...")
sample_activations = {i: [] for i in range(n_layers)}

for idx, text in enumerate(texts):
    if idx % 100 == 0 and idx > 0:
        print(f"  {idx}/{len(texts)}")
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    if tokens.input_ids.shape[1] == 0:
        continue
    mlp_outputs.clear()
    with torch.no_grad():
        model(**tokens)
    for i in range(n_layers):
        sample_activations[i].append(mlp_outputs[i][0, -1, :].cpu().float())

print(f"Collected {len(sample_activations[0])} samples per layer.")

# Remove hooks
for h in hooks:
    h.remove()


# --- PCA per layer; identify distractor subspaces ---
print(f"\nComputing PCA per layer; identifying top {K_DISTRACTORS} distractor directions...")
distractor_subspaces = {}  # layer -> (d_model, K) orthonormal columns = distractors

for layer in range(n_layers):
    X = torch.stack(sample_activations[layer])  # (N, d_model)
    X = X - X.mean(0, keepdim=True)  # centre
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    # Vh rows are principal directions sorted by descending singular value
    top_k = Vh[:K_DISTRACTORS]  # (K, d_model)
    distractor_subspaces[layer] = top_k.T  # (d_model, K)
    if layer in [0, 4, 8, 12, 16, 20, 23]:
        explained_k = (S[:K_DISTRACTORS] ** 2).sum() / (S ** 2).sum()
        explained_1 = (S[0] ** 2) / (S ** 2).sum()
        print(f"  Layer {layer:2d}: top 1 PC = {explained_1*100:5.1f}% var; "
              f"top {K_DISTRACTORS} PCs = {explained_k*100:5.1f}% var")


# --- Load exp5 contrast vectors and apply correction ---
print("\nLoading exp5 contrast vectors...")
data = torch.load("/Users/macn/Documents/embeddingexp/contrast_vectors_410m.pt", weights_only=False)
contrast_vectors = data["contrast_vectors"]
directions = list(contrast_vectors.keys())

print(f"Projecting contrast vectors onto orthogonal complement of distractor subspace...")
contrast_vectors_corrected = {d: {} for d in directions}

for direction in directions:
    for layer in range(n_layers):
        v = contrast_vectors[direction][layer]  # (d_model,)
        D = distractor_subspaces[layer]  # (d_model, K)
        # Orthogonal projection: v_corrected = v - D D^T v
        proj = D @ (D.T @ v)
        v_corrected = v - proj
        v_corrected = v_corrected / v_corrected.norm()
        contrast_vectors_corrected[direction][layer] = v_corrected


# --- Save corrected vectors + distractor subspaces ---
out_path = "/Users/macn/Documents/embeddingexp/contrast_vectors_410m_corrected.pt"
torch.save({
    "contrast_vectors": contrast_vectors_corrected,
    "metadata": {
        **data["metadata"],
        "k_distractors_projected_out": K_DISTRACTORS,
        "covariance_corpus_size": len(sample_activations[0]),
        "covariance_corpus": "wikitext-2-v1 (or hardcoded fallback)",
    },
    "distractor_subspaces": distractor_subspaces,
}, out_path)
print(f"\nSaved to {out_path}")


# --- BEFORE vs AFTER cosine similarity comparison ---
print("\n" + "=" * 80)
print(f"BEFORE vs AFTER distractor projection (top {K_DISTRACTORS} PCs removed)")
print("=" * 80)

for layer in [0, 4, 8, 12, 16, 20, 23]:
    print(f"\nLayer {layer}:")
    print(f"  {'pair':35s} {'BEFORE':>9s} {'AFTER':>9s}  notes")
    for a, b in itertools.combinations(directions, 2):
        before = (contrast_vectors[a][layer] @ contrast_vectors[b][layer]).item()
        after = (contrast_vectors_corrected[a][layer] @ contrast_vectors_corrected[b][layer]).item()
        flag = ""
        if {a, b} == {"LIGHT", "YANG"}:
            flag = "Niamh hypothesis"
        elif "BEVERAGE" in {a, b}:
            flag = "sham (target ≈ 0)"
        print(f"  cos({a:7s}, {b:9s})  {before:+.3f}   {after:+.3f}   {flag}")

# Summary: median |cos| before and after, per layer
print("\n" + "=" * 80)
print("Summary: median |cos| across all 15 direction-pairs, per layer")
print("=" * 80)
print(f"  {'layer':>5s}  {'BEFORE':>8s}  {'AFTER':>8s}")
for layer in range(n_layers):
    before_abs = sorted(abs((contrast_vectors[a][layer] @ contrast_vectors[b][layer]).item())
                         for a, b in itertools.combinations(directions, 2))
    after_abs = sorted(abs((contrast_vectors_corrected[a][layer] @ contrast_vectors_corrected[b][layer]).item())
                        for a, b in itertools.combinations(directions, 2))
    median_before = before_abs[len(before_abs) // 2]
    median_after = after_abs[len(after_abs) // 2]
    print(f"  {layer:5d}  {median_before:8.3f}  {median_after:8.3f}")

print("\nDone.")
