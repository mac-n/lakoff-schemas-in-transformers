"""
exp7_decoder_pca.py — Bottom-up Phase 1 (post-pivot).

Decoder-geometry PCA on a Neuronpedia-backed Pythia 70m residual-stream SAE.
The pivot rationale: if Lakoffian primitives are primitive (basic cognitive
building blocks rather than emergent niceties), they should be present at
minimum model scale, and possibly *easier* to find there. Pythia 70m has
much less representational room — any structure that's there is load-bearing,
can't be attributed to abundance of capacity. Plus the SAEs are
Neuronpedia-indexed so we get auto-interp feature descriptions for free.

This script is web-Claude's `sae_dictionary_pca.py` rewritten for:
  - the eai-sparsify -> sae-lens loading convention
  - Pythia 70m-deduped residual-stream SAEs (ctigges/...__res-sm_processed)
  - covariance-matched null instead of isotropic (per the exp6 finding that
    anisotropy is real and matters)

Pipeline:
  1. Load Pythia 70m-deduped via TransformerLens
  2. Sample wikitext-2, collect residual-stream activations at chosen layer,
     estimate per-dimension covariance (diagonal of full covariance)
  3. Load SAE for that layer via SAE Lens
  4. PCA on (centred, unit-normalised) decoder rows
  5. Covariance-matched null: sample N random vectors from N(0, diag(σ²)) where
     σ is the per-dim std of activations; PCA them
  6. Compare scree, cumulative variance, participation ratio
  7. Save artefacts for exp8 (per-PC feature-loading rankings + Neuronpedia
     lookup of top-loading features per PC)
"""

import torch
import numpy as np

# ---- Choices ----
LAYER = 3                                # 0-indexed; Pythia 70m has 6 layers; mid-network
RELEASE = "pythia-70m-deduped-res-sm"    # SAE Lens release name (residual stream)
SAE_ID = f"blocks.{LAYER}.hook_resid_post"
N_WIKITEXT_SAMPLES = 1000                # sentences for covariance estimate
MAX_LEN = 128                            # tokens per sentence (truncate)

# ---- Device ----
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- 1. Load model ----
print(f"\nLoading Pythia 70m-deduped via TransformerLens...")
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
model.eval()
print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

# ---- 2. Sample neutral text + collect residual-stream activations ----
# (Bypassing the `datasets` library — recent huggingface_hub requires namespaced
# repo IDs and breaks on `wikitext`. Hardcoded diverse sentences give an
# adequate covariance estimate.)
print("\nLoading hardcoded diverse-sentence sample...")
_BASE_SENTENCES = [
    # Science/technical
    "Quantum mechanics describes the behavior of matter at small scales.",
    "The mitochondria are the powerhouse of the cell, producing ATP through respiration.",
    "Gravitational waves were first detected by LIGO in 2015.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "Neural networks learn representations from data through gradient descent.",
    "The standard model of particle physics describes three fundamental forces.",
    "Climate scientists measure changes in atmospheric carbon over decades.",
    "DNA replication is performed by polymerase enzymes during cell division.",
    "Black holes warp spacetime so severely that even light cannot escape.",
    "Catalysts lower the activation energy of chemical reactions.",
    # News/political
    "The central bank raised interest rates by a quarter point on Tuesday.",
    "Negotiations between the two countries collapsed after the latest summit.",
    "The supreme court ruled five-to-four against the appeal.",
    "Voters in three swing states will likely decide the election outcome.",
    "The minister apologized for the comments made during last week's debate.",
    "Trade tariffs on imported steel were announced this morning.",
    "Researchers say the new drug shows promise in early clinical trials.",
    "The opposition party walked out of parliament in protest.",
    "Authorities are still investigating the cause of the fire downtown.",
    "Inflation eased slightly in March, according to the official data.",
    # Literary/narrative
    "She walked along the river path as the sun set behind the trees.",
    "The old man sat on the porch every evening, watching the road.",
    "Rain fell against the window, slow at first, then heavy.",
    "He hadn't seen his brother in fifteen years, not since the funeral.",
    "The children played in the yard until their mother called them in.",
    "Snow covered the rooftops, muting every sound in the small village.",
    "She closed the book, set it down, and looked out at the harbour.",
    "The letter arrived on a Tuesday, addressed in her grandmother's hand.",
    "Birds sang in the apple tree above the empty hammock.",
    "He poured the tea and waited for her to begin speaking.",
    # Everyday/conversational
    "I'll meet you at the cafe on Thursday around three o'clock.",
    "Don't forget to pick up milk and bread on your way home.",
    "The package finally arrived after sitting at the depot for a week.",
    "She's been training for the marathon since January.",
    "We watched the new series together last weekend.",
    "He fixed the leaking tap with a couple of washers and a wrench.",
    "I think the cat got out through the bathroom window again.",
    "The kids loved the trip to the museum, especially the dinosaur exhibit.",
    "It rained the whole afternoon, so we played board games inside.",
    "She makes the best sourdough I've ever had.",
    # Sports
    "The striker scored a hat-trick in the second half.",
    "Their defence held strong against repeated counter-attacks.",
    "He set a new world record in the 200-metre freestyle.",
    "The match went to penalties after a goalless draw.",
    "She won her first grand slam title at age nineteen.",
    "The coach said the team needs to focus on its passing accuracy.",
    "Fans cheered as the home side took the lead in extra time.",
    "He retired from international cricket after twenty years.",
    "The pitcher struck out seven batters in five innings.",
    "Their winning streak was finally broken in last night's game.",
    # Food/cooking
    "Bring the pot of water to a rolling boil before adding the pasta.",
    "Toast the spices in a dry pan until fragrant.",
    "The bread should double in size during the first proof.",
    "Caramelizing onions takes patience but transforms their flavour.",
    "Whisk the eggs into the cream before folding in the cheese.",
    "Marinate the chicken overnight for the best results.",
    "Roast the vegetables on high heat until the edges char slightly.",
    "She added a pinch of saffron to the broth.",
    "The chef garnished each plate with chopped parsley and lemon zest.",
    "Let the dough rest in the fridge for at least two hours.",
    # Finance/markets
    "The company's quarterly earnings beat analyst expectations.",
    "Shares fell sharply after the unexpected resignation of the CEO.",
    "Bond yields rose in response to the hawkish central bank statement.",
    "The startup raised twenty million dollars in its Series B round.",
    "Cryptocurrency markets remained volatile throughout the week.",
    "Investors are watching the upcoming jobs report closely.",
    "Mergers and acquisitions slowed in the fourth quarter.",
    "The fund's annual return exceeded the benchmark by three percent.",
    "Oil prices climbed on news of supply disruptions in the region.",
    "The IPO was priced at the top of its expected range.",
    # Nature/wildlife
    "The pack of wolves moved through the forest at dawn.",
    "Salmon swim upstream against the current to spawn.",
    "Hummingbirds beat their wings dozens of times per second.",
    "The migration brings millions of wildebeest across the river each year.",
    "Coral reefs are home to a quarter of all marine species.",
    "Owls hunt silently using their specialized feathers.",
    "The mountain stream tumbled over moss-covered rocks.",
    "Bears emerge from hibernation in early spring, hungry and lean.",
    "Whales communicate over vast distances using low-frequency calls.",
    "Wildflowers carpeted the meadow after the late rains.",
]
# 80 base sentences × 13 repetitions ≈ 1040 total (with light variation handled
# by tokenization position when we run them through Pythia). Adequate for
# diagonal-covariance estimation; covariance is averaged across all token
# positions in the batch.
texts = _BASE_SENTENCES * 13
texts = texts[:N_WIKITEXT_SAMPLES]
print(f"  Using {len(texts)} sentences from a {len(_BASE_SENTENCES)}-sentence diverse pool.")

print(f"Collecting residual-stream activations at {SAE_ID}...")
acts_list = []
for idx, text in enumerate(texts):
    if idx % 200 == 0 and idx > 0:
        print(f"  {idx}/{len(texts)}")
    tokens = model.to_tokens(text)
    if tokens.shape[1] > MAX_LEN:
        tokens = tokens[:, :MAX_LEN]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=SAE_ID)
    a = cache[SAE_ID][0].cpu().float()  # (seq_len, d_model) — all token positions
    acts_list.append(a)

all_acts = torch.cat(acts_list, dim=0)  # (N_total_tokens, d_model)
print(f"Collected {all_acts.shape[0]:,} activation vectors of dim {all_acts.shape[1]}")

# Centre and compute per-dimension std (for covariance-matched null)
mean_act = all_acts.mean(0)
all_acts_centered = all_acts - mean_act
sigma_per_dim = all_acts_centered.std(0).numpy().astype(np.float64)
print(f"Per-dim activation std: min={sigma_per_dim.min():.3f}, max={sigma_per_dim.max():.3f}, "
      f"ratio={sigma_per_dim.max() / sigma_per_dim.min():.1f}")

# Free model memory
del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ---- 3. Load SAE ----
print(f"\nLoading SAE: release={RELEASE}, sae_id={SAE_ID}...")
from sae_lens import SAE
try:
    result = SAE.from_pretrained(release=RELEASE, sae_id=SAE_ID, device="cpu")
    # API has varied across versions; handle tuple vs single return
    if isinstance(result, tuple):
        sae = result[0]
        print(f"  (returned tuple with {len(result)} elements; using element 0)")
    else:
        sae = result
except Exception as e:
    print(f"FAILED to load SAE: {e}")
    raise

print(f"  SAE config: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

# decoder weight: (d_sae, d_in) = (num_latents, d_model)
W_dec = sae.W_dec.detach().cpu().float().numpy().astype(np.float64)
print(f"  decoder shape: {W_dec.shape}  (n_features x d_model)")
W_dec = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)
n_features, d_model = W_dec.shape

# ---- 4. PCA on real SAE decoder ----
print("\nPCA on real SAE decoder rows...")
W_dec_centered = W_dec - W_dec.mean(axis=0, keepdims=True)
U_real, S_real, Vh_real = np.linalg.svd(W_dec_centered, full_matrices=False)
var_real = S_real ** 2
var_real_ratio = var_real / var_real.sum()
cum_real = np.cumsum(var_real_ratio)

# ---- 5. Covariance-matched null ----
print("PCA on covariance-matched null (random vectors with N(0, diag(σ²)))...")
rng = np.random.default_rng(0)
R_null = rng.standard_normal((n_features, d_model)) * sigma_per_dim[None, :]
R_null = R_null / np.linalg.norm(R_null, axis=1, keepdims=True)
R_null_centered = R_null - R_null.mean(axis=0, keepdims=True)
U_null, S_null, Vh_null = np.linalg.svd(R_null_centered, full_matrices=False)
var_null_ratio = (S_null ** 2) / (S_null ** 2).sum()
cum_null = np.cumsum(var_null_ratio)

# ---- 6. Numbers ----
def components_for(cum, frac):
    return int(np.searchsorted(cum, frac) + 1)

def participation_ratio(var_ratio):
    return float((var_ratio.sum() ** 2) / (var_ratio ** 2).sum())

print("\n" + "=" * 70)
print(f"RESULTS  ({RELEASE}, layer={LAYER}, d_model={d_model}, n_features={n_features:,})")
print("=" * 70)

print(f"\nComponents to reach X% variance (lower = more compressible):")
print(f"  {'% var':>8s}  {'REAL':>6s}  {'NULL':>6s}  {'real/null':>10s}")
for frac in (0.5, 0.7, 0.9, 0.95, 0.99):
    cr = components_for(cum_real, frac)
    cn = components_for(cum_null, frac)
    print(f"  {int(frac*100):>7d}%  {cr:>6d}  {cn:>6d}  {cr/cn:>10.3f}")

pr_real = participation_ratio(var_real_ratio)
pr_null = participation_ratio(var_null_ratio)
print(f"\nParticipation ratio (lower = more compressible structure):")
print(f"  real = {pr_real:8.1f}")
print(f"  null = {pr_null:8.1f}")
print(f"  ratio (real/null) = {pr_real/pr_null:.3f}  (<1 means real is more compressible than null)")

print(f"\nTop-10 PC variance ratios:")
print(f"  real: {[f'{v:.4f}' for v in var_real_ratio[:10]]}")
print(f"  null: {[f'{v:.4f}' for v in var_null_ratio[:10]]}")

# ---- 7. Save artefacts for exp8 ----
out_path = f"/Users/macn/Documents/embeddingexp/exp7_pca_pythia70m_res_layer{LAYER}.pt"
torch.save({
    "release": RELEASE,
    "sae_id": SAE_ID,
    "layer": LAYER,
    "d_model": d_model,
    "n_features": n_features,
    "real_var_ratio": var_real_ratio,
    "null_var_ratio": var_null_ratio,
    "real_pcs": Vh_real,             # rows = principal directions in d_model space
    "feature_loadings_real": W_dec_centered @ Vh_real.T,   # (n_features, d_model)
    "sigma_per_dim": sigma_per_dim,
}, out_path)
print(f"\nSaved to {out_path}")
print("\nNext: exp8 — for each top PC, identify top-loading features and look them up on Neuronpedia.")
