"""
exp7b_sweep.py - Sweep all 6 layers x 2 substrates of Pythia 70m-deduped SAEs.

Following Niamh's question whether the exp7 "no low-dim skeleton" finding is
just SAE-design-evenness rather than a real claim. If it is, every (layer,
substrate) combination should show similar participation ratio (~80-85% of
d_model). If some combination deviates, that's where to look for real
concept structure.

Also adds isotropic null as a third reference point alongside the
covariance-matched null, so we can read off how close the real SAE is to
"maximally spread" vs "anisotropy-collapsed".

Pipeline:
  1. Load Pythia 70m once.
  2. Hook all res_post AND mlp_out hookpoints across 6 layers (12 hooks).
  3. Run hardcoded corpus through model in single pass, collecting activations
     for all hooks.
  4. Compute per-hook per-dim std (for covariance-matched null).
  5. Free model.
  6. For each (substrate, layer) in {res-sm, mlp-sm} x {0..5}:
       - Load SAE
       - PCA decoder vs covariance-matched null vs isotropic null
       - Record participation ratio, PCs-to-90%, top-1 PC variance
  7. Print summary table; save full results.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE

# ---- Config ----
LAYERS = list(range(6))
SUBSTRATES = {
    "res-sm": ("pythia-70m-deduped-res-sm", "blocks.{}.hook_resid_post"),
    "mlp-sm": ("pythia-70m-deduped-mlp-sm", "blocks.{}.hook_mlp_out"),
}
N_SAMPLES = 1000
MAX_LEN = 128

# ---- Device ----
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- 1. Load model ----
print("\nLoading Pythia 70m-deduped...")
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
model.eval()
print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

# ---- 2-3. Collect activations across all hookpoints in single pass ----
_BASE_SENTENCES = [
    # Same diverse 80-sentence pool as exp7
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
texts = (_BASE_SENTENCES * 13)[:N_SAMPLES]
print(f"\nUsing {len(texts)} sentences from a {len(_BASE_SENTENCES)}-sentence diverse pool.")

# All hookpoints we need
all_hooks = set()
for substrate, (release, hook_template) in SUBSTRATES.items():
    for layer in LAYERS:
        all_hooks.add(hook_template.format(layer))
all_hooks = sorted(all_hooks)
print(f"Hookpoints to capture ({len(all_hooks)}): {all_hooks[:3]} ... {all_hooks[-3:]}")

# Collect activations
print(f"\nCollecting activations from {len(texts)} sentences...")
acts_by_hook = {h: [] for h in all_hooks}
for idx, text in enumerate(texts):
    if idx % 200 == 0 and idx > 0:
        print(f"  {idx}/{len(texts)}")
    tokens = model.to_tokens(text)
    if tokens.shape[1] > MAX_LEN:
        tokens = tokens[:, :MAX_LEN]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in all_hooks)
    for h in all_hooks:
        if h in cache:
            acts_by_hook[h].append(cache[h][0].cpu().float())

# Compute per-hook per-dim std for the covariance-matched null
print("\nComputing per-hook activation std...")
sigma_by_hook = {}
for h in all_hooks:
    all_acts = torch.cat(acts_by_hook[h], dim=0)
    centered = all_acts - all_acts.mean(0)
    sigma_by_hook[h] = centered.std(0).numpy().astype(np.float64)
    sig = sigma_by_hook[h]
    print(f"  {h:35s}: shape={tuple(all_acts.shape)}  sigma range [{sig.min():.3f}, {sig.max():.3f}]  ratio {sig.max()/sig.min():.1f}")
    # Free memory as we go
    acts_by_hook[h] = None

# Free model
del model
acts_by_hook = None
import gc; gc.collect()


# ---- 6. Sweep (substrate, layer): load SAE -> PCA -> three reference points ----
def participation_ratio(var_ratio):
    return float((var_ratio.sum() ** 2) / (var_ratio ** 2).sum())

def components_for(cum, frac):
    return int(np.searchsorted(cum, frac) + 1)

def pca_summary(W):
    """W shape (n_features, d_model), rows unit-normalized. Return PR, PCs to 90%, top-1 ratio."""
    W_centered = W - W.mean(0, keepdims=True)
    _, S, _ = np.linalg.svd(W_centered, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    cum = np.cumsum(var)
    return participation_ratio(var), components_for(cum, 0.9), var[0]

results = {}
for substrate, (release, hook_template) in SUBSTRATES.items():
    for layer in LAYERS:
        hook = hook_template.format(layer)
        sae_id = hook
        print(f"\n--- {substrate} layer {layer} ({sae_id}) ---")
        try:
            sae_result = SAE.from_pretrained(release=release, sae_id=sae_id, device="cpu")
            sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        W_dec = sae.W_dec.detach().cpu().float().numpy().astype(np.float64)
        W_dec = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)
        n_features, d_model = W_dec.shape
        sigma = sigma_by_hook[hook]

        pr_real, pcs90_real, top1_real = pca_summary(W_dec)

        rng = np.random.default_rng(0)
        R_aniso = rng.standard_normal((n_features, d_model)) * sigma[None, :]
        R_aniso = R_aniso / np.linalg.norm(R_aniso, axis=1, keepdims=True)
        pr_aniso, pcs90_aniso, top1_aniso = pca_summary(R_aniso)

        R_iso = rng.standard_normal((n_features, d_model))
        R_iso = R_iso / np.linalg.norm(R_iso, axis=1, keepdims=True)
        pr_iso, pcs90_iso, top1_iso = pca_summary(R_iso)

        results[(substrate, layer)] = {
            "n_features": n_features,
            "d_model": d_model,
            "pr_real": pr_real,
            "pr_aniso_null": pr_aniso,
            "pr_iso_null": pr_iso,
            "pcs90_real": pcs90_real,
            "pcs90_aniso_null": pcs90_aniso,
            "pcs90_iso_null": pcs90_iso,
            "top1_real": top1_real,
            "top1_aniso_null": top1_aniso,
            "top1_iso_null": top1_iso,
            "real_frac_of_iso": pr_real / pr_iso,  # how close to "maximally spread"
        }
        print(f"  PR: real={pr_real:6.1f}  aniso-null={pr_aniso:6.1f}  iso-null={pr_iso:6.1f}  "
              f"(real is {pr_real/pr_iso*100:5.1f}% of iso)")
        del sae, W_dec, R_aniso, R_iso
        gc.collect()


# ---- 7. Summary table + save ----
print("\n" + "=" * 110)
print("SUMMARY: Pythia 70m-deduped, SAE decoder PCA across layers and substrates")
print("=" * 110)
print(f"  {'substrate':<10s} {'layer':>5s}  {'d_model':>7s}  {'n_feat':>7s}  {'PR_real':>8s}  {'PR_aniso':>9s}  {'PR_iso':>7s}  {'real/iso':>9s}  {'PCs90_real':>11s}  {'top1_real':>10s}")
for (substrate, layer), r in sorted(results.items()):
    print(f"  {substrate:<10s} {layer:>5d}  {r['d_model']:>7d}  {r['n_features']:>7d}  "
          f"{r['pr_real']:>8.1f}  {r['pr_aniso_null']:>9.1f}  {r['pr_iso_null']:>7.1f}  "
          f"{r['real_frac_of_iso']*100:>8.1f}%  {r['pcs90_real']:>11d}  {r['top1_real']:>10.4f}")

print("\nInterpretation guide:")
print("  - real/iso ≈ 100%: SAE perfectly spread (no structure beyond evenness)")
print("  - real/iso << 100%: SAE concentrates more than isotropic random (some structure)")
print("  - real ≈ aniso-null: SAE collapsed onto anisotropic axes (degenerate)")
print("  - top1_real should be ~0.002 (=1/d_model) if perfectly even")

out_path = "/Users/macn/Documents/embeddingexp/exp7b_sweep_results.pt"
torch.save(results, out_path)
print(f"\nSaved to {out_path}")
