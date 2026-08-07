"""
exp48: Cross-substrate PC comparison.

For each PC in each substrate, build a loading-vector over the SHARED axes
(VALENCE, AROUSAL, UP-DOWN, IN-OUT, FORWARD-BACK, LIGHT-DARK, EXISTENCE,
COHERENCE, SUCCESS, LOSS). Then compute pairwise cosine similarities between
word2vec PCs and SAE PCs in this axis-loading space.

High cosine = "the same kind of structure" — these PCs load on the same axes
in the same directions across substrates, even though they live in different
feature/word spaces.

This gives us a principled answer to: which PCs are substrate-invariant, and
which are substrate-specific?
"""
import numpy as np
import torch


# ===================== LOAD WORD2VEC PCA RESULTS =====================
# exp40 used these axes in this order:
w2v_data = np.load("/Users/macn/Documents/embeddingexp/exp40_results.npz", allow_pickle=True)
# Actually exp40 saved the cluster-only PCA. Let me re-check what's in there.
print("exp40 npz keys:", w2v_data.files)


# Recompute w2v PCA loadings from saved axes
# axes are stored in exp40_results.npz as 'full_axes' dict
w2v_axes_dict = w2v_data["full_axes"].item() if "full_axes" in w2v_data.files else {}
print("w2v axes:", list(w2v_axes_dict.keys())[:5], "...")


# ===================== LOAD SAE PCA RESULTS (exp45) =====================
sae_data = torch.load("/Users/macn/Documents/embeddingexp/exp45_results.pt", weights_only=False)
sae_components = sae_data["pca_components"]
sae_axis_names = sae_data["axis_names"]
sae_axes = sae_data["axes"]
sae_var = sae_data["explained_variance_ratio"]

print(f"\nSAE PCA (exp45, L3):")
print(f"  axis names: {sae_axis_names}")
print(f"  components shape: {sae_components.shape}")


# Compute SAE PC loadings on its input axes
def pc_loadings(component_vec, axes_dict, axis_names):
    """For a given PC vector, compute its cosine with each input axis."""
    pc_unit = component_vec / np.linalg.norm(component_vec)
    return np.array([float(axes_dict[n] @ pc_unit) for n in axis_names])


sae_pc_loadings = []
for pc_idx in range(min(8, len(sae_components))):
    loads = pc_loadings(sae_components[pc_idx], sae_axes, sae_axis_names)
    sae_pc_loadings.append(loads)
sae_pc_loadings = np.array(sae_pc_loadings)


print(f"\nSAE PC loadings on input axes (rows=PCs, cols={sae_axis_names}):")
print(f"  {'PC':>4}  " + "  ".join(f"{n:>10}" for n in sae_axis_names))
for i in range(len(sae_pc_loadings)):
    print(f"  PC{i+1:>2}  " + "  ".join(f"{l:>+10.3f}" for l in sae_pc_loadings[i]))


# ===================== RECOMPUTE WORD2VEC PCA =====================
# We need to rebuild w2v PCA with the same axis ordering as SAE for clean comparison
# exp40 had the full axis set. Let me re-do PCA on just the comparable axes.
import gensim.downloader as api
print("\nLoading GloVe...")
wv = api.load("glove-wiki-gigaword-300")


# Rebuild word-pair-based axes for the 10 shared axes (skip LEFT-RIGHT for now)
# These pair lists match those used in exp40
VALENCE_PAIRS = [
    ("pleasant", "unpleasant"), ("desirable", "undesirable"),
    ("agreeable", "disagreeable"), ("enjoyable", "distasteful"),
    ("delightful", "awful"), ("beneficial", "harmful"),
    ("wonderful", "terrible"), ("excellent", "dreadful"),
    ("favorable", "unfavorable"), ("satisfying", "frustrating"),
    ("nice", "nasty"), ("kind", "cruel"),
]
AROUSAL_PAIRS = [
    ("intense", "mild"), ("intense", "gentle"),
    ("alert", "drowsy"), ("urgent", "leisurely"),
    ("frantic", "tranquil"), ("energetic", "lethargic"),
    ("aroused", "relaxed"), ("sharp", "dull"),
    ("acute", "subtle"), ("vivid", "faint"),
    ("electric", "placid"), ("turbulent", "still"),
]
UD_PAIRS = [
    ("rose", "fell"), ("rising", "falling"), ("climbing", "descending"),
    ("ascended", "descended"), ("soaring", "plunging"), ("lifted", "sunk"),
    ("higher", "lower"), ("upward", "downward"), ("up", "down"),
    ("happy", "sad"), ("cheerful", "gloomy"), ("elated", "dejected"),
    ("more", "less"), ("increase", "decrease"), ("grew", "shrank"),
    ("promoted", "demoted"), ("healthy", "sick"), ("strong", "weak"),
]
IO_PAIRS = [
    ("inside", "outside"), ("contained", "released"), ("enclosed", "freed"),
    ("in", "out"), ("entered", "exited"), ("internal", "external"),
    ("remembered", "forgotten"), ("married", "divorced"),
    ("included", "excluded"), ("trapped", "escaped"),
]
FB_PAIRS = [
    ("forward", "backward"), ("advance", "retreat"), ("ahead", "behind"),
    ("progress", "regress"), ("future", "past"), ("next", "previous"),
    ("evolved", "devolved"), ("onward", "return"),
]
LD_PAIRS = [
    ("bright", "dark"), ("light", "dark"), ("illuminated", "shadowed"),
    ("radiant", "dim"), ("sunny", "gloomy"), ("clear", "obscure"),
    ("hopeful", "hopeless"), ("good", "evil"),
    ("enlightened", "ignorant"), ("informed", "uninformed"),
]
EXISTENCE_PAIRS = [
    ("born", "died"), ("birth", "death"),
    ("creating", "destroying"), ("creation", "destruction"),
    ("created", "destroyed"), ("built", "demolished"),
    ("emerging", "vanishing"), ("emerged", "vanished"),
    ("appearing", "disappearing"), ("appeared", "vanished"),
    ("founded", "abandoned"), ("established", "abandoned"),
    ("began", "ended"), ("started", "ended"),
]
COHERENCE_PAIRS = [
    ("coherent", "incoherent"), ("consistent", "inconsistent"),
    ("aligned", "misaligned"), ("ordered", "disordered"),
    ("organized", "disorganized"), ("harmonious", "discordant"),
    ("predictable", "surprising"), ("expected", "unexpected"),
    ("ordinary", "anomalous"), ("regular", "irregular"),
    ("normal", "aberrant"), ("orderly", "chaotic"),
]
SUCCESS_PAIRS = [
    ("win", "lose"), ("won", "lost"), ("succeed", "fail"),
    ("success", "failure"), ("successful", "unsuccessful"),
    ("score", "miss"), ("correct", "incorrect"),
    ("triumph", "defeat"), ("victory", "defeat"),
    ("pass", "fail"), ("hit", "miss"),
]
LOSS_PAIRS = [
    ("gain", "loss"), ("profit", "loss"),
    ("abundance", "scarcity"), ("fortune", "misfortune"),
    ("security", "threat"), ("safety", "danger"),
    ("wealth", "poverty"), ("plenty", "lack"),
    ("rich", "poor"), ("secure", "vulnerable"), ("safe", "endangered"),
]

W2V_AXIS_PAIRS = {
    "VALENCE": VALENCE_PAIRS, "AROUSAL": AROUSAL_PAIRS,
    "UP-DOWN": UD_PAIRS, "IN-OUT": IO_PAIRS,
    "FORWARD-BACK": FB_PAIRS, "LIGHT-DARK": LD_PAIRS,
    "EXISTENCE": EXISTENCE_PAIRS, "COHERENCE": COHERENCE_PAIRS,
    "SUCCESS": SUCCESS_PAIRS, "LOSS": LOSS_PAIRS,
}


def build_w2v_axis(pairs):
    offs = [wv[a] - wv[c] for a, c in pairs if a in wv.key_to_index and c in wv.key_to_index]
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw)


w2v_axes_dict = {name: build_w2v_axis(pairs) for name, pairs in W2V_AXIS_PAIRS.items()}
w2v_axis_names = list(W2V_AXIS_PAIRS.keys())

# PCA on the 10-axis w2v matrix
from sklearn.decomposition import PCA
M_w2v = np.stack([w2v_axes_dict[n] for n in w2v_axis_names])
pca_w2v = PCA(n_components=min(len(w2v_axis_names), 10))
pca_w2v.fit(M_w2v)

print(f"\nWord2vec PCA (recomputed for 10-axis match):")
print(f"  Variance: " + ", ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(pca_w2v.explained_variance_ratio_)))

w2v_pc_loadings = []
for pc_idx in range(min(8, len(pca_w2v.components_))):
    loads = pc_loadings(pca_w2v.components_[pc_idx], w2v_axes_dict, w2v_axis_names)
    w2v_pc_loadings.append(loads)
w2v_pc_loadings = np.array(w2v_pc_loadings)


print(f"\nWord2vec PC loadings on input axes:")
print(f"  {'PC':>4}  " + "  ".join(f"{n:>10}" for n in w2v_axis_names))
for i in range(len(w2v_pc_loadings)):
    print(f"  PC{i+1:>2}  " + "  ".join(f"{l:>+10.3f}" for l in w2v_pc_loadings[i]))


# ===================== CROSS-SUBSTRATE COMPARISON =====================
# Make sure axes align between substrates (they should)
print(f"\nSAE axis order:  {sae_axis_names}")
print(f"w2v axis order:  {w2v_axis_names}")

# Reorder w2v if needed
sae_to_w2v = {sae_name: w2v_axis_names.index(sae_name) for sae_name in sae_axis_names
              if sae_name in w2v_axis_names}
print(f"\nShared axes: {list(sae_to_w2v.keys())}")

# For SAE, restrict to shared axes
sae_shared_idx = [sae_axis_names.index(n) for n in sae_to_w2v]
w2v_shared_idx = [w2v_axis_names.index(n) for n in sae_to_w2v]

print(f"\n=== Cross-substrate PC similarity matrix (cosines of loading vectors) ===")
print(f"  (rows = w2v PCs, cols = SAE PCs; high |cos| = same structural axis)")
print(f"  {'':>5}  " + "  ".join(f"SAE-PC{j+1:>2}" for j in range(min(6, len(sae_pc_loadings)))))
for i in range(min(6, len(w2v_pc_loadings))):
    w_vec = w2v_pc_loadings[i][w2v_shared_idx]
    w_vec = w_vec / np.linalg.norm(w_vec)
    row = []
    for j in range(min(6, len(sae_pc_loadings))):
        s_vec = sae_pc_loadings[j][sae_shared_idx]
        s_vec = s_vec / np.linalg.norm(s_vec)
        cos = float(w_vec @ s_vec)
        row.append(f"{cos:>+8.3f}")
    var_pct = pca_w2v.explained_variance_ratio_[i] * 100
    print(f"  w2v-PC{i+1} ({var_pct:.1f}%)  " + "  ".join(row))

print(f"\n  Best match per w2v PC (across SAE PCs):")
for i in range(min(6, len(w2v_pc_loadings))):
    w_vec = w2v_pc_loadings[i][w2v_shared_idx]
    w_vec = w_vec / np.linalg.norm(w_vec)
    matches = []
    for j in range(min(8, len(sae_pc_loadings))):
        s_vec = sae_pc_loadings[j][sae_shared_idx]
        s_vec = s_vec / np.linalg.norm(s_vec)
        matches.append((j+1, float(w_vec @ s_vec)))
    matches.sort(key=lambda x: abs(x[1]), reverse=True)
    best = matches[0]
    print(f"  w2v-PC{i+1} → SAE-PC{best[0]} (cos={best[1]:+.3f})")

print(f"\nSaved: nothing yet")
