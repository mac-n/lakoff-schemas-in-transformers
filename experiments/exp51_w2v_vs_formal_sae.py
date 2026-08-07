"""
exp51: Compare word-pair w2v PCs against exp43 (formal-sentence) SAE PCs.

The w2v PCs come from word-pair offsets (the canonical w2v methodology from
exp40/exp48). The SAE PCs come from exp43 (formal sentence triples at L3).

Methodologically asymmetric (word-pairs vs sentences) but uses what we already
have. Comparison via PC-loading cosines on the shared axis set.
"""
import numpy as np
import gensim.downloader as api
import torch
from sklearn.decomposition import PCA


# ===== Load SAE PCA from exp43 (FORMAL sentence triples, L3) =====
sae_data = torch.load("/Users/macn/Documents/embeddingexp/exp43_results.pt", weights_only=False)
sae_l3 = sae_data[3]
sae_components = sae_l3["pca_components"]
sae_axis_names = sae_l3["axis_names"]
sae_axes = sae_l3["axes"]
sae_var = sae_l3["explained_variance_ratio"]

print(f"SAE (exp43, formal sentences, L3):")
print(f"  axes: {sae_axis_names}")
print(f"  variance: " + ", ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(sae_var)))


# ===== Rebuild w2v PCs from word pairs (matches exp48 methodology) =====
print("\nLoading GloVe-300...")
wv = api.load("glove-wiki-gigaword-300")

VALENCE_PAIRS = [("pleasant", "unpleasant"), ("desirable", "undesirable"), ("agreeable", "disagreeable"),
                 ("enjoyable", "distasteful"), ("delightful", "awful"), ("beneficial", "harmful"),
                 ("wonderful", "terrible"), ("excellent", "dreadful"), ("favorable", "unfavorable"),
                 ("satisfying", "frustrating"), ("nice", "nasty"), ("kind", "cruel")]
AROUSAL_PAIRS = [("intense", "mild"), ("intense", "gentle"), ("alert", "drowsy"), ("urgent", "leisurely"),
                 ("frantic", "tranquil"), ("energetic", "lethargic"), ("aroused", "relaxed"),
                 ("sharp", "dull"), ("acute", "subtle"), ("vivid", "faint"),
                 ("electric", "placid"), ("turbulent", "still")]
UD_PAIRS = [("rose", "fell"), ("rising", "falling"), ("climbing", "descending"), ("ascended", "descended"),
            ("soaring", "plunging"), ("lifted", "sunk"), ("higher", "lower"), ("upward", "downward"),
            ("up", "down"), ("happy", "sad"), ("cheerful", "gloomy"), ("elated", "dejected"),
            ("more", "less"), ("increase", "decrease"), ("grew", "shrank"),
            ("promoted", "demoted"), ("healthy", "sick"), ("strong", "weak")]
IO_PAIRS = [("inside", "outside"), ("contained", "released"), ("enclosed", "freed"), ("in", "out"),
            ("entered", "exited"), ("internal", "external"), ("remembered", "forgotten"),
            ("married", "divorced"), ("included", "excluded"), ("trapped", "escaped")]
FB_PAIRS = [("forward", "backward"), ("advance", "retreat"), ("ahead", "behind"), ("progress", "regress"),
            ("future", "past"), ("next", "previous"), ("evolved", "devolved"), ("onward", "return")]
LD_PAIRS = [("bright", "dark"), ("light", "dark"), ("illuminated", "shadowed"), ("radiant", "dim"),
            ("sunny", "gloomy"), ("clear", "obscure"), ("hopeful", "hopeless"), ("good", "evil"),
            ("enlightened", "ignorant"), ("informed", "uninformed")]
LR_PAIRS = [("left", "right"), ("leftward", "rightward"), ("port", "starboard"), ("liberal", "conservative")]
EXISTENCE_PAIRS = [("born", "died"), ("birth", "death"), ("creating", "destroying"),
                   ("creation", "destruction"), ("created", "destroyed"), ("built", "demolished"),
                   ("emerging", "vanishing"), ("emerged", "vanished"), ("appearing", "disappearing"),
                   ("appeared", "vanished"), ("founded", "abandoned"), ("established", "abandoned"),
                   ("began", "ended"), ("started", "ended")]
COHERENCE_PAIRS = [("coherent", "incoherent"), ("consistent", "inconsistent"), ("aligned", "misaligned"),
                   ("ordered", "disordered"), ("organized", "disorganized"), ("harmonious", "discordant"),
                   ("predictable", "surprising"), ("expected", "unexpected"), ("ordinary", "anomalous"),
                   ("regular", "irregular"), ("normal", "aberrant"), ("orderly", "chaotic")]
SUCCESS_PAIRS = [("win", "lose"), ("won", "lost"), ("succeed", "fail"), ("success", "failure"),
                 ("successful", "unsuccessful"), ("score", "miss"), ("correct", "incorrect"),
                 ("triumph", "defeat"), ("victory", "defeat"), ("pass", "fail"), ("hit", "miss")]
LOSS_PAIRS = [("gain", "loss"), ("profit", "loss"), ("abundance", "scarcity"), ("fortune", "misfortune"),
              ("security", "threat"), ("safety", "danger"), ("wealth", "poverty"), ("plenty", "lack"),
              ("rich", "poor"), ("secure", "vulnerable"), ("safe", "endangered")]

W2V_AXIS_PAIRS = {
    "UP-DOWN": UD_PAIRS, "IN-OUT": IO_PAIRS, "FORWARD-BACK": FB_PAIRS, "LIGHT-DARK": LD_PAIRS,
    "LEFT-RIGHT": LR_PAIRS, "VALENCE": VALENCE_PAIRS, "AROUSAL": AROUSAL_PAIRS,
    "EXISTENCE": EXISTENCE_PAIRS, "COHERENCE": COHERENCE_PAIRS, "SUCCESS": SUCCESS_PAIRS,
    "LOSS": LOSS_PAIRS,
}


def build_w2v_axis(pairs):
    offs = [wv[a] - wv[c] for a, c in pairs if a in wv.key_to_index and c in wv.key_to_index]
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw)


w2v_axes = {n: build_w2v_axis(p) for n, p in W2V_AXIS_PAIRS.items()}
# Use the same axis ordering as SAE
w2v_axis_names = sae_axis_names  # match orientation
M_w2v = np.stack([w2v_axes[n] for n in w2v_axis_names])
pca_w2v = PCA(n_components=min(len(w2v_axis_names), 10))
pca_w2v.fit(M_w2v)

print(f"\nw2v (word pairs, matched axis order):")
print(f"  variance: " + ", ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(pca_w2v.explained_variance_ratio_)))


def pc_loadings(component_vec, axes_dict, names):
    pc_unit = component_vec / np.linalg.norm(component_vec)
    return np.array([float(axes_dict[n] @ pc_unit) for n in names])


w2v_pc_loadings = np.array([pc_loadings(pca_w2v.components_[i], w2v_axes, w2v_axis_names) for i in range(min(8, len(pca_w2v.components_)))])
sae_pc_loadings = np.array([pc_loadings(sae_components[i], sae_axes, sae_axis_names) for i in range(min(8, len(sae_components)))])


print(f"\n=== w2v PC loadings (word pairs) ===")
print(f"  PC  " + "  ".join(f"{n[:9]:>9}" for n in w2v_axis_names))
for i in range(min(6, len(w2v_pc_loadings))):
    print(f"  PC{i+1}  " + "  ".join(f"{l:>+9.3f}" for l in w2v_pc_loadings[i]))

print(f"\n=== SAE PC loadings (FORMAL sentences, exp43 L3) ===")
print(f"  PC  " + "  ".join(f"{n[:9]:>9}" for n in sae_axis_names))
for i in range(min(6, len(sae_pc_loadings))):
    print(f"  PC{i+1}  " + "  ".join(f"{l:>+9.3f}" for l in sae_pc_loadings[i]))


print(f"\n=== Cross-substrate cosine matrix ===")
print(f"  (w2v word-pair PCs vs SAE FORMAL-sentence PCs)")
print(f"  {'':>10}  " + "  ".join(f"SAE-PC{j+1:>2}" for j in range(min(6, len(sae_pc_loadings)))))
for i in range(min(6, len(w2v_pc_loadings))):
    w_vec = w2v_pc_loadings[i] / np.linalg.norm(w2v_pc_loadings[i])
    row = []
    for j in range(min(6, len(sae_pc_loadings))):
        s_vec = sae_pc_loadings[j] / np.linalg.norm(sae_pc_loadings[j])
        row.append(f"{float(w_vec @ s_vec):>+8.3f}")
    var_pct = pca_w2v.explained_variance_ratio_[i] * 100
    print(f"  w2v-PC{i+1} ({var_pct:>4.1f}%)  " + "  ".join(row))

print(f"\nBest match per w2v PC:")
for i in range(min(6, len(w2v_pc_loadings))):
    w_vec = w2v_pc_loadings[i] / np.linalg.norm(w2v_pc_loadings[i])
    matches = []
    for j in range(min(8, len(sae_pc_loadings))):
        s_vec = sae_pc_loadings[j] / np.linalg.norm(sae_pc_loadings[j])
        matches.append((j+1, float(w_vec @ s_vec), sae_var[j]*100))
    matches.sort(key=lambda x: abs(x[1]), reverse=True)
    best_j, best_cos, best_var = matches[0]
    print(f"  w2v-PC{i+1} → SAE-PC{best_j} (cos={best_cos:+.3f}, SAE-PC{best_j} captures {best_var:.1f}%)")
