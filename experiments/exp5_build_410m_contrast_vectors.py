"""
exp5_build_410m_contrast_vectors.py — Phase 1, step 3.

Build literal-pole contrast vectors in Pythia 410m's MLP-output activation
space, at every one of its 24 layers, for each direction we plan to project
onto the SAE decoder.

Why MLP-output space: EleutherAI/sae-pythia-410m-65k was trained on all 24
MLP outputs. The contrast vector must live in the same space as the SAE
decoder rows for the cosine-similarity projection to mean anything.

Directions built:
  - UP, LIGHT, IN, FORWARD  (the Lakoffian image-schema poles; their inverses
                              DOWN/DARK/OUT/BACK are just the negations)
  - YANG                     (Niamh's hypothesis: LIGHT/DARK might map onto
                              yang/yin, in which case the children to test
                              against are Taoist not Lakoffian. Vocabulary
                              decorrelated from LIGHT/DARK by construction)
  - BEVERAGE                 (sham non-Lakoffian schema -- coffee/tea vs
                              alcohol. Top SAE features should cluster in
                              beverage-domain features, NOT span decorrelated
                              children. If sham shows multi-child spread,
                              method is too generous)

Output: /Users/macn/Documents/embeddingexp/contrast_vectors_410m.pt
  - dict {direction: {layer_idx: tensor (d_model,) L2-normalised}}
  - plus metadata

The script also reports cosine similarities between all directions at the
mid-layer as a sanity check. Watch in particular:
  - cos(LIGHT, YANG) -- if positive and noticeably large, supports Niamh's
                        LIGHT-as-yang reading at the representation level
  - cos(UP, YANG)   -- should be lower if YANG is specifically about LIGHT
                        rather than a generic activation/energy axis
  - cos(BEVERAGE, anything) -- should be near 0 if the sham is well-chosen
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools

# ---------- Device ----------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# ---------- Load model ----------
print("\nLoading Pythia 410m...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m").to(device)
model.eval()
n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
print(f"Model loaded. {n_layers} layers, d_model={d_model}")

# ---------- Register MLP-output hooks at every layer ----------
mlp_outputs = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        # GPT-NeoX MLP returns a tensor (not a tuple)
        mlp_outputs[layer_idx] = output.detach()
    return hook

hooks = []
for i in range(n_layers):
    h = model.gpt_neox.layers[i].mlp.register_forward_hook(make_hook(i))
    hooks.append(h)


def get_mlp_outputs_for_word(text):
    """Return dict {layer_idx: last-token MLP output tensor of shape (d_model,)}."""
    tokens = tokenizer(text, return_tensors="pt").to(device)
    mlp_outputs.clear()
    with torch.no_grad():
        model(**tokens)
    return {i: mlp_outputs[i][0, -1, :].clone() for i in range(n_layers)}


def build_contrast_vector(positive_words, negative_words):
    """For each layer, return L2-normalised mean(positive MLP) - mean(negative MLP)."""
    pos_per_layer = [get_mlp_outputs_for_word(w) for w in positive_words]
    neg_per_layer = [get_mlp_outputs_for_word(w) for w in negative_words]
    vectors = {}
    for layer in range(n_layers):
        pos_mean = torch.stack([d[layer] for d in pos_per_layer]).mean(0)
        neg_mean = torch.stack([d[layer] for d in neg_per_layer]).mean(0)
        v = pos_mean - neg_mean
        v = v / v.norm()
        vectors[layer] = v.cpu().float()
    return vectors


# ---------- Vocabulary (literal poles only; no overlap with predicted children) ----------

WORDS = {
    "UP": (
        # literal-spatial UP only -- no quantity, status, mood, valence words
        ["rising", "ascending", "climbing", "soaring", "lifting", "elevated"],
        ["falling", "sinking", "descending", "dropping", "lowering", "plunging"],
    ),
    "LIGHT": (
        # literal illumination only -- no clarity, hope, goodness, knowing words
        ["bright", "illuminated", "shining", "glowing", "luminous", "radiant"],
        ["dark", "dim", "shadowy", "murky", "gloomy", "obscure"],
    ),
    "IN": (
        # literal containment only -- no relationship, category, mind words
        ["inside", "contained", "enclosed", "surrounded", "within", "encased"],
        ["outside", "exposed", "uncovered", "released", "external", "unenclosed"],
    ),
    "FORWARD": (
        # literal motion forward -- no progress, future, advancement metaphor words
        ["forward", "ahead", "advancing", "frontward", "onward", "forth"],
        ["backward", "behind", "retreating", "rearward", "reversing", "backwards"],
    ),
    "YANG": (
        # Taoist yang attributes -- deliberately NO overlap with LIGHT/DARK vocabulary
        # (no bright/shining/luminous on yang side, no dark/dim/shadowy on yin side)
        ["fire", "hot", "dry", "hard", "active", "assertive", "vigorous", "forceful"],
        ["water", "cold", "wet", "soft", "passive", "receptive", "yielding", "quiet"],
    ),
    "BEVERAGE": (
        # sham non-Lakoffian direction. Top features should cluster in
        # beverage-domain features, NOT span multiple decorrelated children.
        ["coffee", "espresso", "latte", "cappuccino", "tea", "matcha"],
        ["beer", "wine", "vodka", "whiskey", "bourbon", "cocktail"],
    ),
}

# ---------- Build all contrast vectors ----------
print(f"\nBuilding contrast vectors at all {n_layers} layers...")
print(f"  Directions: {list(WORDS.keys())}")

contrast_vectors = {}
for direction, (positive, negative) in WORDS.items():
    print(f"  {direction:9s} pos={positive[0]}/{positive[1]}/... neg={negative[0]}/{negative[1]}/...")
    contrast_vectors[direction] = build_contrast_vector(positive, negative)

# Remove hooks
for h in hooks:
    h.remove()

# ---------- Save ----------
out_path = "/Users/macn/Documents/embeddingexp/contrast_vectors_410m.pt"
torch.save({
    "contrast_vectors": contrast_vectors,
    "metadata": {
        "model": "EleutherAI/pythia-410m",
        "n_layers": n_layers,
        "d_model": d_model,
        "device_used": device,
        "directions": list(WORDS.keys()),
        "vocabulary": {k: {"positive": v[0], "negative": v[1]} for k, v in WORDS.items()},
    },
}, out_path)
print(f"\nSaved to {out_path}")

# ---------- Sanity check: cosine similarities between directions ----------
# At several layers, not just mid -- the answer might be layer-dependent
print("\n" + "=" * 80)
print("Cosine similarities between directions at selected layers")
print("(YANG-LIGHT and YANG-DARK are the Niamh hypothesis test;")
print(" BEVERAGE-anything should be near zero if the sham is well-chosen)")
print("=" * 80)

directions = list(contrast_vectors.keys())
sample_layers = [0, 4, 8, 12, 16, 20, 23]

for layer in sample_layers:
    print(f"\nLayer {layer}:")
    for a, b in itertools.combinations(directions, 2):
        va = contrast_vectors[a][layer]
        vb = contrast_vectors[b][layer]
        sim = (va @ vb).item()
        flag = ""
        if {a, b} == {"LIGHT", "YANG"}:
            flag = "   <-- Niamh hypothesis: positive value supports LIGHT-as-yang"
        elif {a, b} == {"UP", "YANG"}:
            flag = "   <-- discriminant: should be lower than LIGHT-YANG if YANG is LIGHT-specific"
        elif "BEVERAGE" in {a, b}:
            flag = "   <-- sham; should be near 0"
        print(f"  cos({a:9s}, {b:9s}) = {sim:+.4f}{flag}")

print("\nDone.")
