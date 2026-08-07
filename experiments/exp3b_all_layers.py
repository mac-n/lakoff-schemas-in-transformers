"""
Experiment 3b: ALL-LAYER STEERING + LAYER SWEEP

Two questions:
  1. What happens when you inject UP at EVERY layer? (the "full nervous system" test)
  2. Which individual layers respond most strongly? (where does UP live?)

Uses the same model/direction approach as exp3, but:
  - Part A: Inject at all 24 layers simultaneously (lower per-layer strength since it accumulates)
  - Part B: Sweep individual layers at fixed strength to map where UP has the most effect
"""

import torch
from transformer_lens import HookedTransformer

# ---------- Device setup ----------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# ---------- Load model ----------
print("\nLoading Pythia 1.4B...")
model = HookedTransformer.from_pretrained("pythia-1.4b", device=device)
model.eval()
n_layers = model.cfg.n_layers
print(f"Model loaded. {n_layers} layers, d_model={model.cfg.d_model}")

# ---------- Build UP direction at each layer ----------
up_words = ["up", "rising", "lifting", "ascending", "climbing", "soaring",
            "elevating", "uplifting", "higher", "upward"]
down_words = ["down", "falling", "sinking", "descending", "dropping", "plummeting",
              "lowering", "collapsing", "lower", "downward"]


def get_residual_at_last_token(text, hook_name):
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
    return cache[hook_name][0, -1, :].clone()


# Build UP direction at every layer (we need per-layer directions because
# the residual stream changes character as it flows through the model)
print("\nBuilding UP directions at all layers...")
up_directions = {}
for layer in range(n_layers):
    hook_name = f"blocks.{layer}.hook_resid_post"
    up_acts = torch.stack([get_residual_at_last_token(w, hook_name) for w in up_words])
    down_acts = torch.stack([get_residual_at_last_token(w, hook_name) for w in down_words])
    d = up_acts.mean(dim=0) - down_acts.mean(dim=0)
    up_directions[layer] = d / d.norm()
    print(f"  Layer {layer:2d}: done")

print("All layer directions computed.")


# ---------- Generation helpers ----------

def make_steering_hook(direction, strength):
    def hook_fn(resid, hook):
        return resid + strength * direction
    return hook_fn


def generate_steered(prompt, hooks_list, max_new_tokens=50, temperature=0.8, seed=42):
    torch.manual_seed(seed)
    with model.hooks(fwd_hooks=hooks_list):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            output = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                verbose=False,
            )
    return model.to_string(output[0])


def generate_baseline(prompt, max_new_tokens=50, temperature=0.8, seed=42):
    torch.manual_seed(seed)
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        output = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            verbose=False,
        )
    return model.to_string(output[0])


def show(prompt, text):
    completion = text[len(prompt):] if text.startswith(prompt) else text
    print(f"  {completion.strip()}")


# ---------- Test prompts ----------
prompts = [
    "I want to tell you about my day. It was",
    "The weather today is",
    "When I think about the future, I feel",
]


# ==================== PART A: ALL-LAYER STEERING ====================
print("\n" + "=" * 70)
print("PART A: ALL-LAYER STEERING")
print("Injecting UP at every layer simultaneously.")
print("Using lower per-layer strengths since the effect accumulates (24x).")
print("=" * 70)

# Per-layer strengths to try. Since we're adding at 24 layers,
# effective total = per_layer * 24. So per_layer=1 ~ single-layer strength=24.
per_layer_strengths = [0, 0.5, 1.0, 2.0, 4.0]

for prompt in prompts:
    print(f"\n{'#' * 70}")
    print(f"PROMPT: '{prompt}'")
    print('#' * 70)

    for s in per_layer_strengths:
        if s == 0:
            text = generate_baseline(prompt)
            print(f"\n[baseline (no steering)]")
        else:
            hooks = [
                (f"blocks.{layer}.hook_resid_post", make_steering_hook(up_directions[layer], s))
                for layer in range(n_layers)
            ]
            text = generate_steered(prompt, hooks)
            print(f"\n[all layers, per-layer strength = {s} (total ~ {s * n_layers:.0f})]")
        show(prompt, text)


# ==================== PART B: LAYER SWEEP ====================
print("\n\n" + "=" * 70)
print("PART B: LAYER SWEEP")
print("Injecting UP at one layer at a time, strength=6, to find where UP lives.")
print("=" * 70)

# Use the prompt that responded best in exp3
sweep_prompt = "The weather today is"
sweep_strength = 6

print(f"\nPROMPT: '{sweep_prompt}'")
print(f"Strength: {sweep_strength}")
print()

# Baseline
text = generate_baseline(sweep_prompt)
print(f"[baseline]")
show(sweep_prompt, text)
print()

# Sweep every 3rd layer to keep output manageable, plus first and last
sweep_layers = [0, 3, 6, 9, 12, 15, 18, 21, 23]
for layer in sweep_layers:
    hook_name = f"blocks.{layer}.hook_resid_post"
    hooks = [(hook_name, make_steering_hook(up_directions[layer], sweep_strength))]
    text = generate_steered(sweep_prompt, hooks)
    print(f"[layer {layer:2d}]")
    show(sweep_prompt, text)
    print()


print("=" * 70)
print("Done.")
print("""
PART A tells us: what happens when the whole model gets the UP signal?
  - Low per-layer strength: gentle warming of the whole system
  - High per-layer strength: the model might go full cosmic / incoherent
  - This is the "nervous system" test — UP everywhere, not just one spot

PART B tells us: where does UP live?
  - Early layers (0-6): probably lexical, might just swap in UP-related words
  - Mid layers (9-15): probably semantic/conceptual, should produce meaning-shifts
  - Late layers (18-23): probably output-head-specific, might produce weird effects
  - If one layer band is clearly stronger, that's where the schema is operative
""")
print("=" * 70)
