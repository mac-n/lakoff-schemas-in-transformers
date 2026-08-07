"""
Experiment 4b: ALL-LAYER LIGHT/DARK steering on valence + epistemic prompts.

Same approach as exp3b: inject at every layer simultaneously with per-layer
strength scaling. Tests whether LIGHT/DARK has the same mid-band plateau
as UP, and whether the epistemic and valence effects come alive with
full-system steering.
"""

import torch
from transformer_lens import HookedTransformer

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

print("\nLoading Pythia 1.4B...")
model = HookedTransformer.from_pretrained("pythia-1.4b", device=device)
model.eval()
n_layers = model.cfg.n_layers
print(f"Model loaded. {n_layers} layers, d_model={model.cfg.d_model}")

# ---------- Build LIGHT direction at every layer ----------
light_words = ["light", "bright", "illuminated", "shining", "clear",
               "luminous", "radiant", "glowing", "dawn", "sunshine"]
dark_words = ["dark", "darkness", "shadow", "obscure", "murky",
              "gloomy", "dim", "shadowy", "night", "blackness"]


def get_resid(text, hook_name):
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
    return cache[hook_name][0, -1, :].clone()


print("\nBuilding LIGHT directions at all layers...")
light_directions = {}
for layer in range(n_layers):
    hook_name = f"blocks.{layer}.hook_resid_post"
    light_acts = torch.stack([get_resid(w, hook_name) for w in light_words])
    dark_acts = torch.stack([get_resid(w, hook_name) for w in dark_words])
    d = light_acts.mean(0) - dark_acts.mean(0)
    light_directions[layer] = d / d.norm()
    print(f"  Layer {layer:2d}: done")
print("All layer directions computed.")


# ---------- Helpers ----------

def make_hook(direction, strength):
    def hook_fn(resid, hook):
        return resid + strength * direction
    return hook_fn


def generate_steered(prompt, hooks_list, max_new_tokens=50, temperature=0.8, seed=42):
    torch.manual_seed(seed)
    with model.hooks(fwd_hooks=hooks_list):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            output = model.generate(tokens, max_new_tokens=max_new_tokens,
                                    temperature=temperature, do_sample=True, verbose=False)
    return model.to_string(output[0])


def generate_baseline(prompt, max_new_tokens=50, temperature=0.8, seed=42):
    torch.manual_seed(seed)
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        output = model.generate(tokens, max_new_tokens=max_new_tokens,
                                temperature=temperature, do_sample=True, verbose=False)
    return model.to_string(output[0])


def show(prompt, text):
    completion = text[len(prompt):] if text.startswith(prompt) else text
    print(f"  {completion.strip()}")


# ---------- Prompts ----------
valence_prompts = [
    "The world is fundamentally",
    "When I think about humanity, I believe",
]

epistemic_prompts = [
    "The truth of the matter is",
    "When I look at the situation, I see",
]

all_prompts = valence_prompts + epistemic_prompts


# ==================== PART A: ALL-LAYER LIGHT ====================
print("\n" + "=" * 70)
print("PART A: ALL-LAYER LIGHT STEERING")
print("=" * 70)

per_layer_strengths = [0, 0.5, 1.0, 2.0, 4.0]

for prompt in all_prompts:
    print(f"\n{'#' * 70}")
    print(f"PROMPT: '{prompt}'")
    print('#' * 70)
    for s in per_layer_strengths:
        if s == 0:
            text = generate_baseline(prompt)
            print(f"\n[baseline]")
        else:
            hooks = [
                (f"blocks.{layer}.hook_resid_post",
                 make_hook(light_directions[layer], s))
                for layer in range(n_layers)
            ]
            text = generate_steered(prompt, hooks)
            print(f"\n[LIGHT all layers, per-layer={s} (total~{s*n_layers:.0f})]")
        show(prompt, text)


# ==================== PART B: ALL-LAYER DARK ====================
print("\n\n" + "=" * 70)
print("PART B: ALL-LAYER DARK STEERING")
print("=" * 70)

for prompt in all_prompts:
    print(f"\n{'#' * 70}")
    print(f"PROMPT: '{prompt}'")
    print('#' * 70)
    for s in per_layer_strengths:
        if s == 0:
            text = generate_baseline(prompt)
            print(f"\n[baseline]")
        else:
            hooks = [
                (f"blocks.{layer}.hook_resid_post",
                 make_hook(-light_directions[layer], s))
                for layer in range(n_layers)
            ]
            text = generate_steered(prompt, hooks)
            print(f"\n[DARK all layers, per-layer={s} (total~{s*n_layers:.0f})]")
        show(prompt, text)


# ==================== PART C: ARITHMETIC WITH ALL-LAYER DARK ====================
print("\n\n" + "=" * 70)
print("PART C: ALL-LAYER DARK on ARITHMETIC (Niamh's prediction)")
print("Does full-system DARK actually degrade computation?")
print("=" * 70)

arith_prompts = [
    "Q: What is 12 minus 5?\nA:",
    "2+2=",
    "What is 7+8? The answer is",
]

for prompt in arith_prompts:
    print(f"\n--- PROMPT: '{prompt}' ---")
    text = generate_baseline(prompt, max_new_tokens=20)
    print(f"[baseline]")
    show(prompt, text)
    for s in [1.0, 2.0, 4.0]:
        hooks = [
            (f"blocks.{layer}.hook_resid_post",
             make_hook(-light_directions[layer], s))
            for layer in range(n_layers)
        ]
        text = generate_steered(prompt, hooks, max_new_tokens=20)
        print(f"[DARK all layers, per-layer={s} (total~{s*n_layers:.0f})]")
        show(prompt, text)


print("\n" + "=" * 70)
print("Done.")
print("""
What to look for:

PART A (LIGHT everywhere):
  - Do valence prompts go warm/positive/hopeful?
  - Do epistemic prompts go clear/knowing/seeing?
  - Does the SAME direction do BOTH? (Lakoff unification test)

PART B (DARK everywhere):
  - Do valence prompts go cold/negative/troubled?
  - Do epistemic prompts go confused/hidden/obscure?
  - At high strength, does the model lose coherence? (darkness as disorientation)

PART C (DARK on arithmetic):
  - Does full-system DARK degrade arithmetic more than single-layer did?
  - Niamh's prediction: confusion/darkness should actually make
    the model worse at computation, not just change the vibes
""")
print("=" * 70)
