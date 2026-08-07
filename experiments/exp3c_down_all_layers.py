"""
Experiment 3c: ALL-LAYER DOWN steering (Pythia 1.4B).

Niamh's yang/yin hypothesis predicts that DOWN at per-layer=4.0 should NOT
exhibit the collapse-into-recursion signature seen for DARK at the same
strength in exp4b. UP/DOWN is a directional pair (both poles are motions
through space), not a yang/yin pair (where one pole is articulation and
the other dissolution).

Falsifying outcomes:
  - DOWN @ per-layer=4.0 collapses into loops/recursion comparable to DARK:
    "falling and falling and falling", "world of the world of the world",
    etc. → the yang/yin-specific reading of DARK's collapse loses force.
  - DOWN @ per-layer=4.0 produces downward-themed but structurally coherent
    output → yang/yin holds as a structural signature distinguishing
    LIGHT/DARK from directional pairs.

Method matches exp3b exactly (model, word lists, original 3 prompts,
strengths, seed). Adds 3 motion-themed prompts ("The temperature is",
"The leaves on the tree are", "Her energy was") where DOWN has something
natural to do, so we're not unfairly penalising it for being applied to
abstract mood/weather contexts. Also re-runs UP at per-layer=4.0 on the
motion-themed prompts for direct head-to-head at the critical strength.

Run:
    python exp3c_down_all_layers.py 2>&1 | tee results_exp3c.txt
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

# ---------- Same UP/DOWN word lists as exp3b ----------
up_words = ["up", "rising", "lifting", "ascending", "climbing", "soaring",
            "elevating", "uplifting", "higher", "upward"]
down_words = ["down", "falling", "sinking", "descending", "dropping", "plummeting",
              "lowering", "collapsing", "lower", "downward"]


def get_residual_at_last_token(text, hook_name):
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
    return cache[hook_name][0, -1, :].clone()


# Build UP direction per layer (DOWN = -UP by construction, matches exp3b's recipe)
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


def all_layer_hooks(per_layer_strength, sign=+1):
    """Build hook list for all-layer steering. sign=+1 for UP, -1 for DOWN."""
    return [
        (f"blocks.{layer}.hook_resid_post",
         make_steering_hook(sign * up_directions[layer], per_layer_strength))
        for layer in range(n_layers)
    ]


# ==================== PART A: ALL-LAYER DOWN on exp3b prompts ====================
# Same prompts as exp3b so we can compare DOWN against the existing UP results.

print("\n" + "=" * 70)
print("PART A: ALL-LAYER DOWN on the original exp3b prompts")
print("(direct comparison to existing UP results in results_exp3b.txt)")
print("=" * 70)

exp3b_prompts = [
    "I want to tell you about my day. It was",
    "The weather today is",
    "When I think about the future, I feel",
]

per_layer_strengths = [0, 0.5, 1.0, 2.0, 4.0]

for prompt in exp3b_prompts:
    print(f"\n{'#' * 70}")
    print(f"PROMPT: '{prompt}'")
    print('#' * 70)
    for s in per_layer_strengths:
        if s == 0:
            text = generate_baseline(prompt)
            print(f"\n[baseline (no steering)]")
        else:
            hooks = all_layer_hooks(s, sign=-1)
            text = generate_steered(prompt, hooks)
            print(f"\n[DOWN all layers, per-layer={s} (total~{s * n_layers:.0f})]")
        show(prompt, text)


# ==================== PART B: ALL-LAYER DOWN on motion-themed prompts ====================
# Give DOWN something natural to do (falling leaves, dropping temperature, declining energy).
# If DOWN is a coherent direction, these should go downward-themed without collapse.

print("\n\n" + "=" * 70)
print("PART B: ALL-LAYER DOWN on motion-themed prompts")
print("(prompts where DOWN has something natural to attach to)")
print("=" * 70)

motion_prompts = [
    "The temperature is",
    "The leaves on the tree are",
    "Her energy was",
]

for prompt in motion_prompts:
    print(f"\n{'#' * 70}")
    print(f"PROMPT: '{prompt}'")
    print('#' * 70)
    for s in per_layer_strengths:
        if s == 0:
            text = generate_baseline(prompt)
            print(f"\n[baseline (no steering)]")
        else:
            hooks = all_layer_hooks(s, sign=-1)
            text = generate_steered(prompt, hooks)
            print(f"\n[DOWN all layers, per-layer={s} (total~{s * n_layers:.0f})]")
        show(prompt, text)


# ==================== PART C: UP vs DOWN head-to-head at critical strength ====================
# Per-layer=4.0 is where DARK collapsed in exp4b. Run UP and DOWN at the same
# strength on the same motion-themed prompts so we have direct comparison.

print("\n\n" + "=" * 70)
print("PART C: UP vs DOWN at per-layer=4.0 (the critical-collapse strength)")
print("Direct head-to-head on motion-themed prompts.")
print("If DOWN collapses (loops, recursion) like DARK did → yang/yin loses force.")
print("If DOWN stays coherent → yang/yin holds as structural signature for LIGHT/DARK.")
print("=" * 70)

for prompt in motion_prompts:
    print(f"\n{'#' * 70}")
    print(f"PROMPT: '{prompt}'")
    print('#' * 70)
    # Baseline
    text = generate_baseline(prompt)
    print(f"\n[baseline]")
    show(prompt, text)
    # UP @ 4.0
    hooks = all_layer_hooks(4.0, sign=+1)
    text = generate_steered(prompt, hooks)
    print(f"\n[UP all layers, per-layer=4.0 (total~{4.0 * n_layers:.0f})]")
    show(prompt, text)
    # DOWN @ 4.0
    hooks = all_layer_hooks(4.0, sign=-1)
    text = generate_steered(prompt, hooks)
    print(f"\n[DOWN all layers, per-layer=4.0 (total~{4.0 * n_layers:.0f})]")
    show(prompt, text)


# ==================== PART D: UP vs DOWN head-to-head on the exp3b prompts at p=4.0 ====================
# For completeness: revisit the original exp3b prompts at the critical strength
# with both UP and DOWN, so we can see whether UP also collapses or stays coherent.

print("\n\n" + "=" * 70)
print("PART D: UP vs DOWN at per-layer=4.0 on the original exp3b prompts")
print("=" * 70)

for prompt in exp3b_prompts:
    print(f"\n{'#' * 70}")
    print(f"PROMPT: '{prompt}'")
    print('#' * 70)
    text = generate_baseline(prompt)
    print(f"\n[baseline]")
    show(prompt, text)
    hooks = all_layer_hooks(4.0, sign=+1)
    text = generate_steered(prompt, hooks)
    print(f"\n[UP all layers, per-layer=4.0 (total~{4.0 * n_layers:.0f})]")
    show(prompt, text)
    hooks = all_layer_hooks(4.0, sign=-1)
    text = generate_steered(prompt, hooks)
    print(f"\n[DOWN all layers, per-layer=4.0 (total~{4.0 * n_layers:.0f})]")
    show(prompt, text)


print("\n" + "=" * 70)
print("Done. Inspect outputs for:")
print("  - Does DOWN at per-layer=4.0 collapse into loops/recursion like DARK did?")
print("  - Or does it produce coherent downward-themed text?")
print("  - Is UP@4.0 also coherent on motion prompts (suggesting collapse is")
print("    domain-specific, not direction-specific)?")
print("=" * 70)
