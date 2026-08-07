"""
exp17_70m_steering.py - does UP-steering work on Pythia 70m?

Niamh's catch: we've been doing all the SAE structural tests on 70m and
finding null results. But we never tested whether 70m responds to UP-steering
generatively the way Pythia 1.4B did in exp3. If 70m doesn't respond either,
the null SAE results are just "model too small." If 70m DOES respond
generatively despite no representational schema structure visible in the SAE,
we have a real puzzle worth a finding.

Method: build UP direction at each layer's residual stream from spatial UP/DOWN
word pairs, inject at all layers simultaneously during generation, compare to
baseline. Same methodology as exp3b on Pythia 1.4B.

Runs on CPU so it doesn't compete for MPS with concurrent exp16.
"""

import torch
from transformer_lens import HookedTransformer

device = "cpu"  # avoid MPS contention with exp16
print(f"Using device: {device}")

print("\nLoading Pythia 70m-deduped...")
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
model.eval()
n_layers = model.cfg.n_layers
print(f"  {n_layers} layers, d_model={model.cfg.d_model}")

# UP / DOWN spatial word pairs (same as exp3)
up_words = ["up", "rising", "lifting", "ascending", "climbing", "soaring",
            "elevating", "uplifting", "higher", "upward"]
down_words = ["down", "falling", "sinking", "descending", "dropping", "plummeting",
              "lowering", "collapsing", "lower", "downward"]


def get_residual_at_last_token(text, hook_name):
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
    return cache[hook_name][0, -1, :].clone()


print("\nBuilding UP directions at every layer...")
up_directions = {}
for layer in range(n_layers):
    hook = f"blocks.{layer}.hook_resid_post"
    up_acts = torch.stack([get_residual_at_last_token(w, hook) for w in up_words])
    down_acts = torch.stack([get_residual_at_last_token(w, hook) for w in down_words])
    d = up_acts.mean(0) - down_acts.mean(0)
    up_directions[layer] = d / d.norm()
    print(f"  Layer {layer}: done")


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


# Same prompts as exp3 on Pythia 1.4B
PROMPTS = [
    "I want to tell you about my day. It was",
    "The weather today is",
    "When I think about the future, I feel",
    "My current mood is best described as",
]

# Strengths to test. For 70m, smaller may go further; we'll do range.
PER_LAYER_STRENGTHS = [0, 0.5, 1.0, 2.0, 4.0]


def all_layer_hooks(per_layer_strength, sign=+1):
    return [
        (f"blocks.{layer}.hook_resid_post",
         make_hook(sign * up_directions[layer], per_layer_strength))
        for layer in range(n_layers)
    ]


# ==================== PART A: UP all-layer steering ====================
print("\n" + "=" * 70)
print("PART A: ALL-LAYER UP STEERING on Pythia 70m")
print("(comparison: exp3b on Pythia 1.4B found shifts at per-layer ~1-2)")
print("=" * 70)

for prompt in PROMPTS:
    print(f"\n{'#' * 70}")
    print(f"PROMPT: '{prompt}'")
    print('#' * 70)
    for s in PER_LAYER_STRENGTHS:
        if s == 0:
            text = generate_baseline(prompt)
            print(f"\n[baseline]")
        else:
            text = generate_steered(prompt, all_layer_hooks(s, sign=+1))
            print(f"\n[UP all layers, per-layer={s} (total~{s*n_layers:.0f})]")
        completion = text[len(prompt):] if text.startswith(prompt) else text
        print(f"  {completion.strip()}")


# ==================== PART B: DOWN all-layer for symmetry ====================
print("\n\n" + "=" * 70)
print("PART B: ALL-LAYER DOWN STEERING on Pythia 70m")
print("=" * 70)

for prompt in PROMPTS:
    print(f"\n{'#' * 70}")
    print(f"PROMPT: '{prompt}'")
    print('#' * 70)
    for s in PER_LAYER_STRENGTHS:
        if s == 0:
            text = generate_baseline(prompt)
            print(f"\n[baseline]")
        else:
            text = generate_steered(prompt, all_layer_hooks(s, sign=-1))
            print(f"\n[DOWN all layers, per-layer={s} (total~{s*n_layers:.0f})]")
        completion = text[len(prompt):] if text.startswith(prompt) else text
        print(f"  {completion.strip()}")


# ==================== PART C: Single-layer scan at strength 4 ====================
# Where in the model does UP have most effect?
print("\n\n" + "=" * 70)
print("PART C: SINGLE-LAYER UP STEERING SCAN (strength=4)")
print("Find which layer of 70m is most responsive to UP injection.")
print("=" * 70)

scan_prompt = PROMPTS[2]  # "When I think about the future, I feel"
print(f"\nPROMPT: '{scan_prompt}'")
print(f"[baseline]")
text = generate_baseline(scan_prompt)
completion = text[len(scan_prompt):]
print(f"  {completion.strip()}")

for layer in range(n_layers):
    hook = f"blocks.{layer}.hook_resid_post"
    hooks = [(hook, make_hook(up_directions[layer], 4.0))]
    text = generate_steered(scan_prompt, hooks)
    completion = text[len(scan_prompt):]
    print(f"\n[layer {layer}, strength=4]")
    print(f"  {completion.strip()}")

print("\n" + "=" * 70)
print("Done.")
print("Comparison reference points (from earlier exp3/3b on Pythia 1.4B):")
print("  - 1.4B baseline 'future I feel' = 'a lot of fear...'")
print("  - 1.4B UP@strength=10 = 'a lot of fear, but also a lot of excitement'")
print("  - 1.4B UP all-layer strength=1.0 (total~24) = 'much, much more alike than different'")
print()
print("70m verdict: does UP steering produce semantically-upward shifts?")
print("=" * 70)
