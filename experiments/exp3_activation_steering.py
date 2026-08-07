"""
Experiment 3: ACTIVATION STEERING

Does the UP direction operate causally during generation, not just exist as
structure in the embeddings?

Method:
  1. Load Pythia 1.4B with TransformerLens
  2. Find the UP direction in the residual stream at a mid-layer by averaging
     activations on UP-words minus DOWN-words
  3. Add that direction (at various strengths) to the residual stream during
     generation on neutral prompts
  4. Compare generated text with/without the intervention

Prediction: steered generations should be more elevated/positive/energetic/
            "looking up" - more cocaine-Gemma-ish, in Niamh's framing

Setup:
    pip install transformer-lens
    (this pulls torch automatically; on M-series Mac it'll use MPS via PyTorch)

Run:
    python exp3_activation_steering.py

Expected runtime:
    - First run: ~5min model download (~3GB) + ~30sec to run experiments
    - Subsequent runs: ~30sec total
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
print("\nLoading Pythia 1.4B (downloads ~3GB on first run)...")
model = HookedTransformer.from_pretrained(
    "pythia-1.4b",
    device=device,
)
model.eval()
print(f"Model loaded. {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

# ---------- Build UP direction in residual stream ----------
# Strategy: for each word in our UP/DOWN sets, get the residual stream activation
# at the final token at a chosen middle layer, then take the difference of means.

# Middle layer for a 24-layer Pythia 1.4B - layer 12 is a reasonable first guess.
# Mid-layers tend to encode abstract features; early = lexical, late = output-specific.
LAYER = 12
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"

up_words = ["up", "rising", "lifting", "ascending", "climbing", "soaring",
            "elevating", "uplifting", "higher", "upward"]
down_words = ["down", "falling", "sinking", "descending", "dropping", "plummeting",
              "lowering", "collapsing", "lower", "downward"]


def get_residual_at_last_token(text, layer_hook_name):
    """Get residual stream activation at the final token of `text` at the given layer."""
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=layer_hook_name)
    # cache[hook_name] has shape [batch, seq, d_model]; take last token
    return cache[layer_hook_name][0, -1, :].clone()


print(f"\nBuilding UP direction at layer {LAYER}...")
up_activations = torch.stack([get_residual_at_last_token(w, HOOK_NAME) for w in up_words])
down_activations = torch.stack([get_residual_at_last_token(w, HOOK_NAME) for w in down_words])

up_direction = up_activations.mean(dim=0) - down_activations.mean(dim=0)
up_direction = up_direction / up_direction.norm()
print(f"UP direction computed. Norm before normalization was a vector in R^{up_direction.shape[0]}")


# ---------- Steering hook ----------

def make_steering_hook(direction, strength):
    """
    Returns a hook function that adds `strength * direction` to the residual
    stream at every position. The hook will be attached at `HOOK_NAME`.
    """
    def hook_fn(resid, hook):
        # resid shape: [batch, seq, d_model]
        return resid + strength * direction
    return hook_fn


# ---------- Generation function ----------

def generate_with_steering(prompt, strength, max_new_tokens=50, temperature=0.8, seed=0):
    """Generate text with the UP direction added to the residual stream."""
    torch.manual_seed(seed)
    if strength == 0:
        # No intervention - just sample
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
    else:
        hook = make_steering_hook(up_direction, strength)
        with model.hooks(fwd_hooks=[(HOOK_NAME, hook)]):
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


# ---------- Run experiments ----------

prompts = [
    "I want to tell you about my day. It was",
    "The weather today is",
    "When I think about the future, I feel",
    "My current mood is best described as",
    "The state of the world right now is",
]

# Strengths to try. Pythia 1.4B residual stream has typical norms in the
# tens-to-hundreds range, so steering strengths of 1-10 are usually meaningful.
# We'll sweep a few.
strengths = [0, 3, 6, 10]

print("\n" + "=" * 70)
print("STEERING EXPERIMENT")
print("=" * 70)

for prompt in prompts:
    print(f"\n{'#' * 70}")
    print(f"PROMPT: '{prompt}'")
    print('#' * 70)
    for strength in strengths:
        # Use same seed within a prompt so the only variation is the intervention
        output = generate_with_steering(prompt, strength, seed=42)
        # Strip prompt prefix for cleaner display
        completion = output[len(prompt):] if output.startswith(prompt) else output
        print(f"\n[strength = {strength}]")
        print(f"  {completion.strip()}")


# ---------- Negative control: random direction ----------
print("\n\n" + "=" * 70)
print("CONTROL: random direction (should produce noise, not 'more up' content)")
print("=" * 70)

torch.manual_seed(7)
random_direction = torch.randn_like(up_direction)
random_direction = random_direction / random_direction.norm()


def generate_with_random_steering(prompt, strength, seed=42):
    torch.manual_seed(seed)
    hook = make_steering_hook(random_direction, strength)
    with model.hooks(fwd_hooks=[(HOOK_NAME, hook)]):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            output = model.generate(
                tokens,
                max_new_tokens=50,
                temperature=0.8,
                do_sample=True,
                verbose=False,
            )
    return model.to_string(output[0])


for prompt in prompts[:2]:  # just first two prompts for the control
    print(f"\nPROMPT: '{prompt}'")
    for strength in [6, 10]:
        output = generate_with_random_steering(prompt, strength)
        completion = output[len(prompt):] if output.startswith(prompt) else output
        print(f"  [random, strength={strength}] {completion.strip()}")


print("\n" + "=" * 70)
print("Done.")
print("""
What to look for:
  - strength=0: baseline, neutral / coherent
  - strength rising: outputs should get progressively more elevated/positive/
    energetic IF the UP direction is operative during generation
  - too high: model will likely become incoherent (over-steering)
  - random direction control: should produce noise or incoherence,
    NOT 'more up' content

If steered outputs are systematically more 'looking-up'-shaped than baseline,
and the random control doesn't do this, then UP is causally operative -
not just present in the embedding geometry but actively used by the model
during generation.
""")
print("=" * 70)
