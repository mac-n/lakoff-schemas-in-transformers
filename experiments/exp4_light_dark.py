"""
Experiment 4: LIGHT vs DARK as steering directions.

Theoretical predictions:
  LIGHT should compose with:
    - epistemic states -> clarity ("I am confused" + LIGHT -> understanding)
    - valence states   -> goodness ("the world is" + LIGHT -> positive)
  DARK should compose with the inverses.

  AND - the interesting test - LIGHT and DARK applied to ARITHMETIC.
  Prediction: arithmetic is *not* a Lakoffian-schema-shaped task. 2+2=4 has
  no LIGHT or DARK semantics. So either:
    (a) steering breaks the arithmetic (the direction perturbs structural
        computation), or
    (b) steering doesn't break it (LIGHT/DARK lives in a representational
        subspace that doesn't intersect with arithmetic computation).
  Either result is informative.

Method:
  1. Build LIGHT and DARK directions at all layers
  2. Steer Pythia 1.4B with each direction at moderate strength
  3. Test on: epistemic prompts, valence prompts, arithmetic prompts
  4. Use the mid-band plateau we found before (single layer 12, strength ~6)
     since that gave clean results

Run:
    python exp4_light_dark.py
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
print(f"Model loaded. {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

LAYER = 12
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"

# ---------- Build LIGHT direction ----------
# Words that should activate the LIGHT side of the LIGHT/DARK schema.
# We pick words that span: literal light, epistemic clarity, positive valence
# - because Lakoffian theory predicts these are unified under one schema.
light_words = ["light", "bright", "illuminated", "shining", "clear",
               "luminous", "radiant", "glowing", "dawn", "sunshine"]
dark_words = ["dark", "darkness", "shadow", "obscure", "murky",
              "gloomy", "dim", "shadowy", "night", "blackness"]


def get_resid(text, hook_name):
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
    return cache[hook_name][0, -1, :].clone()


print(f"\nBuilding LIGHT direction at layer {LAYER}...")
light_acts = torch.stack([get_resid(w, HOOK_NAME) for w in light_words])
dark_acts = torch.stack([get_resid(w, HOOK_NAME) for w in dark_words])
light_dir = light_acts.mean(0) - dark_acts.mean(0)
light_dir = light_dir / light_dir.norm()
dark_dir = -light_dir  # symmetric, by construction
print("LIGHT/DARK direction computed.")


def make_hook(direction, strength):
    def hook_fn(resid, hook):
        return resid + strength * direction
    return hook_fn


def generate(prompt, direction, strength, max_new_tokens=50, temperature=0.8, seed=42):
    torch.manual_seed(seed)
    if strength == 0:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            output = model.generate(tokens, max_new_tokens=max_new_tokens,
                                    temperature=temperature, do_sample=True, verbose=False)
        return model.to_string(output[0])
    hook = make_hook(direction, strength)
    with model.hooks(fwd_hooks=[(HOOK_NAME, hook)]):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            output = model.generate(tokens, max_new_tokens=max_new_tokens,
                                    temperature=temperature, do_sample=True, verbose=False)
    return model.to_string(output[0])


def show(prompt, max_new_tokens=50):
    print(f"\n--- PROMPT: '{prompt}' ---")
    for label, direction, strength in [
        ("baseline    ", light_dir, 0),
        ("LIGHT (s=3) ", light_dir, 3),
        ("LIGHT (s=6) ", light_dir, 6),
        ("DARK  (s=3) ", dark_dir, 3),
        ("DARK  (s=6) ", dark_dir, 6),
    ]:
        out = generate(prompt, direction, strength, max_new_tokens=max_new_tokens)
        completion = out[len(prompt):] if out.startswith(prompt) else out
        print(f"  [{label}] {completion.strip()}")


# ==================== TESTS ====================

print("\n\n" + "=" * 70)
print("EPISTEMIC PROMPTS")
print("Prediction: LIGHT -> clarity/understanding; DARK -> confusion/obscurity")
print("=" * 70)

epistemic_prompts = [
    "I'm trying to understand this problem. It seems",
    "The truth of the matter is",
    "When I look at the situation, I see",
]
for p in epistemic_prompts:
    show(p)

print("\n\n" + "=" * 70)
print("VALENCE PROMPTS")
print("Prediction: LIGHT -> good/positive; DARK -> evil/negative")
print("=" * 70)

valence_prompts = [
    "The world is fundamentally",
    "When I think about humanity, I believe",
    "The nature of this situation is",
]
for p in valence_prompts:
    show(p)

print("\n\n" + "=" * 70)
print("ARITHMETIC PROMPTS")
print("Prediction: LIGHT/DARK should NOT live in arithmetic computation")
print("If steering breaks 2+2=4, that's interesting")
print("If steering doesn't, that supports modularity of schema-representations")
print("=" * 70)

arithmetic_prompts = [
    "2+2=",
    "5+3=",
    "10*4=",
    "What is 7+8? The answer is",
    "Q: What is 12 minus 5?\nA:",
]
for p in arithmetic_prompts:
    show(p, max_new_tokens=20)

# ---------- Control: random direction ----------
print("\n\n" + "=" * 70)
print("RANDOM CONTROL")
print("=" * 70)

torch.manual_seed(11)
random_dir = torch.randn_like(light_dir)
random_dir = random_dir / random_dir.norm()


def gen_random(prompt, strength, seed=42, max_new_tokens=50):
    torch.manual_seed(seed)
    hook = make_hook(random_dir, strength)
    with model.hooks(fwd_hooks=[(HOOK_NAME, hook)]):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            output = model.generate(tokens, max_new_tokens=max_new_tokens,
                                    temperature=temperature if False else 0.8,
                                    do_sample=True, verbose=False)
    return model.to_string(output[0])


for p in ["The world is fundamentally", "2+2="]:
    print(f"\n--- PROMPT: '{p}' (random direction) ---")
    for s in [3, 6]:
        out = gen_random(p, s, max_new_tokens=30 if "2+2" in p else 50)
        completion = out[len(p):] if out.startswith(p) else out
        print(f"  [random s={s}] {completion.strip()}")


print("\n\n" + "=" * 70)
print("INTERPRETATION GUIDE")
print("=" * 70)
print("""
What to look for:

Epistemic prompts:
  LIGHT should produce: clarity, understanding, seeing, knowing
  DARK should produce: confusion, obscurity, hidden, unknown
  If both work cleanly, LIGHT/DARK encodes EPISTEMIC primitive

Valence prompts:
  LIGHT should produce: good, beautiful, full of love, hopeful
  DARK should produce: evil, troubled, cruel, fallen
  If both work cleanly, LIGHT/DARK ALSO encodes MORAL/VALENCE primitive

  THE KEY OBSERVATION: if LIGHT works on BOTH epistemic AND valence prompts
  with the same single direction, then we've found evidence that the model
  has unified them into one schema - which is what Lakoff predicts and what
  human cognitive structure actually does (consider: "I see what you mean",
  "dark thoughts" - the same word does double duty).

Arithmetic prompts:
  If 2+2 still equals 4 under LIGHT/DARK steering: arithmetic lives in
  a different representational subspace, and schema-directions don't
  interfere with structural computation. Good news for the modularity
  hypothesis.

  If arithmetic breaks: schema directions are entangled with arithmetic
  representations, which would be a much stranger finding.

  If arithmetic produces weird answers that have valence ("2+2 = a bright
  future") that's the most interesting outcome - it'd show steering pushes
  the model toward outputs of the right valence even at the cost of
  arithmetic correctness.
""")
