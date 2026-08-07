"""
exp112_diag2.py — find the bug that gives cos(up,down)=1.000.
"""
import torch
from transformer_lens import HookedTransformer

device = "mps"
print("Loading Pythia 1.4B...")
model = HookedTransformer.from_pretrained("pythia-1.4b", device=device)
model.eval()
LAYER = 12
HOOK = f"blocks.{LAYER}.hook_resid_post"

# (1) What do these tokenize to?
print("\n--- (1) Tokenizations ---")
for w in ["up", "down", "rise", "fall", "high", "low", "peak", "valley",
         "ceiling", "floor", "ascend", "descend"]:
    toks = model.to_tokens(w)
    tok_strs = [model.tokenizer.decode([t]) for t in toks[0].tolist()]
    print(f"  '{w}': tokens={toks[0].tolist()} = {tok_strs}")

# (2) Print residuals at last position for "up" and "down" — first 10 elements + norm
print("\n--- (2) Actual residuals at L12, last position ---")
for w in ["up", "down", "rise", "fall", "high", "low"]:
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=HOOK)
    r = cache[HOOK][0, -1, :].clone()
    print(f"  '{w}': shape={r.shape}, ‖r‖={r.norm().item():.3f}, r[:5]={r[:5].tolist()}")

# (3) Maybe the issue is the LAST position is BOS, not the word.
# Print residuals at EACH position for a multi-token input.
print("\n--- (3) Full sequence of residuals for 'up' ---")
toks = model.to_tokens("up")
print(f"  Tokens: {toks[0].tolist()}  ({[model.tokenizer.decode([t]) for t in toks[0].tolist()]})")
with torch.no_grad():
    _, cache = model.run_with_cache(toks, names_filter=HOOK)
r = cache[HOOK][0]  # [seq, d_model]
for i in range(r.shape[0]):
    print(f"  pos {i} (tok={toks[0,i].item()}={model.tokenizer.decode([toks[0,i].item()])!r}): "
          f"‖r‖={r[i].norm().item():.3f}, r[:5]={r[i,:5].tolist()}")

# (4) Is this an MPS bug? Run the same single-word check on CPU.
print("\n--- (4) CPU re-check: cos(up, down) on CPU ---")
model_cpu = HookedTransformer.from_pretrained("pythia-1.4b", device="cpu")
model_cpu.eval()
def res_cpu(w):
    toks = model_cpu.to_tokens(w)
    with torch.no_grad():
        _, cache = model_cpu.run_with_cache(toks, names_filter=HOOK)
    return cache[HOOK][0, -1, :].clone()
ru = res_cpu("up"); rd = res_cpu("down")
print(f"  ‖res_cpu('up')‖={ru.norm().item():.4f}, ‖res_cpu('down')‖={rd.norm().item():.4f}")
print(f"  cos_cpu(up, down) = {(ru @ rd / (ru.norm() * rd.norm())).item():+.6f}")
ru2, rd2 = res_cpu("rise"), res_cpu("fall")
print(f"  cos_cpu(rise, fall) = {(ru2 @ rd2 / (ru2.norm() * rd2.norm())).item():+.6f}")
ru3 = res_cpu("peak"); rd3 = res_cpu("valley")
print(f"  cos_cpu(peak, valley) = {(ru3 @ rd3 / (ru3.norm() * rd3.norm())).item():+.6f}")
