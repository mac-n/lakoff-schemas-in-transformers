"""
exp19_within_pair_polarity.py - direct polarity measurement at 70m and 410m.

For each (domain, pair_idx), the "within-pair polarity" is:

    cos(UP_offset, DOWN_offset)

where both offsets are taken from the SAME baseline. This directly answers:
"in the same domain, with the same baseline, do UP and DOWN point in opposite
directions in the model's SAE feature space?"

Interpretation:
  - cos ≈ -1: perfect polar opposites (the "ruler" finding Niamh wants)
  - cos ≈  0: orthogonal (UP and DOWN are unrelated transformations)
  - cos ≈ +1: same direction (UP and DOWN both add similar shared change)

Aggregated across all 25 schema pairs at each layer, plotted across scale.

Substrates (NOT directly comparable but same analytical question):
  - 70m: SAELens residual-stream SAE (32k features)
  - 410m: eai-sparsify MLP-output SAE (65k features)

The shift in cos(UP, DOWN) across scale is the load-bearing measurement.
"""

import gc
from collections import defaultdict

import numpy as np
import torch
from transformer_lens import HookedTransformer

# Same matched triples
DOMAINS = {
    "temperature": [
        ("The temperature was constant throughout the day.",
         "The temperature rose throughout the day.",
         "The temperature plunged throughout the day."),
        ("The thermometer reading held steady overnight.",
         "The thermometer reading climbed overnight.",
         "The thermometer reading plummeted overnight."),
        ("Room temperature stayed the same all afternoon.",
         "Room temperature soared all afternoon.",
         "Room temperature sank all afternoon."),
        ("The water temperature was even before boiling.",
         "The water temperature ascended past boiling.",
         "The water temperature descended past freezing."),
        ("The forecast showed stable temperatures this week.",
         "The forecast showed increasing temperatures this week.",
         "The forecast showed tumbling temperatures this week."),
    ],
    "mood": [
        ("Her mood was neutral after the meeting.",
         "Her mood was elated after the meeting.",
         "Her mood was dejected after the meeting."),
        ("He felt nothing about the news.",
         "He felt jubilant about the news.",
         "He felt morose about the news."),
        ("Their spirits were average that morning.",
         "Their spirits were radiant that morning.",
         "Their spirits were forlorn that morning."),
        ("She seemed unchanged by the praise.",
         "She seemed ecstatic from the praise.",
         "She seemed glum despite the praise."),
        ("The audience reacted plainly to the song.",
         "The audience reacted joyfully to the song.",
         "The audience reacted somberly to the song."),
    ],
    "quantity": [
        ("The company's revenue was stable last quarter.",
         "The company's revenue grew last quarter.",
         "The company's revenue declined last quarter."),
        ("Sales held steady through summer.",
         "Sales increased through summer.",
         "Sales decreased through summer."),
        ("The population stayed flat over the decade.",
         "The population multiplied over the decade.",
         "The population dwindled over the decade."),
        ("Inventory remained constant this month.",
         "Inventory expanded this month.",
         "Inventory shrank this month."),
        ("Subscribers stayed the same all year.",
         "Subscribers accrued all year.",
         "Subscribers waned all year."),
    ],
    "status": [
        ("Her position was unchanged at the firm.",
         "Her position was promoted at the firm.",
         "Her position was demoted at the firm."),
        ("His reputation remained the same in the field.",
         "His reputation became prominent in the field.",
         "His reputation was disgraced in the field."),
        ("She held the same rank for years.",
         "She held a distinguished rank for years.",
         "She was ousted from her rank."),
        ("His standing was ordinary among peers.",
         "His standing was esteemed among peers.",
         "His standing was discredited among peers."),
        ("The professor's recognition was middling.",
         "The professor's recognition was prestigious.",
         "The professor's recognition was dethroned."),
    ],
    "health": [
        ("His condition was stable yesterday.",
         "His condition was thriving yesterday.",
         "His condition was ailing yesterday."),
        ("Her vitality stayed even after the surgery.",
         "Her vitality returned vigorously after the surgery.",
         "Her vitality deteriorated after the surgery."),
        ("The patient remained unchanged.",
         "The patient was recuperating.",
         "The patient was languishing."),
        ("The plants looked the same in the garden.",
         "The plants looked robust in the garden.",
         "The plants looked sickly in the garden."),
        ("His energy was average that week.",
         "His energy was vital that week.",
         "His energy was feeble that week."),
    ],
}

# Flatten: list of (domain, idx, baseline, up, down)
schema_triples = []
for domain, triples in DOMAINS.items():
    for i, (b, u, d) in enumerate(triples):
        schema_triples.append((domain, i, b, u, d))
print(f"Schema triples: {len(schema_triples)}")


def cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---- Pythia 70m ----
print("\n" + "=" * 70)
print("PYTHIA 70m (residual-stream SAEs from SAELens)")
print("=" * 70)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

from sae_lens import SAE

print("\nLoading Pythia 70m-deduped...")
model70 = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
model70.eval()
n_layers_70 = model70.cfg.n_layers


def collect_70m(text, hook):
    tokens = model70.to_tokens(text)
    if tokens.shape[1] > 64:
        tokens = tokens[:, :64]
    with torch.no_grad():
        _, cache = model70.run_with_cache(tokens, names_filter=hook)
    return cache[hook][0].cpu().float()


results_70 = {}
for layer in range(n_layers_70):
    print(f"\n--- 70m Layer {layer} ---")
    hook = f"blocks.{layer}.hook_resid_post"
    sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=hook, device="cpu")
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res

    within_pair_cosines = []
    for domain, idx, baseline, up, down in schema_triples:
        b_acts = collect_70m(baseline, hook)
        u_acts = collect_70m(up, hook)
        d_acts = collect_70m(down, hook)
        with torch.no_grad():
            b_feat = sae.encode(b_acts).max(0).values.numpy().astype(np.float64)
            u_feat = sae.encode(u_acts).max(0).values.numpy().astype(np.float64)
            d_feat = sae.encode(d_acts).max(0).values.numpy().astype(np.float64)
        up_off = u_feat - b_feat
        dn_off = d_feat - b_feat
        within_pair_cosines.append(cos(up_off, dn_off))

    results_70[layer] = {
        "mean": float(np.mean(within_pair_cosines)),
        "median": float(np.median(within_pair_cosines)),
        "min": float(np.min(within_pair_cosines)),
        "max": float(np.max(within_pair_cosines)),
        "n": len(within_pair_cosines),
    }
    r = results_70[layer]
    print(f"  within-pair cos(UP, DOWN): mean={r['mean']:+.4f}, median={r['median']:+.4f}, range=[{r['min']:+.4f}, {r['max']:+.4f}]")

    del sae
    gc.collect()

del model70
gc.collect()
if device == "mps":
    torch.mps.empty_cache()


# ---- Pythia 410m ----
print("\n" + "=" * 70)
print("PYTHIA 410m (MLP-output SAEs from eai-sparsify)")
print("=" * 70)

from sparsify import Sae as SparsifySae

print("\nLoading Pythia 410m via TransformerLens...")
model410 = HookedTransformer.from_pretrained("EleutherAI/pythia-410m", device=device)
model410.eval()
n_layers_410 = model410.cfg.n_layers

print("Loading EleutherAI/sae-pythia-410m-65k...")
saes_410 = SparsifySae.load_many("EleutherAI/sae-pythia-410m-65k")


def collect_410m_mlp(text, layer):
    hook = f"blocks.{layer}.hook_mlp_out"
    tokens = model410.to_tokens(text)
    if tokens.shape[1] > 64:
        tokens = tokens[:, :64]
    with torch.no_grad():
        _, cache = model410.run_with_cache(tokens, names_filter=hook)
    mlp = cache[hook][0]
    sae = saes_410[f"layers.{layer}.mlp"]
    with torch.no_grad():
        latents = sae.encode(mlp.cpu().float())
        # Handle TopK SAE output
        if hasattr(latents, 'top_indices') and hasattr(latents, 'top_acts'):
            n_features = sae.num_latents
            seq_len = mlp.shape[0]
            full = torch.zeros(seq_len, n_features)
            for s in range(seq_len):
                full[s, latents.top_indices[s]] = latents.top_acts[s].float()
            return full.max(0).values.numpy().astype(np.float64)
        else:
            return latents.max(0).values.cpu().numpy().astype(np.float64)


results_410 = {}
for layer in range(n_layers_410):
    print(f"\n--- 410m Layer {layer} ---")
    if f"layers.{layer}.mlp" not in saes_410:
        continue
    within_pair_cosines = []
    for domain, idx, baseline, up, down in schema_triples:
        b_feat = collect_410m_mlp(baseline, layer)
        u_feat = collect_410m_mlp(up, layer)
        d_feat = collect_410m_mlp(down, layer)
        up_off = u_feat - b_feat
        dn_off = d_feat - b_feat
        within_pair_cosines.append(cos(up_off, dn_off))

    results_410[layer] = {
        "mean": float(np.mean(within_pair_cosines)),
        "median": float(np.median(within_pair_cosines)),
        "min": float(np.min(within_pair_cosines)),
        "max": float(np.max(within_pair_cosines)),
        "n": len(within_pair_cosines),
    }
    r = results_410[layer]
    print(f"  within-pair cos(UP, DOWN): mean={r['mean']:+.4f}, median={r['median']:+.4f}, range=[{r['min']:+.4f}, {r['max']:+.4f}]")


# ---- Report ----
report_path = "/Users/macn/Documents/embeddingexp/results_exp19_within_pair_polarity.md"
with open(report_path, "w") as f:
    def out(s=""):
        print(s)
        f.write(s + "\n")

    out("# exp19 — within-pair polarity of UP and DOWN, across scale")
    out()
    out("Per-pair cosine similarity between UP_offset and DOWN_offset, taken from")
    out("the SAME baseline within the same (domain, pair_idx). Measures whether the")
    out("model represents UP and DOWN as polar opposites within a single domain.")
    out()
    out("- cos ≈ -1: perfect antipolarity (the 'ruler' finding)")
    out("- cos ≈  0: orthogonal (unrelated transformations)")
    out("- cos ≈ +1: same direction (shared 'change' component dominates)")
    out()
    out("## Pythia 70m (6 layers, residual-stream SAE, 32k features)")
    out()
    out(f"  {'layer':>5} {'mean':>9} {'median':>9} {'min':>9} {'max':>9}  n=25")
    for layer in sorted(results_70.keys()):
        r = results_70[layer]
        out(f"  {layer:>5d} {r['mean']:>+9.4f} {r['median']:>+9.4f} {r['min']:>+9.4f} {r['max']:>+9.4f}")
    out()
    mean_70 = float(np.mean([r["mean"] for r in results_70.values()]))
    out(f"Mean across all layers (70m): **{mean_70:+.4f}**")
    out()
    out("## Pythia 410m (24 layers, MLP-output SAE, 65k features)")
    out()
    out(f"  {'layer':>5} {'mean':>9} {'median':>9} {'min':>9} {'max':>9}  n=25")
    for layer in sorted(results_410.keys()):
        r = results_410[layer]
        out(f"  {layer:>5d} {r['mean']:>+9.4f} {r['median']:>+9.4f} {r['min']:>+9.4f} {r['max']:>+9.4f}")
    out()
    mean_410 = float(np.mean([r["mean"] for r in results_410.values()]))
    out(f"Mean across all layers (410m): **{mean_410:+.4f}**")
    out()
    out("## Scale comparison")
    out()
    out(f"  Pythia 70m:  mean within-pair cos(UP, DOWN) across all layers = {mean_70:+.4f}")
    out(f"  Pythia 410m: mean within-pair cos(UP, DOWN) across all layers = {mean_410:+.4f}")
    out(f"  Δ (410m − 70m) = {mean_410 - mean_70:+.4f}")
    out()
    out("**Interpretation:**")
    out("- If 410m mean is MORE NEGATIVE than 70m: ruler is emerging with scale.")
    out("- If 410m mean is MORE POSITIVE: polar opposite structure regresses (or noise).")
    out("- If similar: no scale effect at this range; would need bigger models.")

print(f"\nReport: {report_path}")

torch.save({
    "results_70": results_70,
    "results_410": results_410,
}, "/Users/macn/Documents/embeddingexp/exp19_results.pt")
