"""
exp20_within_pair_1b.py - within-pair cos(UP, DOWN) at Pythia 1B (one layer).

Continuing the scale comparison:
  - 70m: mean +0.5546 across 6 layers (exp19)
  - 410m: mean +0.5409 across 24 layers (exp19)
  - 1B: ? at layer 11 of 16 (only layer with SAE available via timhua/pythia1b_deduped_saes)

For comparable layer-depth comparison, 1B L11 corresponds roughly to:
  - 70m L4 (~67% through) which had mean +0.4915
  - 410m L17-18 (~71-75% through) which had mean +0.5358 to +0.5554

If 1B L11 is close to those values, scale doesn't shift polarity within this range.
If 1B L11 drops to e.g. +0.2, ruler is emerging at scale.
"""

from collections import defaultdict

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

# Same matched triples as exp19
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


# ---- Device ----
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# ---- Load Pythia 1B ----
print("\nLoading Pythia 1B-deduped...")
model = HookedTransformer.from_pretrained("EleutherAI/pythia-1b-deduped", device=device)
model.eval()
n_layers = model.cfg.n_layers
print(f"  {n_layers} layers, d_model={model.cfg.d_model}")

# ---- Load SAE for layer 11 ----
LAYER = 11
HOOK = f"blocks.{LAYER}.hook_resid_post"
print(f"\nLoading SAE: timhua/pythia1b_deduped_saes, hook={HOOK}")
sae_res = SAE.from_pretrained(
    release="timhua/pythia1b_deduped_saes",
    sae_id="pythia1b_614mtoks",
    device="cpu",
)
sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res
print(f"  d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")


def collect_features(text):
    tokens = model.to_tokens(text)
    if tokens.shape[1] > 64:
        tokens = tokens[:, :64]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=HOOK)
    acts = cache[HOOK][0].cpu().float()  # (seq_len, d_model)
    with torch.no_grad():
        feat_acts = sae.encode(acts)  # (seq_len, n_features)
    return feat_acts.max(0).values.numpy().astype(np.float64)


# ---- Compute within-pair cosines ----
print(f"\nComputing within-pair cos(UP, DOWN) at layer {LAYER}...")
within_pair = []
per_triple = []
for ti, (domain, idx, baseline, up, down) in enumerate(schema_triples):
    if ti % 5 == 0:
        print(f"  {ti}/{len(schema_triples)}")
    b_feat = collect_features(baseline)
    u_feat = collect_features(up)
    d_feat = collect_features(down)
    up_off = u_feat - b_feat
    dn_off = d_feat - b_feat
    c = cos(up_off, dn_off)
    within_pair.append(c)
    per_triple.append((domain, idx, c))

mean_cos = float(np.mean(within_pair))
median_cos = float(np.median(within_pair))
min_cos = float(np.min(within_pair))
max_cos = float(np.max(within_pair))

print()
print("=" * 70)
print(f"PYTHIA 1B (1 layer covered: layer {LAYER} of {n_layers}, residual-stream JumpReLU SAE)")
print("=" * 70)
print(f"  within-pair cos(UP, DOWN) at layer {LAYER}:")
print(f"    mean   = {mean_cos:+.4f}")
print(f"    median = {median_cos:+.4f}")
print(f"    range  = [{min_cos:+.4f}, {max_cos:+.4f}]")
print()
print("=" * 70)
print("SCALE COMPARISON (within-pair cos(UP, DOWN), comparable depths)")
print("=" * 70)
print(f"  70m  L4 (67% through): +0.4915  (from exp19)")
print(f"  410m L17-18 (71-75%):  +0.5358 to +0.5554  (from exp19)")
print(f"  1B   L11 (69% through): {mean_cos:+.4f}  (this experiment)")
print()
print("All-layer means from exp19:")
print(f"  70m  (6 layers):  +0.5546")
print(f"  410m (24 layers): +0.5409")
print(f"  1B   (1 layer):   {mean_cos:+.4f}")

# Report
report_path = "/Users/macn/Documents/embeddingexp/results_exp20_within_pair_1b.md"
with open(report_path, "w") as f:
    f.write("# exp20 — within-pair cos(UP, DOWN) at Pythia 1B layer 11\n\n")
    f.write("Single layer of SAE coverage at 1B (timhua/pythia1b_deduped_saes covers only L11).\n")
    f.write("L11 of 16 is ~69% through model — comparable depth to 70m L4 and 410m L17.\n\n")
    f.write(f"**Within-pair cos(UP, DOWN) at Pythia 1B layer 11:**\n")
    f.write(f"- mean   = {mean_cos:+.4f}\n")
    f.write(f"- median = {median_cos:+.4f}\n")
    f.write(f"- range  = [{min_cos:+.4f}, {max_cos:+.4f}]\n\n")
    f.write("## Cross-scale comparison\n\n")
    f.write(f"  70m  L4 (67% through): +0.4915  (exp19)\n")
    f.write(f"  410m L17-18 (71-75%):  +0.5358 to +0.5554  (exp19)\n")
    f.write(f"  1B   L11 (69% through): {mean_cos:+.4f}  (this experiment)\n\n")
    f.write(f"  70m  all 6 layers:  +0.5546 (exp19)\n")
    f.write(f"  410m all 24 layers: +0.5409 (exp19)\n")
    f.write(f"  1B   layer 11 only: {mean_cos:+.4f} (this experiment)\n\n")
    f.write("## Per-triple cosines\n\n")
    f.write(f"  {'domain':>12} {'pair':>4} {'cos(UP, DOWN)':>15}\n")
    for d, i, c in per_triple:
        f.write(f"  {d:>12} {i:>4} {c:>+15.4f}\n")

print(f"\nReport: {report_path}")
torch.save({"mean_cos": mean_cos, "per_triple": per_triple},
           "/Users/macn/Documents/embeddingexp/exp20_results.pt")
