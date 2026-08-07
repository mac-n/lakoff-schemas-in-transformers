"""
exp25_left_right.py - is LEFT-RIGHT orthogonal in UD-space, and does it have
the negativity-salience asymmetry that UP-DOWN has?

Niamh's question after the negativity-salience reframe: LEFT and RIGHT are
another spatial axis but neither carries strong negativity-attention the way
DOWN does. Under the salience hypothesis:

  - cos(A_leftright, A_updown) ≈ 0 — different spatial axis
  - within-pair cos(LEFT_offset, RIGHT_offset) > +0.55 (UP/DOWN's value) —
    neither pole has strong salience-feature to pull them apart
  - mean(LEFT_proj onto A_updown) and mean(RIGHT_proj onto A_updown) both
    small and roughly balanced — no negativity-attractor pulling either

If all three predictions hold, negativity-salience explains a lot of our
project. If LEFT-RIGHT also projects strongly negative like everything else,
the "any state change" interpretation wins instead.

Note: LEFT-RIGHT genuinely doesn't have the rich Lakoffian metaphorical
extensions of UP-DOWN (which IS itself Lakoff-consistent — symmetric body
schemas don't generate strong cross-domain metaphors because there's no
embodied asymmetry to ground them). So we use mostly literal-spatial triples.
"""

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

# UP-DOWN triples (for A_updown construction)
UP_DOWN_FLAT = [
    ("The temperature was constant throughout the day.", "The temperature rose throughout the day.", "The temperature plunged throughout the day."),
    ("The thermometer reading held steady overnight.", "The thermometer reading climbed overnight.", "The thermometer reading plummeted overnight."),
    ("Room temperature stayed the same all afternoon.", "Room temperature soared all afternoon.", "Room temperature sank all afternoon."),
    ("The water temperature was even before boiling.", "The water temperature ascended past boiling.", "The water temperature descended past freezing."),
    ("The forecast showed stable temperatures this week.", "The forecast showed increasing temperatures this week.", "The forecast showed tumbling temperatures this week."),
    ("Her mood was neutral after the meeting.", "Her mood was elated after the meeting.", "Her mood was dejected after the meeting."),
    ("He felt nothing about the news.", "He felt jubilant about the news.", "He felt morose about the news."),
    ("Their spirits were average that morning.", "Their spirits were radiant that morning.", "Their spirits were forlorn that morning."),
    ("She seemed unchanged by the praise.", "She seemed ecstatic from the praise.", "She seemed glum despite the praise."),
    ("The audience reacted plainly to the song.", "The audience reacted joyfully to the song.", "The audience reacted somberly to the song."),
    ("The company's revenue was stable last quarter.", "The company's revenue grew last quarter.", "The company's revenue declined last quarter."),
    ("Sales held steady through summer.", "Sales increased through summer.", "Sales decreased through summer."),
    ("The population stayed flat over the decade.", "The population multiplied over the decade.", "The population dwindled over the decade."),
    ("Inventory remained constant this month.", "Inventory expanded this month.", "Inventory shrank this month."),
    ("Subscribers stayed the same all year.", "Subscribers accrued all year.", "Subscribers waned all year."),
    ("Her position was unchanged at the firm.", "Her position was promoted at the firm.", "Her position was demoted at the firm."),
    ("His reputation remained the same in the field.", "His reputation became prominent in the field.", "His reputation was disgraced in the field."),
    ("She held the same rank for years.", "She held a distinguished rank for years.", "She was ousted from her rank."),
    ("His standing was ordinary among peers.", "His standing was esteemed among peers.", "His standing was discredited among peers."),
    ("The professor's recognition was middling.", "The professor's recognition was prestigious.", "The professor's recognition was dethroned."),
    ("His condition was stable yesterday.", "His condition was thriving yesterday.", "His condition was ailing yesterday."),
    ("Her vitality stayed even after the surgery.", "Her vitality returned vigorously after the surgery.", "Her vitality deteriorated after the surgery."),
    ("The patient remained unchanged.", "The patient was recuperating.", "The patient was languishing."),
    ("The plants looked the same in the garden.", "The plants looked robust in the garden.", "The plants looked sickly in the garden."),
    ("His energy was average that week.", "His energy was vital that week.", "His energy was feeble that week."),
]

# LEFT-RIGHT triples (literal spatial, body, orientation, driving — minimal metaphor)
LEFT_RIGHT = [
    # Spatial motion
    ("She stood in the middle of the room.", "She walked to the left side of the room.", "She walked to the right side of the room."),
    ("The car was parked in the center.", "The car turned left at the intersection.", "The car turned right at the intersection."),
    ("The arrow points straight ahead.", "The arrow points to the left.", "The arrow points to the right."),
    ("The dancer stood centered.", "The dancer stepped to the left.", "The dancer stepped to the right."),
    ("They walked down the middle of the road.", "They walked down the left side of the road.", "They walked down the right side of the road."),
    # Body / handedness
    ("She used both hands equally.", "She used her left hand for everything.", "She used her right hand for everything."),
    ("He was ambidextrous from childhood.", "He was left-handed from childhood.", "He was right-handed from childhood."),
    ("The player swung with either side.", "The player swung from the left side.", "The player swung from the right side."),
    ("She wore a watch on either wrist.", "She wore a watch on her left wrist.", "She wore a watch on her right wrist."),
    ("The artist held the brush in either hand.", "The artist held the brush in the left hand.", "The artist held the brush in the right hand."),
    # Driving / lane
    ("She drove down the road carefully.", "She drove on the left side of the road.", "She drove on the right side of the road."),
    ("The driver stayed in the lane.", "The driver moved to the left lane.", "The driver moved to the right lane."),
    ("The car drifted on the highway.", "The car drifted to the left lane.", "The car drifted to the right lane."),
    # Orientation
    ("The compass needle pointed straight.", "The compass needle pointed left.", "The compass needle pointed right."),
    ("The path forked ahead.", "The path forked to the left.", "The path forked to the right."),
]

print(f"UP-DOWN triples: {len(UP_DOWN_FLAT)}")
print(f"LEFT-RIGHT triples: {len(LEFT_RIGHT)}")


def cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")
print("Loading Pythia 70m-deduped...")
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
model.eval()
n_layers = model.cfg.n_layers


def collect(text, hook):
    tokens = model.to_tokens(text)
    if tokens.shape[1] > 64:
        tokens = tokens[:, :64]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook)
    return cache[hook][0].cpu().float()


results = {}
for layer in range(n_layers):
    print(f"\n--- Layer {layer} ---")
    hook = f"blocks.{layer}.hook_resid_post"
    sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=hook, device="cpu")
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res

    def enc(text):
        with torch.no_grad():
            return sae.encode(collect(text, hook)).max(0).values.numpy().astype(np.float64)

    # Build UD axis
    ud_polar = []
    for b, u, d in UP_DOWN_FLAT:
        b_v = enc(b)
        u_v = enc(u)
        d_v = enc(d)
        ud_polar.append((u_v - b_v) - (d_v - b_v))
    A_updown = np.mean(np.stack(ud_polar), axis=0)
    A_updown_norm = A_updown / np.linalg.norm(A_updown)

    # Build LR axis + collect per-pair offsets
    lr_polar = []
    left_offsets = []
    right_offsets = []
    within_pair_cos = []
    for b, l, r in LEFT_RIGHT:
        b_v = enc(b)
        l_v = enc(l)
        r_v = enc(r)
        l_off = l_v - b_v
        r_off = r_v - b_v
        left_offsets.append(l_off)
        right_offsets.append(r_off)
        within_pair_cos.append(cos(l_off, r_off))
        lr_polar.append(l_off - r_off)
    A_leftright = np.mean(np.stack(lr_polar), axis=0)

    # Test 1: orthogonality of axes
    cos_lr_ud = cos(A_leftright, A_updown)

    # Test 2: within-pair LEFT-RIGHT polarity
    mean_within = float(np.mean(within_pair_cos))
    median_within = float(np.median(within_pair_cos))

    # Test 3: individual projections onto A_updown
    left_projs = [float(np.dot(o, A_updown_norm)) for o in left_offsets]
    right_projs = [float(np.dot(o, A_updown_norm)) for o in right_offsets]
    mean_left_proj = float(np.mean(left_projs))
    mean_right_proj = float(np.mean(right_projs))

    results[layer] = {
        "cos_LR_UD": cos_lr_ud,
        "within_pair_LR_cos_mean": mean_within,
        "within_pair_LR_cos_median": median_within,
        "mean_left_proj_onto_UD": mean_left_proj,
        "mean_right_proj_onto_UD": mean_right_proj,
    }
    r = results[layer]
    print(f"  cos(A_leftright, A_updown): {cos_lr_ud:+.4f}  (orthogonality test)")
    print(f"  within-pair cos(L, R):       mean={mean_within:+.4f}  median={median_within:+.4f}")
    print(f"    (compare to UP/DOWN's +0.55 — if higher, LR is more aligned within-pair → less polar)")
    print(f"  mean LEFT projection onto A_updown:  {mean_left_proj:+.4f}")
    print(f"  mean RIGHT projection onto A_updown: {mean_right_proj:+.4f}")
    print(f"    (compare to IN={-1.85:+.2f}, OUT={-1.49:+.2f} at L3 — if LR closer to 0, salience hypothesis supported)")

    del sae

del model

report_path = "/Users/macn/Documents/embeddingexp/results_exp25_left_right.md"
with open(report_path, "w") as f:
    def out(s=""):
        print(s)
        f.write(s + "\n")

    out("# exp25 — LEFT-RIGHT axis: orthogonality and salience asymmetry")
    out()
    out("Three predictions under the negativity-salience hypothesis:")
    out("1. cos(A_leftright, A_updown) ≈ 0 — different spatial axis")
    out("2. within-pair cos(L_offset, R_offset) > +0.55 (UP/DOWN's) — neither pole has salience-attractor")
    out("3. mean LEFT_proj and RIGHT_proj onto A_updown small and balanced — no DOWN-pull")
    out()
    out(f"Triples: UP-DOWN={len(UP_DOWN_FLAT)} (for A_updown), LEFT-RIGHT={len(LEFT_RIGHT)} (test)")
    out()
    out("## Per-layer results")
    out()
    out(f"  {'layer':>5}  {'cos(LR,UD)':>11}  {'within-pair L,R':>17}  {'mean L→UD':>11}  {'mean R→UD':>11}")
    for layer in sorted(results.keys()):
        r = results[layer]
        out(f"  {layer:>5d}  {r['cos_LR_UD']:>+11.4f}  {r['within_pair_LR_cos_mean']:>+17.4f}  "
            f"{r['mean_left_proj_onto_UD']:>+11.4f}  {r['mean_right_proj_onto_UD']:>+11.4f}")
    out()
    out("## Reference (from prior experiments at L3)")
    out()
    out("  UP-DOWN within-pair: +0.55 (exp19)")
    out("  IN mean projection onto UD: -1.85 (exp24)")
    out("  OUT mean projection onto UD: -1.49 (exp24)")
    out("  UP mean projection onto UD (self): +0.18 at L3")
    out("  DOWN mean projection onto UD (self): -2.08 at L3")
    out()
    out("## Interpretation guide")
    out()
    out("- If within-pair L-R cos > 0.7 AND L/R projections both small (|< 0.5|): negativity-salience hypothesis SUPPORTED")
    out("  → LR is a true symmetric spatial axis without the salience asymmetry UD has")
    out("- If within-pair L-R cos ≈ 0.55 (like UD): symmetric polar structure, salience hypothesis weaker")
    out("- If L and R both project strongly negative (like IN/OUT did): 'any state change' interpretation wins, salience hypothesis weaker")

print(f"\nReport: {report_path}")
torch.save({"results": results}, "/Users/macn/Documents/embeddingexp/exp25_results.pt")
