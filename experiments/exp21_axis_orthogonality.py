"""
exp21_axis_orthogonality.py - are schema polar axes orthogonal?

Niamh's reframe: instead of asking "are UP and DOWN naturally opposite in
the SAE basis?" (which gave us +0.55), CONSTRUCT polar axes by fiat and
ask whether different schemas' constructed axes are orthogonal or aligned.

  A_updown = mean((UP_offset_i) - (DOWN_offset_i)) across UP/DOWN pairs
  A_inout  = mean((IN_offset_i)  - (OUT_offset_i))  across IN/OUT pairs
  A_bev    = mean((A_offset_i)   - (B_offset_i))    across BEVERAGE sham pairs
            (coffee vs tea — no polar embodied meaning, should be orthogonal
             to any real schema axis)

Then compute pairwise cosines:
  cos(A_updown, A_inout):
    ≈ +1: schemas collapse to ONE generic polarity ruler (valence-like)
    ≈  0: schemas are SEPARABLE polar dimensions (Lakoffian-distinct)
    ≈ -1: weirdly opposite (no obvious reason)
  cos(A_updown, A_bev):
    Should ≈ 0 — coffee-vs-tea isn't a polar embodied axis
  cos(A_inout, A_bev):
    Should ≈ 0

If we find UP-DOWN and IN-OUT are orthogonal while BEVERAGE is also
orthogonal to both, schemas exist as separable polar dimensions (good for
Lakoff). If UP-DOWN and IN-OUT are highly aligned, there's one collapsed
polarity ruler that all schemas project onto (interesting different finding).
"""

import gc
from collections import defaultdict

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

# ---- UP-DOWN triples (same as exp15) ----
UP_DOWN = {
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

# ---- IN-OUT triples (new, constructed to mirror UP-DOWN structure) ----
# Each triple: (baseline, IN-transformed, OUT-transformed)
IN_OUT = {
    "literal": [
        ("The papers stayed on the table.",
         "The papers were contained in the folder.",
         "The papers were extracted from the folder."),
        ("The puppy sat quietly in the room.",
         "The puppy was enclosed in its crate.",
         "The puppy was released from its crate."),
        ("The wine sat on the counter.",
         "The wine was sealed in the bottle.",
         "The wine was extracted from the bottle."),
        ("The marbles lay scattered everywhere.",
         "The marbles were encased in the jar.",
         "The marbles were ejected from the jar."),
        ("The documents stayed on the desk.",
         "The documents were wrapped in the envelope.",
         "The documents were released from the envelope."),
    ],
    "mind": [
        ("She knew nothing about the question.",
         "She pondered the question for hours.",
         "She forgot the question entirely."),
        ("He felt indifferent to the memory.",
         "He harbored the memory for years.",
         "He banished the memory completely."),
        ("The idea was unfamiliar to her.",
         "The idea was contemplated by her.",
         "The idea was discarded by her."),
        ("The thought remained vague all day.",
         "The thought was ruminated upon for weeks.",
         "The thought was dismissed in seconds."),
        ("The plan was unconsidered for now.",
         "The plan was meditated upon at length.",
         "The plan was purged from memory."),
    ],
    "relationship": [
        ("Their status was ambiguous last year.",
         "They were married last year.",
         "They were divorced last year."),
        ("She wasn't sure about him.",
         "She was engaged to him.",
         "She was estranged from him."),
        ("He had no particular feelings then.",
         "He was committed to her formally.",
         "He was separated from her formally."),
        ("The couple lived ordinarily for years.",
         "The couple was partnered together for years.",
         "The couple was single and apart for years."),
        ("Their bond was unclear over time.",
         "Their bond was a deep marriage over time.",
         "Their bond was an ugly separation over time."),
    ],
    "time": [
        ("She paid no attention to timing.",
         "She acted within the hour.",
         "She acted after the deadline had expired."),
        ("The conversation happened at some point.",
         "The conversation happened during the meeting.",
         "The conversation happened after the meeting elapsed."),
        ("He had time vaguely available.",
         "He had time throughout the day.",
         "He had time only after the day had lapsed."),
        ("The opportunity was timing-agnostic.",
         "The opportunity existed amid the negotiations.",
         "The opportunity was bygone by negotiation's end."),
        ("The decision was timeless in scope.",
         "The decision spanned the entire year.",
         "The decision was outdated by the next year."),
    ],
    "difficulty": [
        ("The traveler walked along the path.",
         "The traveler was stranded in the wilderness.",
         "The traveler was rescued from the wilderness."),
        ("She faced her usual workload.",
         "She was mired in endless tasks.",
         "She was extricated from endless tasks."),
        ("The team handled normal challenges.",
         "The team was ensnared by hidden problems.",
         "The team was liberated from hidden problems."),
        ("He worked at the usual pace.",
         "He was embroiled in the project's chaos.",
         "He was relieved from the project's chaos."),
        ("The patient maintained their condition.",
         "The patient was beleaguered by symptoms.",
         "The patient was salvaged from severe symptoms."),
    ],
}

# ---- BEVERAGE sham: coffee-vs-tea (no polar embodied meaning) ----
BEVERAGE_SHAM = [
    ("She ordered a beverage at the cafe.",
     "She ordered coffee at the cafe.",
     "She ordered tea at the cafe."),
    ("He drank a beverage in the morning.",
     "He drank espresso in the morning.",
     "He drank chai in the morning."),
    ("They served drinks at the meeting.",
     "They served lattes at the meeting.",
     "They served matcha at the meeting."),
    ("She prefers a drink after dinner.",
     "She prefers cappuccino after dinner.",
     "She prefers oolong after dinner."),
    ("The cafe offered various drinks.",
     "The cafe offered mocha.",
     "The cafe offered kombucha."),
]

# Flatten
ud_triples = []
for d, ts in UP_DOWN.items():
    for b, u, dn in ts:
        ud_triples.append((d, b, u, dn))
io_triples = []
for d, ts in IN_OUT.items():
    for b, i, o in ts:
        io_triples.append((d, b, i, o))
bev_triples = [(None, b, a, c) for b, a, c in BEVERAGE_SHAM]

print(f"UP/DOWN triples: {len(ud_triples)}")
print(f"IN/OUT triples: {len(io_triples)}")
print(f"BEVERAGE sham triples: {len(bev_triples)}")


def cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---- Device + model ----
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")
print("\nLoading Pythia 70m-deduped...")
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


# ---- For each layer: compute constructed axes and their pairwise cosines ----
results = {}
for layer in range(n_layers):
    print(f"\n--- Layer {layer} ---")
    hook = f"blocks.{layer}.hook_resid_post"
    sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=hook, device="cpu")
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res

    def enc(text):
        with torch.no_grad():
            return sae.encode(collect(text, hook)).max(0).values.numpy().astype(np.float64)

    # Compute UP-DOWN polar offsets per pair
    polar_ud = []
    for _, b, u, dn in ud_triples:
        b_v = enc(b)
        u_v = enc(u)
        d_v = enc(dn)
        polar_ud.append((u_v - b_v) - (d_v - b_v))  # = u_v - d_v

    # Compute IN-OUT polar offsets per pair
    polar_io = []
    for _, b, i, o in io_triples:
        b_v = enc(b)
        i_v = enc(i)
        o_v = enc(o)
        polar_io.append((i_v - b_v) - (o_v - b_v))  # = i_v - o_v

    # Compute BEVERAGE sham polar offsets per pair
    polar_bev = []
    for _, b, a, c in bev_triples:
        b_v = enc(b)
        a_v = enc(a)
        c_v = enc(c)
        polar_bev.append((a_v - b_v) - (c_v - b_v))  # = a_v - c_v

    # Mean polar axes per schema
    A_ud = np.mean(np.stack(polar_ud), axis=0)
    A_io = np.mean(np.stack(polar_io), axis=0)
    A_bev = np.mean(np.stack(polar_bev), axis=0)

    cos_ud_io = cos(A_ud, A_io)
    cos_ud_bev = cos(A_ud, A_bev)
    cos_io_bev = cos(A_io, A_bev)

    # Also: median of pairwise cosines (less sensitive to one big domain dominating mean)
    polar_ud_stack = np.stack(polar_ud)
    polar_io_stack = np.stack(polar_io)

    # Cross-cosines between every UP-DOWN pair and every IN-OUT pair
    pair_cosines_ud_io = []
    for u in polar_ud:
        for i in polar_io:
            pair_cosines_ud_io.append(cos(u, i))

    results[layer] = {
        "cos_A_updown_A_inout": cos_ud_io,
        "cos_A_updown_A_beverage": cos_ud_bev,
        "cos_A_inout_A_beverage": cos_io_bev,
        "mean_pairwise_ud_io": float(np.mean(pair_cosines_ud_io)),
        "median_pairwise_ud_io": float(np.median(pair_cosines_ud_io)),
    }

    print(f"  cos(A_updown, A_inout):    {cos_ud_io:+.4f}")
    print(f"  cos(A_updown, A_beverage): {cos_ud_bev:+.4f}  (sham, should be ≈ 0)")
    print(f"  cos(A_inout,  A_beverage): {cos_io_bev:+.4f}  (sham, should be ≈ 0)")
    print(f"  mean pairwise cos(UD pair, IO pair):   {results[layer]['mean_pairwise_ud_io']:+.4f}")
    print(f"  median pairwise cos(UD pair, IO pair): {results[layer]['median_pairwise_ud_io']:+.4f}")

    del sae
    gc.collect()

del model

# ---- Report ----
report_path = "/Users/macn/Documents/embeddingexp/results_exp21_axis_orthogonality.md"
with open(report_path, "w") as f:
    def out(s=""):
        print(s)
        f.write(s + "\n")

    out("# exp21 — are schema polar axes orthogonal?")
    out()
    out("Constructed polar axes by fiat (mean of within-pair UP-DOWN offsets across pairs)")
    out("and checked whether different schemas' axes align or are orthogonal.")
    out()
    out("**Predictions:**")
    out("- cos(A_updown, A_inout) ≈ +1: schemas collapse to one generic polarity ruler")
    out("- cos(A_updown, A_inout) ≈  0: schemas are separable polar dimensions (Lakoff-consistent)")
    out("- cos(A_x, A_beverage) ≈ 0: sham (coffee vs tea) shouldn't align with real polar axes")
    out()
    out("## Summary across layers (Pythia 70m)")
    out()
    out(f"  {'layer':>5}  {'cos(UD,IO)':>12}  {'cos(UD,BEV)':>13}  {'cos(IO,BEV)':>13}  {'mean cos(UDpair,IOpair)':>25}")
    for layer in sorted(results.keys()):
        r = results[layer]
        out(f"  {layer:>5d}  {r['cos_A_updown_A_inout']:>+12.4f}  {r['cos_A_updown_A_beverage']:>+13.4f}  "
            f"{r['cos_A_inout_A_beverage']:>+13.4f}  {r['mean_pairwise_ud_io']:>+25.4f}")
    out()
    out("## Interpretation")
    out()
    out("Look at the cos(UD, IO) column. If consistently:")
    out("- > 0.7: schemas have collapsed to one polarity. The model treats UP↔DOWN and IN↔OUT as the same dimension.")
    out("- 0.2-0.6: partially aligned. Some shared valence component, some distinctness.")
    out("- < 0.2: schemas are separable polar dimensions. UP-DOWN and IN-OUT live on different axes.")
    out("- negative: even more separated — they have inverted polar structure.")
    out()
    out("The sham columns should hover near 0 if BEVERAGE is doing its job as a non-polar control.")

print(f"\nReport: {report_path}")
torch.save({"results": results}, "/Users/macn/Documents/embeddingexp/exp21_results.pt")
