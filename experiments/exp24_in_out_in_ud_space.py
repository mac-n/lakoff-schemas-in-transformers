"""
exp24_in_out_in_ud_space.py - are IN and OUT actually opposite directions
within UD-space?

We previously tested cos(A_inout, A_updown) = −0.14 — the IO axis points
slightly toward DOWN. But that's the DIFFERENCE between IN-projection and
OUT-projection onto UD. We never computed the projections individually.

This experiment: at each layer, build A_updown axis. Then for each IN-offset
and each OUT-offset (from our matched IN/OUT triples), compute its scalar
projection onto A_updown. Report:

  mean(IN_offset · A_updown_normalized)  — how UP-ish is IN on average?
  mean(OUT_offset · A_updown_normalized) — how UP-ish is OUT on average?

Possible outcomes:
  - IN > 0 and OUT < 0 (opposite signs): IN and OUT ARE polar opposites in UD-space
  - IN > 0 and OUT > 0 (both positive): both more UP-like than baseline, no polarity
  - IN < 0 and OUT < 0 (both negative): both more DOWN-like than baseline
  - Both ≈ 0: orthogonal, no individual UD-content
"""

from collections import defaultdict

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

# Same UP-DOWN and IN-OUT triples as exp21
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

ud_triples = [(d, b, u, dn) for d, ts in UP_DOWN.items() for b, u, dn in ts]
io_triples = [(d, b, i, o) for d, ts in IN_OUT.items() for b, i, o in ts]


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

    # Compute UD axis from polar offsets
    ud_polar = []
    for _, b, u, dn in ud_triples:
        b_v = enc(b)
        u_v = enc(u)
        d_v = enc(dn)
        ud_polar.append((u_v - b_v) - (d_v - b_v))
    A_updown = np.mean(np.stack(ud_polar), axis=0)
    A_updown_normalized = A_updown / np.linalg.norm(A_updown)

    # Compute IN-offsets and OUT-offsets individually
    in_projs = []
    out_projs = []
    for _, b, i, o in io_triples:
        b_v = enc(b)
        i_v = enc(i)
        o_v = enc(o)
        in_off = i_v - b_v
        out_off = o_v - b_v
        in_projs.append(float(np.dot(in_off, A_updown_normalized)))
        out_projs.append(float(np.dot(out_off, A_updown_normalized)))

    # Also compute the UP and DOWN sentence-offsets projected onto A_updown for reference
    up_self_projs = []
    down_self_projs = []
    for _, b, u, dn in ud_triples:
        b_v = enc(b)
        u_v = enc(u)
        d_v = enc(dn)
        up_self_projs.append(float(np.dot(u_v - b_v, A_updown_normalized)))
        down_self_projs.append(float(np.dot(d_v - b_v, A_updown_normalized)))

    results[layer] = {
        "mean_in_proj": float(np.mean(in_projs)),
        "mean_out_proj": float(np.mean(out_projs)),
        "median_in_proj": float(np.median(in_projs)),
        "median_out_proj": float(np.median(out_projs)),
        "mean_up_self_proj": float(np.mean(up_self_projs)),
        "mean_down_self_proj": float(np.mean(down_self_projs)),
    }
    r = results[layer]
    print(f"  Reference (sanity check that A_updown points UP):")
    print(f"    mean(UP_offset · A_updown):   {r['mean_up_self_proj']:+.4f}  (should be POSITIVE = UP-aligned)")
    print(f"    mean(DOWN_offset · A_updown): {r['mean_down_self_proj']:+.4f}  (should be NEGATIVE = DOWN-aligned)")
    print(f"  Test:")
    print(f"    mean(IN_offset · A_updown):   {r['mean_in_proj']:+.4f}")
    print(f"    mean(OUT_offset · A_updown):  {r['mean_out_proj']:+.4f}")

    sign_match = (r['mean_in_proj'] > 0) != (r['mean_out_proj'] > 0)
    if sign_match:
        in_dir = "UP" if r['mean_in_proj'] > 0 else "DOWN"
        out_dir = "UP" if r['mean_out_proj'] > 0 else "DOWN"
        print(f"  → IN projects {in_dir}, OUT projects {out_dir}. They ARE polar opposites in UD-space.")
    else:
        same_dir = "UP" if r['mean_in_proj'] > 0 else "DOWN"
        print(f"  → IN and OUT both project toward {same_dir}. NOT polar opposites in UD-space.")

    del sae

del model

report_path = "/Users/macn/Documents/embeddingexp/results_exp24_in_out_in_ud_space.md"
with open(report_path, "w") as f:
    def out(s=""):
        print(s)
        f.write(s + "\n")

    out("# exp24 — are IN and OUT opposite directions WITHIN UD-space?")
    out()
    out("Test: project each IN_offset and OUT_offset (from matched IO triples) onto")
    out("the constructed A_updown axis. Compare mean projections.")
    out()
    out("## Per-layer projections (Pythia 70m residual-stream SAE)")
    out()
    out(f"  {'layer':>5}  {'UP_proj':>9}  {'DOWN_proj':>10}  {'IN_proj':>9}  {'OUT_proj':>10}  {'verdict':>40}")
    for layer in sorted(results.keys()):
        r = results[layer]
        sign_match = (r['mean_in_proj'] > 0) != (r['mean_out_proj'] > 0)
        if sign_match:
            in_dir = "UP" if r['mean_in_proj'] > 0 else "DOWN"
            out_dir = "UP" if r['mean_out_proj'] > 0 else "DOWN"
            verdict = f"IN→{in_dir}, OUT→{out_dir} (polar)"
        else:
            same_dir = "UP" if r['mean_in_proj'] > 0 else "DOWN"
            verdict = f"both→{same_dir} (NOT polar)"
        out(f"  {layer:>5d}  {r['mean_up_self_proj']:>+9.4f}  {r['mean_down_self_proj']:>+10.4f}  "
            f"{r['mean_in_proj']:>+9.4f}  {r['mean_out_proj']:>+10.4f}  {verdict:>40}")
    out()
    out("## Interpretation")
    out()
    out("- If IN > 0 and OUT < 0 across layers: IN ↔ UP, OUT ↔ DOWN. The model has a")
    out("  shared polarity convention between containment and verticality.")
    out("- If both > 0 or both < 0: IN and OUT both lean the same way on UD-axis.")
    out("  No individual polarity, the IO axis is rotated but not polar in UD-space.")
    out("- If both ≈ 0: IO and UD are genuinely orthogonal at all levels.")

print(f"\nReport: {report_path}")
torch.save({"results": results}, "/Users/macn/Documents/embeddingexp/exp24_results.pt")
