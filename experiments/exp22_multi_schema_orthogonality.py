"""
exp22_multi_schema_orthogonality.py - are MANY schema polar axes orthogonal?

Builds on exp21's finding (cos(A_updown, A_inout) ≈ −0.1, decisively below
the "schemas collapse to one ruler" threshold). If schemas exist as multiple
separable embodied dimensions, FORWARD-BACK and LIGHT-DARK should also be
approximately orthogonal to UP-DOWN and IN-OUT and to each other.

Schemas tested:
  - UP-DOWN  (25 pairs, from exp15)
  - IN-OUT   (25 pairs, from exp21)
  - FORWARD-BACK (25 pairs, new — literal motion, progress, time, development, journey)
  - LIGHT-DARK (25 pairs, new — illumination, clarity, hope, goodness, knowledge)
  - TUESDAY-WEDNESDAY sham (5 pairs — arbitrary weekday names with no embodied content)

For each schema, constructed polar axis = mean within-pair offset.
Output: 5x5 pairwise cosine matrix at each layer.

Predictions:
  - 4 schema axes mutually orthogonal: multiple separable embodied dimensions (strong Lakoff structural support)
  - Some pairs align (e.g. cos(UD, FB) high): schemas collapse partially toward shared valence axis
  - Sham axis orthogonal to all schemas: methodology working
  - Sham axis aligned with any schema: sham still has hidden content (per the exp21 beverage lesson)
"""

import gc
from collections import defaultdict

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

# ---- UP-DOWN ----
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

# ---- IN-OUT (from exp21) ----
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

# ---- FORWARD-BACK (new) ----
# Vocabulary deliberately decorrelated from UP/DOWN (no fall/plunge/decline/etc.)
FORWARD_BACK = {
    "literal_motion": [
        ("She stood at the door.",
         "She walked forward into the room.",
         "She walked backward into the hallway."),
        ("The car was parked.",
         "The car drove forward down the street.",
         "The car drove backward into the garage."),
        ("He stood in the field.",
         "He moved forward across the field.",
         "He moved backward toward the gate."),
        ("They sat in the conference room.",
         "They leaned forward toward the speaker.",
         "They leaned back away from the speaker."),
        ("The horse stood still.",
         "The horse advanced toward the fence.",
         "The horse retreated from the fence."),
    ],
    "progress": [
        ("The project was in its current state.",
         "The project advanced significantly.",
         "The project regressed significantly."),
        ("His skills were the same.",
         "His skills progressed steadily.",
         "His skills regressed steadily."),
        ("The negotiations were ongoing.",
         "The negotiations advanced toward agreement.",
         "The negotiations stalled into deadlock."),
        ("Her training was constant.",
         "Her training advanced rapidly.",
         "Her training regressed rapidly."),
        ("The team's performance was stable.",
         "The team's performance progressed all season.",
         "The team's performance regressed all season."),
    ],
    "time": [
        ("She thought about her life generically.",
         "She thought about her future ahead.",
         "She thought about her past behind."),
        ("The conversation drifted.",
         "The conversation moved forward to future plans.",
         "The conversation moved back to old memories."),
        ("He considered his options.",
         "He looked ahead to next year.",
         "He looked back at last year."),
        ("The committee discussed business.",
         "The committee planned ahead for next quarter.",
         "The committee reviewed prior quarters."),
        ("She read the article.",
         "She read forward through future predictions.",
         "She read backward through historical records."),
    ],
    "development": [
        ("The technology stayed the same.",
         "The technology advanced substantially.",
         "The technology regressed substantially."),
        ("Her thinking was unchanged.",
         "Her thinking evolved forward.",
         "Her thinking reverted backward."),
        ("The economy was steady.",
         "The economy progressed forward.",
         "The economy retreated backward."),
        ("His career was stable.",
         "His career advanced over the years.",
         "His career regressed over the years."),
        ("The community remained the same.",
         "The community moved forward together.",
         "The community moved backward together."),
    ],
    "journey": [
        ("The traveler stopped.",
         "The traveler continued onward.",
         "The traveler turned back homeward."),
        ("The hiker rested.",
         "The hiker pressed onward.",
         "The hiker turned back to the camp."),
        ("The ship sailed.",
         "The ship continued forward to port.",
         "The ship reversed course backward."),
        ("The explorers paused.",
         "The explorers advanced into the wilderness.",
         "The explorers retreated from the wilderness."),
        ("She drove the route.",
         "She continued forward to the destination.",
         "She turned back toward home."),
    ],
}

# ---- LIGHT-DARK (new) ----
LIGHT_DARK = {
    "literal_illumination": [
        ("The room was at neutral light.",
         "The room was brightly illuminated.",
         "The room was dimly shadowed."),
        ("The candle stood ready.",
         "The candle glowed warmly.",
         "The candle was extinguished into darkness."),
        ("The sky had some clouds.",
         "The sky was radiantly clear.",
         "The sky was murkily overcast."),
        ("The garden was at twilight.",
         "The garden was bright with sunshine.",
         "The garden was dark with shadow."),
        ("The lamp stood on the table.",
         "The lamp shone brightly.",
         "The lamp was dimmed completely."),
    ],
    "clarity": [
        ("The situation was undetermined.",
         "The situation was illuminated by analysis.",
         "The situation was shrouded in confusion."),
        ("The question remained.",
         "The question became luminously clear.",
         "The question became hopelessly obscure."),
        ("Her perspective was indifferent.",
         "Her perspective was radiantly informed.",
         "Her perspective was darkly clouded."),
        ("The investigation continued.",
         "The investigation shed light on the truth.",
         "The investigation deepened the mystery."),
        ("The lecture was happening.",
         "The lecture clarified the concept brilliantly.",
         "The lecture obscured the concept entirely."),
    ],
    "hope": [
        ("Her outlook was steady.",
         "Her outlook was bright with hope.",
         "Her outlook was dark with despair."),
        ("The future was uncertain.",
         "The future shone with promise.",
         "The future loomed darkly grim."),
        ("He felt the day calmly.",
         "He felt the day glowing with optimism.",
         "He felt the day shadowed by dread."),
        ("The community waited.",
         "The community was bright with anticipation.",
         "The community was grim with foreboding."),
        ("Their plans were drafted.",
         "Their plans glowed with possibility.",
         "Their plans were shadowed by setbacks."),
    ],
    "goodness": [
        ("His character was ordinary.",
         "His character was bright and pure.",
         "His character was tainted and evil."),
        ("The deed was unremarkable.",
         "The deed was radiantly noble.",
         "The deed was darkly cruel."),
        ("The story was neutral.",
         "The story celebrated luminous goodness.",
         "The story portrayed shadowy wickedness."),
        ("Their motives were unknown.",
         "Their motives shone with virtue.",
         "Their motives were shadowed by greed."),
        ("The ruler was indifferent.",
         "The ruler was illuminated by justice.",
         "The ruler was darkened by tyranny."),
    ],
    "knowledge": [
        ("The topic was unfamiliar.",
         "The topic was illuminated by study.",
         "The topic remained shrouded in mystery."),
        ("The truth was hidden.",
         "The truth came to brilliant light.",
         "The truth was buried in darkness."),
        ("The mystery existed.",
         "The mystery was solved brilliantly.",
         "The mystery deepened obscurely."),
        ("She approached the subject.",
         "She approached the subject with enlightenment.",
         "She approached the subject with darkness."),
        ("The data was uninterpreted.",
         "The data was clarified luminously.",
         "The data was obscured by complexity."),
    ],
}

# ---- TUESDAY-WEDNESDAY sham ----
TUE_WED_SHAM = [
    ("The meeting is scheduled this week.",
     "The meeting is on Tuesday this week.",
     "The meeting is on Wednesday this week."),
    ("The deadline is upcoming.",
     "The deadline is Tuesday at noon.",
     "The deadline is Wednesday at noon."),
    ("She has an appointment.",
     "She has an appointment Tuesday morning.",
     "She has an appointment Wednesday morning."),
    ("He plans to visit soon.",
     "He plans to visit Tuesday afternoon.",
     "He plans to visit Wednesday afternoon."),
    ("Class meets weekly.",
     "Class meets every Tuesday.",
     "Class meets every Wednesday."),
]

# Flatten
def flatten(d_or_l):
    out = []
    if isinstance(d_or_l, dict):
        for d, ts in d_or_l.items():
            for b, a, c in ts:
                out.append((d, b, a, c))
    else:
        for b, a, c in d_or_l:
            out.append((None, b, a, c))
    return out

SCHEMAS = {
    "UP-DOWN": flatten(UP_DOWN),
    "IN-OUT": flatten(IN_OUT),
    "FORWARD-BACK": flatten(FORWARD_BACK),
    "LIGHT-DARK": flatten(LIGHT_DARK),
    "TUE-WED_sham": flatten(TUE_WED_SHAM),
}

for name, triples in SCHEMAS.items():
    print(f"{name}: {len(triples)} triples")


def cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---- Device + model ----
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\ndevice: {device}")
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


# ---- Per layer: compute mean polar axis per schema, then pairwise cosine matrix ----
results = {}
schema_names = list(SCHEMAS.keys())

for layer in range(n_layers):
    print(f"\n--- Layer {layer} ---")
    hook = f"blocks.{layer}.hook_resid_post"
    sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=hook, device="cpu")
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res

    def enc(text):
        with torch.no_grad():
            return sae.encode(collect(text, hook)).max(0).values.numpy().astype(np.float64)

    # Compute polar offsets per schema
    polar_offsets = {}  # schema_name -> list of polar offset vectors
    for name, triples in SCHEMAS.items():
        offsets = []
        for _, b, a, c in triples:
            b_v = enc(b)
            a_v = enc(a)
            c_v = enc(c)
            offsets.append(a_v - c_v)  # polar offset = positive_pole − negative_pole
        polar_offsets[name] = offsets

    # Mean axis per schema
    axes = {name: np.mean(np.stack(offsets), axis=0) for name, offsets in polar_offsets.items()}

    # Pairwise cosine matrix
    n = len(schema_names)
    cosine_matrix = np.zeros((n, n))
    for i, ni in enumerate(schema_names):
        for j, nj in enumerate(schema_names):
            cosine_matrix[i, j] = cos(axes[ni], axes[nj])

    results[layer] = cosine_matrix

    # Print matrix
    print(f"  Pairwise cosines between mean polar axes:")
    print(f"  {'':>14}  " + "  ".join(f"{n[:11]:>11}" for n in schema_names))
    for i, ni in enumerate(schema_names):
        row = "  ".join(f"{cosine_matrix[i,j]:>+11.4f}" for j in range(n))
        print(f"  {ni:>14}  {row}")

    del sae
    gc.collect()

del model

# ---- Report ----
report_path = "/Users/macn/Documents/embeddingexp/results_exp22_multi_schema_orthogonality.md"
with open(report_path, "w") as f:
    def out(s=""):
        print(s)
        f.write(s + "\n")

    out("# exp22 — multiple schema polar axes: orthogonality matrix")
    out()
    out("Building on exp21 (cos(UD, IO) ≈ −0.1). Now testing whether 4 Lakoff schemas")
    out("have mutually orthogonal constructed polar axes — strong evidence for")
    out("multiple separable embodied dimensions.")
    out()
    out(f"Triples per schema: UP-DOWN={len(SCHEMAS['UP-DOWN'])}, IN-OUT={len(SCHEMAS['IN-OUT'])}, ")
    out(f"FORWARD-BACK={len(SCHEMAS['FORWARD-BACK'])}, LIGHT-DARK={len(SCHEMAS['LIGHT-DARK'])}, ")
    out(f"TUE-WED sham={len(SCHEMAS['TUE-WED_sham'])}.")
    out()
    out("## Pairwise cosine matrices per layer (Pythia 70m)")
    out()
    for layer in range(n_layers):
        out(f"### Layer {layer}")
        out()
        m = results[layer]
        # Markdown table
        header = "| | " + " | ".join(schema_names) + " |"
        sep = "|---|" + "|".join("---" for _ in schema_names) + "|"
        out(header)
        out(sep)
        for i, ni in enumerate(schema_names):
            row = f"| **{ni}** | " + " | ".join(f"{m[i,j]:+.3f}" for j in range(len(schema_names))) + " |"
            out(row)
        out()
    out("## Interpretation")
    out()
    out("Look at the off-diagonal entries of the 4-schema sub-matrix (UD, IO, FB, LD).")
    out()
    out("- All |cos| < 0.2: 4 mutually orthogonal embodied dimensions ✓ (Lakoff-supported)")
    out("- Some pairs at |cos| > 0.4: those schemas partially collapse into shared axis")
    out("- All cos > 0.5: schemas collapse to one generic polarity ruler")
    out()
    out("Tue-Wed sham row/column: should all be ≈ 0 if methodology working.")
    out("If sham aligns with any schema, sham still has hidden content.")

print(f"\nReport: {report_path}")
torch.save({"results": results, "schema_names": schema_names},
           "/Users/macn/Documents/embeddingexp/exp22_results.pt")
