"""
exp26: Phase A — fill the two cheap measurement gaps from Entry 12.

A1. Within-pair cos(IN_offset, OUT_offset) across layers.
    UD has +0.55 (separated), LR has +0.70-0.87 (mostly aligned).
    IO never directly measured. If near LR → topological/symmetric (consistent
    with IO-as-self/other topology, where the boundary is the same regardless of
    side). If near UD → directional asymmetry to explain.

A2. LIGHT and DARK individual self-projections onto A_updown and A_lightdark.
    exp24 showed DOWN-dominance on A_updown (UP +0.18, DOWN -2.08). Never broke
    out LD analogously. Three possible shapes:
      - DARK-dominant (matches valence-shadow story)
      - symmetric (balanced valence axis)
      - LIGHT-dominant (matches Niamh's "we attend to light" intuition)

For completeness, also report the analogous individual projections for all
poles (UP, DOWN, IN, OUT, FORWARD, BACK, LEFT, RIGHT) onto A_updown and
A_lightdark, plus within-pair cosines for all five schemas at all layers.
"""
from collections import defaultdict

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

# ============================== TRIPLES ==============================
# Imported by copy from exp22 (UD, IO, FB, LD) and exp25 (LR).

UP_DOWN = {
    "temperature": [
        ("The temperature was constant throughout the day.", "The temperature rose throughout the day.", "The temperature plunged throughout the day."),
        ("The thermometer reading held steady overnight.", "The thermometer reading climbed overnight.", "The thermometer reading plummeted overnight."),
        ("Room temperature stayed the same all afternoon.", "Room temperature soared all afternoon.", "Room temperature sank all afternoon."),
        ("The water temperature was even before boiling.", "The water temperature ascended past boiling.", "The water temperature descended past freezing."),
        ("The forecast showed stable temperatures this week.", "The forecast showed increasing temperatures this week.", "The forecast showed tumbling temperatures this week."),
    ],
    "mood": [
        ("Her mood was neutral after the meeting.", "Her mood was elated after the meeting.", "Her mood was dejected after the meeting."),
        ("He felt nothing about the news.", "He felt jubilant about the news.", "He felt morose about the news."),
        ("Their spirits were average that morning.", "Their spirits were radiant that morning.", "Their spirits were forlorn that morning."),
        ("She seemed unchanged by the praise.", "She seemed ecstatic from the praise.", "She seemed glum despite the praise."),
        ("The audience reacted plainly to the song.", "The audience reacted joyfully to the song.", "The audience reacted somberly to the song."),
    ],
    "quantity": [
        ("The company's revenue was stable last quarter.", "The company's revenue grew last quarter.", "The company's revenue declined last quarter."),
        ("Sales held steady through summer.", "Sales increased through summer.", "Sales decreased through summer."),
        ("The population stayed flat over the decade.", "The population multiplied over the decade.", "The population dwindled over the decade."),
        ("Inventory remained constant this month.", "Inventory expanded this month.", "Inventory shrank this month."),
        ("Subscribers stayed the same all year.", "Subscribers accrued all year.", "Subscribers waned all year."),
    ],
    "status": [
        ("Her position was unchanged at the firm.", "Her position was promoted at the firm.", "Her position was demoted at the firm."),
        ("His reputation remained the same in the field.", "His reputation became prominent in the field.", "His reputation was disgraced in the field."),
        ("She held the same rank for years.", "She held a distinguished rank for years.", "She was ousted from her rank."),
        ("His standing was ordinary among peers.", "His standing was esteemed among peers.", "His standing was discredited among peers."),
        ("The professor's recognition was middling.", "The professor's recognition was prestigious.", "The professor's recognition was dethroned."),
    ],
    "health": [
        ("His condition was stable yesterday.", "His condition was thriving yesterday.", "His condition was ailing yesterday."),
        ("Her vitality stayed even after the surgery.", "Her vitality returned vigorously after the surgery.", "Her vitality deteriorated after the surgery."),
        ("The patient remained unchanged.", "The patient was recuperating.", "The patient was languishing."),
        ("The plants looked the same in the garden.", "The plants looked robust in the garden.", "The plants looked sickly in the garden."),
        ("His energy was average that week.", "His energy was vital that week.", "His energy was feeble that week."),
    ],
}

IN_OUT = {
    "literal": [
        ("The papers stayed on the table.", "The papers were contained in the folder.", "The papers were extracted from the folder."),
        ("The puppy sat quietly in the room.", "The puppy was enclosed in its crate.", "The puppy was released from its crate."),
        ("The wine sat on the counter.", "The wine was sealed in the bottle.", "The wine was extracted from the bottle."),
        ("The marbles lay scattered everywhere.", "The marbles were encased in the jar.", "The marbles were ejected from the jar."),
        ("The documents stayed on the desk.", "The documents were wrapped in the envelope.", "The documents were released from the envelope."),
    ],
    "mind": [
        ("She knew nothing about the question.", "She pondered the question for hours.", "She forgot the question entirely."),
        ("He felt indifferent to the memory.", "He harbored the memory for years.", "He banished the memory completely."),
        ("The idea was unfamiliar to her.", "The idea was contemplated by her.", "The idea was discarded by her."),
        ("The thought remained vague all day.", "The thought was ruminated upon for weeks.", "The thought was dismissed in seconds."),
        ("The plan was unconsidered for now.", "The plan was meditated upon at length.", "The plan was purged from memory."),
    ],
    "relationship": [
        ("Their status was ambiguous last year.", "They were married last year.", "They were divorced last year."),
        ("She wasn't sure about him.", "She was engaged to him.", "She was estranged from him."),
        ("He had no particular feelings then.", "He was committed to her formally.", "He was separated from her formally."),
        ("The couple lived ordinarily for years.", "The couple was partnered together for years.", "The couple was single and apart for years."),
        ("Their bond was unclear over time.", "Their bond was a deep marriage over time.", "Their bond was an ugly separation over time."),
    ],
    "time": [
        ("She paid no attention to timing.", "She acted within the hour.", "She acted after the deadline had expired."),
        ("The conversation happened at some point.", "The conversation happened during the meeting.", "The conversation happened after the meeting elapsed."),
        ("He had time vaguely available.", "He had time throughout the day.", "He had time only after the day had lapsed."),
        ("The opportunity was timing-agnostic.", "The opportunity existed amid the negotiations.", "The opportunity was bygone by negotiation's end."),
        ("The decision was timeless in scope.", "The decision spanned the entire year.", "The decision was outdated by the next year."),
    ],
    "difficulty": [
        ("The traveler walked along the path.", "The traveler was stranded in the wilderness.", "The traveler was rescued from the wilderness."),
        ("She faced her usual workload.", "She was mired in endless tasks.", "She was extricated from endless tasks."),
        ("The team handled normal challenges.", "The team was ensnared by hidden problems.", "The team was liberated from hidden problems."),
        ("He worked at the usual pace.", "He was embroiled in the project's chaos.", "He was relieved from the project's chaos."),
        ("The patient maintained their condition.", "The patient was beleaguered by symptoms.", "The patient was salvaged from severe symptoms."),
    ],
}

FORWARD_BACK = {
    "literal_motion": [
        ("She stood still on the path.", "She walked forward along the path.", "She walked backward along the path."),
        ("The car was parked.", "The car drove forward down the street.", "The car reversed down the street."),
        ("He held his position.", "He stepped forward into the room.", "He stepped backward out of the room."),
        ("The runner paused at the line.", "The runner advanced past the line.", "The runner retreated past the line."),
        ("The dancer was still.", "The dancer moved forward across the stage.", "The dancer moved backward across the stage."),
    ],
    "progress": [
        ("The project remained where it was.", "The project advanced significantly.", "The project regressed significantly."),
        ("Sales held at last year's level.", "Sales progressed beyond expectations.", "Sales slipped behind expectations."),
        ("The team's skills were unchanged.", "The team's skills improved markedly.", "The team's skills declined markedly."),
        ("The technology stayed where it was.", "The technology made forward strides.", "The technology took backward steps."),
        ("Her career remained level.", "Her career moved forward rapidly.", "Her career moved backward steadily."),
    ],
    "time": [
        ("She thought about the present.", "She thought about the future ahead.", "She thought about the past behind her."),
        ("The schedule was today.", "The schedule was pushed forward.", "The schedule was pushed back."),
        ("The deadline was current.", "The deadline was moved forward in time.", "The deadline was moved backward in time."),
        ("The meeting was scheduled.", "The meeting was rescheduled to a later date.", "The meeting was rescheduled to an earlier date."),
        ("The plan was in motion.", "The plan was projected into the future.", "The plan was rooted in the past."),
    ],
    "development": [
        ("Her thinking stayed the same.", "Her thinking evolved forward.", "Her thinking devolved backward."),
        ("The conversation continued.", "The conversation moved forward productively.", "The conversation backtracked over old ground."),
        ("The story was in progress.", "The story moved the plot forward.", "The story flashed back to the past."),
        ("The argument continued.", "The argument advanced new points.", "The argument retreated to old positions."),
        ("Her understanding was steady.", "Her understanding grew forward.", "Her understanding reverted backward."),
    ],
    "journey": [
        ("The path stretched ahead.", "She journeyed forward on the path.", "She turned back along the path."),
        ("The ship was at sea.", "The ship sailed onward to the destination.", "The ship returned to the port."),
        ("The expedition was underway.", "The expedition pressed onward into the wilderness.", "The expedition retreated from the wilderness."),
        ("The pilgrimage continued.", "The pilgrims advanced toward the shrine.", "The pilgrims withdrew from the shrine."),
        ("The voyage was in progress.", "The vessel proceeded toward harbor.", "The vessel reversed course from harbor."),
    ],
}

LIGHT_DARK = {
    "literal_illumination": [
        ("The room was at neutral light.", "The room was brightly illuminated.", "The room was dimly shadowed."),
        ("The candle stood ready.", "The candle glowed warmly.", "The candle was extinguished into darkness."),
        ("The sky had some clouds.", "The sky was radiantly clear.", "The sky was murkily overcast."),
        ("The garden was at twilight.", "The garden was bright with sunshine.", "The garden was dark with shadow."),
        ("The lamp stood on the table.", "The lamp shone brightly.", "The lamp was dimmed completely."),
    ],
    "clarity": [
        ("The situation was undetermined.", "The situation was illuminated by analysis.", "The situation was shrouded in confusion."),
        ("The question remained.", "The question became luminously clear.", "The question became hopelessly obscure."),
        ("Her perspective was indifferent.", "Her perspective was radiantly informed.", "Her perspective was darkly clouded."),
        ("The investigation continued.", "The investigation shed light on the truth.", "The investigation deepened the mystery."),
        ("The lecture was happening.", "The lecture clarified the concept brilliantly.", "The lecture obscured the concept entirely."),
    ],
    "hope": [
        ("Her outlook was steady.", "Her outlook was bright with hope.", "Her outlook was dark with despair."),
        ("The future was uncertain.", "The future shone with promise.", "The future loomed darkly grim."),
        ("He felt the day calmly.", "He felt the day glowing with optimism.", "He felt the day shadowed by dread."),
        ("The community waited.", "The community was bright with anticipation.", "The community was grim with foreboding."),
        ("Their plans were drafted.", "Their plans glowed with possibility.", "Their plans were shadowed by setbacks."),
    ],
    "goodness": [
        ("His character was ordinary.", "His character was bright and pure.", "His character was tainted and evil."),
        ("The deed was unremarkable.", "The deed was radiantly noble.", "The deed was darkly cruel."),
        ("The story was neutral.", "The story celebrated luminous goodness.", "The story portrayed shadowy wickedness."),
        ("Their motives were unknown.", "Their motives shone with virtue.", "Their motives were shadowed by greed."),
        ("The ruler was indifferent.", "The ruler was illuminated by justice.", "The ruler was darkened by tyranny."),
    ],
    "knowledge": [
        ("The topic was unfamiliar.", "The topic was illuminated by study.", "The topic remained shrouded in mystery."),
        ("The truth was hidden.", "The truth came to brilliant light.", "The truth was buried in darkness."),
        ("The mystery existed.", "The mystery was solved brilliantly.", "The mystery deepened obscurely."),
        ("She approached the subject.", "She approached the subject with enlightenment.", "She approached the subject with darkness."),
        ("The data was uninterpreted.", "The data was clarified luminously.", "The data was obscured by complexity."),
    ],
}

LEFT_RIGHT = [
    ("She stood in the middle of the room.", "She walked to the left side of the room.", "She walked to the right side of the room."),
    ("The car was parked in the center.", "The car turned left at the intersection.", "The car turned right at the intersection."),
    ("The arrow points straight ahead.", "The arrow points to the left.", "The arrow points to the right."),
    ("The dancer stood centered.", "The dancer stepped to the left.", "The dancer stepped to the right."),
    ("They walked down the middle of the road.", "They walked down the left side of the road.", "They walked down the right side of the road."),
    ("She used both hands equally.", "She used her left hand for everything.", "She used her right hand for everything."),
    ("He was ambidextrous from childhood.", "He was left-handed from childhood.", "He was right-handed from childhood."),
    ("The player swung with either side.", "The player swung from the left side.", "The player swung from the right side."),
    ("She wore a watch on either wrist.", "She wore a watch on her left wrist.", "She wore a watch on her right wrist."),
    ("The artist held the brush in either hand.", "The artist held the brush in the left hand.", "The artist held the brush in the right hand."),
    ("She drove down the road carefully.", "She drove on the left side of the road.", "She drove on the right side of the road."),
    ("The driver stayed in the lane.", "The driver moved to the left lane.", "The driver moved to the right lane."),
    ("The car drifted on the highway.", "The car drifted to the left lane.", "The car drifted to the right lane."),
    ("The compass needle pointed straight.", "The compass needle pointed left.", "The compass needle pointed right."),
    ("The path forked ahead.", "The path forked to the left.", "The path forked to the right."),
]

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
    "UP-DOWN":       (flatten(UP_DOWN),     "UP",      "DOWN"),
    "IN-OUT":        (flatten(IN_OUT),      "IN",      "OUT"),
    "FORWARD-BACK":  (flatten(FORWARD_BACK), "FORWARD", "BACK"),
    "LIGHT-DARK":    (flatten(LIGHT_DARK),  "LIGHT",   "DARK"),
    "LEFT-RIGHT":    (flatten(LEFT_RIGHT),  "LEFT",    "RIGHT"),
}

for name, (triples, _, _) in SCHEMAS.items():
    print(f"{name}: {len(triples)} triples")


# ============================== MODEL ==============================
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


# ============================== ANALYSIS ==============================
def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


per_layer = {}

for layer in range(n_layers):
    print(f"\n=== Layer {layer} ===")
    hook = f"blocks.{layer}.hook_resid_post"
    sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=hook, device="cpu")
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res

    def enc(text):
        with torch.no_grad():
            return sae.encode(collect(text, hook)).max(0).values.numpy().astype(np.float64)

    # First pass: encode all sentences and compute offsets per (schema, pair, pole)
    offsets = {schema: {"a": [], "c": []} for schema in SCHEMAS}  # "a" = first pole, "c" = second
    within_pair_cos = {}

    for schema, (triples, name_a, name_c) in SCHEMAS.items():
        cosines = []
        for (dom, b, a, c) in triples:
            b_v = enc(b)
            a_v = enc(a)
            c_v = enc(c)
            off_a = a_v - b_v
            off_c = c_v - b_v
            offsets[schema]["a"].append(off_a)
            offsets[schema]["c"].append(off_c)
            cosines.append(cos(off_a, off_c))
        within_pair_cos[schema] = float(np.mean(cosines))

    # Build polar axes
    A_axes = {}
    for schema, (_, _, _) in SCHEMAS.items():
        a_arr = np.stack(offsets[schema]["a"])
        c_arr = np.stack(offsets[schema]["c"])
        axis = (a_arr - c_arr).mean(axis=0)
        nrm = np.linalg.norm(axis)
        A_axes[schema] = axis / nrm if nrm > 1e-12 else axis

    # Project each pole's offsets onto each axis
    projections = {}
    for axis_schema, axis_vec in A_axes.items():
        projections[axis_schema] = {}
        for off_schema, (_, name_a, name_c) in SCHEMAS.items():
            a_arr = np.stack(offsets[off_schema]["a"])
            c_arr = np.stack(offsets[off_schema]["c"])
            projections[axis_schema][off_schema] = {
                name_a: float(np.mean(a_arr @ axis_vec)),
                name_c: float(np.mean(c_arr @ axis_vec)),
            }

    # Report within-pair cosines
    print("\nWithin-pair cosines (cos(offset_a, offset_c) averaged across pairs):")
    for schema, val in within_pair_cos.items():
        print(f"  {schema:>14}:  cos = {val:+.4f}")

    # Report projections onto A_updown and A_lightdark (the two we care about most)
    for axis_schema in ["UP-DOWN", "LIGHT-DARK"]:
        axis_name = axis_schema.replace("-", "/")
        print(f"\nProjections onto A_{axis_name.lower()}:")
        for off_schema, polepairs in projections[axis_schema].items():
            poles = list(polepairs.items())
            p1, p2 = poles[0], poles[1]
            print(f"  {off_schema:>14}:  {p1[0]:>7} = {p1[1]:>+8.4f}    {p2[0]:>7} = {p2[1]:>+8.4f}")

    per_layer[layer] = {
        "within_pair_cos": within_pair_cos,
        "projections": projections,
    }
    del sae


# ============================== SUMMARY TABLES ==============================
print("\n\n" + "="*80)
print("SUMMARY: within-pair cosines by layer")
print("="*80)
print(f"  {'layer':>5}  " + "  ".join(f"{s:>14}" for s in SCHEMAS))
for layer in range(n_layers):
    row = "  ".join(f"{per_layer[layer]['within_pair_cos'][s]:>+14.4f}" for s in SCHEMAS)
    print(f"  {layer:>5}  {row}")

print("\n\n" + "="*80)
print("SUMMARY: projections onto A_updown by layer (each pole)")
print("="*80)
print(f"  {'layer':>5}  {'UP':>7}  {'DOWN':>7}  {'IN':>7}  {'OUT':>7}  {'FORWARD':>7}  {'BACK':>7}  {'LIGHT':>7}  {'DARK':>7}  {'LEFT':>7}  {'RIGHT':>7}")
for layer in range(n_layers):
    p = per_layer[layer]["projections"]["UP-DOWN"]
    vals = [p["UP-DOWN"]["UP"], p["UP-DOWN"]["DOWN"],
            p["IN-OUT"]["IN"], p["IN-OUT"]["OUT"],
            p["FORWARD-BACK"]["FORWARD"], p["FORWARD-BACK"]["BACK"],
            p["LIGHT-DARK"]["LIGHT"], p["LIGHT-DARK"]["DARK"],
            p["LEFT-RIGHT"]["LEFT"], p["LEFT-RIGHT"]["RIGHT"]]
    print(f"  {layer:>5}  " + "  ".join(f"{v:>+7.3f}" for v in vals))

print("\n\n" + "="*80)
print("SUMMARY: projections onto A_lightdark by layer (each pole)")
print("="*80)
print(f"  {'layer':>5}  {'UP':>7}  {'DOWN':>7}  {'IN':>7}  {'OUT':>7}  {'FORWARD':>7}  {'BACK':>7}  {'LIGHT':>7}  {'DARK':>7}  {'LEFT':>7}  {'RIGHT':>7}")
for layer in range(n_layers):
    p = per_layer[layer]["projections"]["LIGHT-DARK"]
    vals = [p["UP-DOWN"]["UP"], p["UP-DOWN"]["DOWN"],
            p["IN-OUT"]["IN"], p["IN-OUT"]["OUT"],
            p["FORWARD-BACK"]["FORWARD"], p["FORWARD-BACK"]["BACK"],
            p["LIGHT-DARK"]["LIGHT"], p["LIGHT-DARK"]["DARK"],
            p["LEFT-RIGHT"]["LEFT"], p["LEFT-RIGHT"]["RIGHT"]]
    print(f"  {layer:>5}  " + "  ".join(f"{v:>+7.3f}" for v in vals))


# Save results
torch.save({"per_layer": per_layer}, "/Users/macn/Documents/embeddingexp/exp26_results.pt")
print("\nSaved: exp26_results.pt")
