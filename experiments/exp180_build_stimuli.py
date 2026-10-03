"""exp180_build_stimuli.py — frozen stimuli for exp180 (paired-opposites retry
of the word-level "holding against gravity" lean). Tokenizer only, no model.
A pair is kept only if BOTH words are one token with a leading space in BOTH
tokenizers (Pythia and GPT-2) and neither word has been used already."""
import hashlib, json, sys
from transformers import AutoTokenizer
OUT = "/Users/macn/Documents/embeddingexp/exp180_stimuli.json"
VERTICAL = """up/down above/below over/under top/bottom high/low higher/lower highest/lowest upward/downward
rise/fall rising/falling rose/fell raise/drop raised/dropped lift/sink lifted/fallen climb/descend climbing/sinking
climbed/sank upstairs/downstairs ceiling/floor roof/basement sky/ground peak/valley summit/pit tall/deep height/depth
heights/depths overhead/underneath atop/beneath elevated/buried tower/cellar""".split()
POLAR = """big/small large/little long/short wide/narrow fast/slow old/young hot/cold heavy/light hard/soft thick/thin
strong/weak full/empty rich/poor early/late first/last open/closed wet/dry loud/quiet bright/dark many/few more/less
near/far front/back forward/backward inside/outside before/after in/out on/off major/minor plus/minus""".split()
VALENCE = """good/bad happy/sad love/hate joy/grief kind/cruel gentle/harsh beautiful/ugly wonderful/terrible calm/angry
hope/fear proud/ashamed smile/cry laugh/weep merry/gloomy lovely/awful sweet/bitter friend/enemy win/lose
success/failure safe/dangerous clean/dirty glad/upset""".split()
FRAMES = [
    "the teacher asked us to spell the word {w} before we opened our books for the day",
    "my first guess in the game was {w} and the second player wrote it on a card",
    "he repeated the word {w} twice so that everyone in the hall could hear it",
    "the label on the old box said {w} and nothing else was written there at all",
    "at the start of the list we found {w} and then a long row of other entries",
    "her note contained only the word {w} and a small drawing of a bird beside it",
    "the password for that week was {w} and it changed again on the following monday",
    "somebody had typed {w} into the search field and left the screen as it was",
]
tks = [AutoTokenizer.from_pretrained(n) for n in ("EleutherAI/pythia-410m", "gpt2-medium")]
one = lambda w: all(len(tk.encode(" " + w, add_special_tokens=False)) == 1 for tk in tks)
used, sets, dropped = set(), {}, {}
for name, raw in (("vertical", VERTICAL), ("polar", POLAR), ("valence", VALENCE)):
    keep, drop = [], []
    for p in raw:
        a, b = p.split("/")
        if one(a) and one(b) and a not in used and b not in used and a != b:
            keep.append([a, b]); used |= {a, b}
        else:
            drop.append(p)
    sets[name] = keep; dropped[name] = drop
old = json.load(open("/Users/macn/Documents/embeddingexp/exp175_stimuli.json"))
assert not (set(FRAMES) & set(old["frames"])), "a frame is reused from exp175"
vert_words = {w for p in sets["vertical"] for w in p}
for fr in FRAMES:
    ws = fr.split(" ")
    assert ws.count("{w}") == 1 and len(ws) - 1 - ws.index("{w}") >= 8, fr
    assert not (set(ws) & vert_words), (fr, set(ws) & vert_words)
payload = {"experiment": "exp180", "frames": FRAMES, "pairs": sets, "order": "first word of each pair is the UP / unmarked / positive member"}
blob = json.dumps(payload, indent=1, sort_keys=True).encode()
open(OUT, "wb").write(blob)
for name in sets:
    print(f"{name}: {len(sets[name])} pairs kept; dropped {dropped[name]}")
    print("   ", " ".join("/".join(p) for p in sets[name]))
print("sha256:", hashlib.sha256(blob).hexdigest())
