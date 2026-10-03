"""exp182_build_stimuli.py — frozen stimuli for exp182 (position-by-position
test of "UPness is like potential energy"). Tokenizers only, no model.
Pairs are exp180's, unchanged. Frames are new, with a long continuation
after the slot, and contain none of the 162 pair words."""
import hashlib, json
from transformers import AutoTokenizer
ROOT = "/Users/macn/Documents/embeddingexp"
K = 12
FRAMES = [
    "the card at the centre of the table said {w} and the children passed it around until everyone there had read it twice",
    "someone had painted the single word {w} beside the door and the paint still smelled when we arrived that morning with the keys",
    "the answer she gave was simply {w} and the rest of the class wrote it carefully with pencils while the teacher waited",
    "the ticket carried only the word {w} and the man at the gate looked at it for a while and then nodded",
    "his reply to the message was only {w} and then he said nothing else to anyone for the rest of the week",
    "the sailor muttered the word {w} and then went straight to mending the net as though nobody had asked him anything",
    "the next clue of the puzzle read {w} and we spent most of the afternoon arguing about what it could possibly mean",
    "at the end of the letter she wrote {w} and signed her name with the green pen she always kept by the window",
]
old = json.load(open(f"{ROOT}/exp180_stimuli.json"))
pairs = old["pairs"]
words = {w for ps in pairs.values() for p in ps for w in p}
prev = set(old["frames"]) | set(json.load(open(f"{ROOT}/exp175_stimuli.json"))["frames"])
tks = [AutoTokenizer.from_pretrained(n) for n in ("EleutherAI/pythia-410m", "gpt2-medium")]
for fr in FRAMES:
    assert fr not in prev, "frame reused"
    ws = fr.split(" ")
    assert ws.count("{w}") == 1
    hit = set(ws) & words
    assert not hit, (fr, hit)
    tail = " " + " ".join(ws[ws.index("{w}") + 1:])
    for tk in tks:
        n = len(tk.encode(tail, add_special_tokens=False))
        assert n >= K, (fr, n)
payload = {"experiment": "exp182", "frames": FRAMES, "pairs": pairs, "K": K,
           "order": "first word of each pair is the UP / default / positive member"}
blob = json.dumps(payload, indent=1, sort_keys=True).encode()
open(f"{ROOT}/exp182_stimuli.json", "wb").write(blob)
print("frames valid:", len(FRAMES), "| continuation tokens after the slot (pythia, gpt2):",
      [tuple(len(tk.encode(" " + " ".join(fr.split(" ")[fr.split(" ").index("{w}") + 1:]), add_special_tokens=False)) for tk in tks) for fr in FRAMES])
print("sha256:", hashlib.sha256(blob).hexdigest())
