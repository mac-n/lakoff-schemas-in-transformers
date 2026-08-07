"""
exp154d_subsample_control.py — is exp154c's 30% retention about EXCLUDING
the test words, or just about estimating d_norm from fewer words?

exp154c stripped with d_norm estimated on 312 held-out words and the
inflectional sink retained ~28-42% at L8-L20 (vs ~0% with the full-set
d_norm in exp154). Two readings:
  (a) genuine: the extra collapse in exp154 was circular (d_norm partly
      fit to the test words); the retained component is real content.
  (b) noise: 312 words just estimate d_norm less well than 489; any
      same-size subset would leave ~30% behind.

Design: 20 seeds. Each seed samples a random 312-word estimation subset
from ALL 489 words (test words allowed in, in their natural proportion),
builds d_norm/d_disp exactly as exp154c does, strips, and records the
inflectional mean sink after. This gives a size-matched band. Compare
exp154c's held-out retention (parsed from exp154c_output.txt on disk)
against the band.

PRE-REGISTRATION (2026-06-11, before running; this Claude):
  Committed prediction: (a). Random size-312 subsets contain ~64% of the
  test words on average and will collapse the sink nearly as well as the
  full set — retained fraction ~0-15%, band well separated from the
  held-out values at L8+.
  Decision rule (per layer at L8, L12, L16, L20; L4 already collapses
  under both, uninformative):
    S1 held-out retention LESS NEGATIVE than (above) the subsample band
       max at >= 3 of 4 layers: exclusion-specific. exp154c's H3 stands —
       two-component story, prefer held-out numbers in the paper.
    S2 held-out retention within the band at >= 2 of 4 layers: the
       retention is estimation noise. exp154's full collapse stands;
       withdraw the circularity concern.
    S3 otherwise: unclear; design a larger-vocabulary version before
       deciding (band may be too wide to discriminate).
"""

import re

import numpy as np
import torch
from transformer_lens import HookedTransformer

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML

device = "mps"
print("Loading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
model.eval()

LAYERS = [4, 8, 12, 16, 20]
DECISION_LAYERS = [8, 12, 16, 20]
hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]

SUFFIX_PAIRS = {
    "ER_comparative": [("big","bigger"),("small","smaller"),("tall","taller"),
        ("high","higher"),("low","lower"),("deep","deeper"),("wide","wider"),
        ("fast","faster"),("slow","slower"),("old","older"),("new","newer"),
        ("hot","hotter"),("cold","colder"),("hard","harder"),("soft","softer")],
    "EST_superlative": [("big","biggest"),("small","smallest"),("tall","tallest"),
        ("high","highest"),("low","lowest"),("deep","deepest"),("old","oldest"),
        ("new","newest"),("hot","hottest"),("cold","coldest"),("hard","hardest")],
    "ING_progressive": [("walk","walking"),("run","running"),("jump","jumping"),
        ("sit","sitting"),("stand","standing"),("swim","swimming"),("think","thinking"),
        ("talk","talking"),("sing","singing"),("dance","dancing"),("play","playing"),
        ("work","working"),("read","reading"),("write","writing"),("eat","eating")],
    "ED_past": [("walk","walked"),("jump","jumped"),("look","looked"),
        ("talk","talked"),("play","played"),("work","worked"),("ask","asked"),
        ("call","called"),("learn","learned"),("move","moved"),("stop","stopped"),
        ("start","started")],
    "S_plural": [("cat","cats"),("dog","dogs"),("book","books"),("house","houses"),
        ("car","cars"),("tree","trees"),("bird","birds"),("hand","hands"),
        ("eye","eyes"),("girl","girls"),("boy","boys"),("year","years")],
    "UN_negation": [("happy","unhappy"),("kind","unkind"),("healthy","unhealthy"),
        ("safe","unsafe"),("clear","unclear"),("clean","unclean"),("fair","unfair"),
        ("certain","uncertain"),("known","unknown"),("seen","unseen")],
    "RE_repetition": [("do","redo"),("make","remake"),("build","rebuild"),
        ("write","rewrite"),("read","reread"),("start","restart"),
        ("create","recreate"),("paint","repaint")],
}
SCHEMA_NAMES = ["UP-DOWN", "IN-OUT_CLEAN", "FORWARD-BACK", "PATH-MOTION",
                "LIGHT-DARK", "FORCE", "BALANCE", "DIFFICULTY-BURDEN"]
COMMON = ["the","of","and","to","in","is","it","you","that","he","was","for",
          "on","are","with","as","his","they","at","be"]
RARE = ["serendipity","ostracize","perspicacity","obfuscate","sycophant"]
SUFFIX_ORDER = list(SUFFIX_PAIRS.keys())
INFL = ["ER_comparative","EST_superlative","ING_progressive","ED_past","S_plural"]

all_words = set(COMMON + RARE)
for pairs in SUFFIX_PAIRS.values():
    for b, i in pairs:
        all_words.add(b); all_words.add(i)
for sn in SCHEMA_NAMES:
    for p, n in LAKOFF_SCHEMAS_MML[sn]:
        all_words.add(p); all_words.add(n)
all_words = sorted(all_words)

EST_SIZE = 312  # match exp154c's estimation-set size exactly
N_SEEDS = 20

# ---- parse exp154c reference values from disk ----
ref_before, ref_after = {}, {}
with open("/Users/macn/Documents/embeddingexp/exp154c_output.txt") as f:
    cur = None
    for line in f:
        m = re.match(r"--- Layer (\d+) ---", line.strip())
        if m:
            cur = int(m.group(1)); continue
        m = re.match(r"inflectional mean sink:\s*([+-]\d+\.\d+)\s*->\s*([+-]\d+\.\d+)",
                     line.strip())
        if m and cur is not None:
            ref_before[cur] = float(m.group(1)); ref_after[cur] = float(m.group(2))
print(f"exp154c held-out reference (parsed): after-strip = {ref_after}")
assert set(ref_after) == set(LAYERS), "failed to parse exp154c reference values"

print(f"Collecting residuals for {len(all_words)} words at {LAYERS}...")
residuals = {}
for k, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    residuals[w] = {L: cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy()
                    for L in LAYERS}
    if (k + 1) % 100 == 0:
        print(f"  {k+1}/{len(all_words)}")

print("\n" + "=" * 78)
print("exp154d — size-matched random-subset d_norm control")
print("=" * 78)

rng = np.random.default_rng(11)
verdict_rows = {}

for L in LAYERS:
    arr = np.stack([residuals[w][L] for w in all_words], axis=0)
    aniso = arr.mean(axis=0); aniso = aniso / np.linalg.norm(aniso)

    def mean_acts(words):
        return np.mean([residuals[w][L] for w in words], axis=0)

    freq_raw = mean_acts(COMMON) - mean_acts(RARE)
    freq = freq_raw / np.linalg.norm(freq_raw)
    freq_orth = freq - (freq @ aniso) * aniso
    freq_orth = freq_orth / np.linalg.norm(freq_orth)

    def strip(d):
        d = d - (d @ aniso) * aniso
        d = d - (d @ freq_orth) * freq_orth
        return d / np.linalg.norm(d)

    def proj_out(d, axis):
        d = d - (d @ axis) * axis
        return d / np.linalg.norm(d)

    norms = {w: float(np.linalg.norm(residuals[w][L])) for w in all_words}
    units = {w: residuals[w][L] / norms[w] for w in all_words}

    def build_norm_dirs(words):
        nv = np.array([norms[w] for w in words])
        zz = {w: (norms[w] - nv.mean()) / nv.std() for w in words}
        dn = np.sum([zz[w] * units[w] for w in words], axis=0)
        dn = strip(dn / np.linalg.norm(dn))
        az = {w: abs(zz[w]) for w in words}
        am = np.mean(list(az.values())); asd = np.std(list(az.values()))
        dd = np.sum([((az[w] - am) / asd) * units[w] for w in words], axis=0)
        dd = strip(dd / np.linalg.norm(dd))
        return dn, dd

    def schema_dir(sn):
        pairs = LAKOFF_SCHEMAS_MML[sn]
        pos = sorted(set(p[0] for p in pairs)); neg = sorted(set(p[1] for p in pairs))
        raw = mean_acts(pos) - mean_acts(neg)
        return strip(raw / np.linalg.norm(raw))

    bal = schema_dir("BALANCE")

    def suffix_dir(sn):
        diffs = []
        for base, infl in SUFFIX_PAIRS[sn]:
            b = residuals[base][L]; i = residuals[infl][L]
            diffs.append(i / np.linalg.norm(i) - b / np.linalg.norm(b))
        raw = np.mean(diffs, axis=0)
        return strip(raw / np.linalg.norm(raw))

    sufs = {sn: suffix_dir(sn) for sn in SUFFIX_ORDER}
    infl_before = float(np.mean([sufs[sn] @ bal for sn in INFL]))

    after_vals = []
    for _ in range(N_SEEDS):
        subset = list(rng.choice(all_words, size=EST_SIZE, replace=False))
        dn, dd = build_norm_dirs(subset)
        dd_o = proj_out(dd, dn)
        bal_s = proj_out(proj_out(bal, dn), dd_o)
        vals = [float(proj_out(proj_out(sufs[sn], dn), dd_o) @ bal_s) for sn in INFL]
        after_vals.append(np.mean(vals))
    after_vals = np.array(after_vals)

    # Compare RETAINED FRACTIONS (after/before), not raw after-values:
    # retaining more sink = larger fraction. (v1 of this script compared raw
    # values with the inequality reversed and mislabelled the outcome; the
    # printed numbers were correct, the labels were not. Fixed 2026-06-11.)
    ho = ref_after[L]
    ho_frac = ho / infl_before
    band_fracs = after_vals / infl_before
    exclusion = ho_frac > band_fracs.max()  # held-out retains more than every subsample
    verdict_rows[L] = (band_fracs, ho_frac, exclusion)

    print(f"\n--- Layer {L} ---")
    print(f"  sink before = {infl_before:+.3f} (exp154c parity: {ref_before[L]:+.3f})")
    print(f"  random size-{EST_SIZE} subset strip, sink after ({N_SEEDS} seeds): "
          f"{after_vals.mean():+.3f} ± {after_vals.std():.3f} "
          f"[{after_vals.min():+.3f}, {after_vals.max():+.3f}]")
    print(f"  retained fraction band: "
          f"[{100*after_vals.min()/infl_before:.0f}%, {100*after_vals.max()/infl_before:.0f}%] "
          f"(mean {100*after_vals.mean()/infl_before:.0f}%)")
    print(f"  exp154c HELD-OUT sink after = {ho:+.3f} (retains {100*ho_frac:.0f}%) -> "
          f"{'OUTSIDE band, retains more (exclusion-specific)' if exclusion else 'within band (size noise)'}")

print("\n" + "=" * 78)
print("VERDICT vs pre-registered decision rule")
print("=" * 78)
n_excl = sum(1 for L in DECISION_LAYERS if verdict_rows[L][2])
n_within = sum(1 for L in DECISION_LAYERS
               if verdict_rows[L][0].min() <= verdict_rows[L][1] <= verdict_rows[L][0].max())
print(f"  decision layers {DECISION_LAYERS}: held-out retains more than entire band "
      f"at {n_excl}/4, within band at {n_within}/4")
if n_excl >= 3:
    print("  -> S1: exclusion-specific. exp154's extra collapse was circular;")
    print("     exp154c's H3 (two-component) stands. Prefer held-out numbers.")
elif n_within >= 2:
    print("  -> S2: estimation noise. exp154's full collapse stands;")
    print("     withdraw the circularity concern.")
else:
    print("  -> S3: unclear; band too wide — design a larger-vocabulary version.")
