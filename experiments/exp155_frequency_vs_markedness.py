"""
exp155_frequency_vs_markedness.py — is the norm displacement of inflected
forms (exp154) frequency or markedness?

exp154 found inflected forms sit at uniformly lower residual norms in
Pythia (the physiological substrate of the now-collapsed BALANCE sink).
Inflected forms are also systematically RARER than bases. This experiment
decides which word the paper uses.

Design problem acknowledged up front: frequency and markedness travel
together in natural language (frequency is one of Greenberg's markedness
diagnostics), and reversed-frequency morphological pairs ("scissors",
"renowned") are mostly LEXICALISED — markedness arguably flips with
frequency there. So reversed pairs alone cannot decide. Three legs:

  LEG A — continuous: across the exp138 suffix pairs, regress per-pair
    Δnorm on Δzipf (wordfreq zipf scale). Slope = frequency component.
    INTERCEPT at Δzipf=0 = markedness displacement with frequency held
    equal. (Also corr(zipf, ||r||) across all pool words per layer.)
  LEG B — pseudo-pair control (the decisive one): random UNRELATED word
    pairs from the pool, matched to each morphological pair's Δzipf
    (±0.15). If any pair with the same frequency gap shows the same
    Δnorm, the displacement is frequency, not morphology.
  LEG C — reversed-frequency morphological pairs (zipf(infl) > base
    + 0.3, verified at runtime): supplementary, interpreted with the
    lexicalisation caveat.

PRE-REGISTRATION (2026-06-10, before running):
  Committed prediction (this Claude, deliberately NOT hedging toward the
  middle after three same-direction misses today — this is what I actually
  believe): F1-leaning — frequency carries the bulk. Specifically:
  corr(zipf, norm) will be strong positive (rarer = lower norm,
  |r| >= 0.5 at L8+); pseudo-pairs at matched Δzipf will reproduce >= 70%
  of the morphological pairs' Δnorm; the Leg-A intercept will be small
  relative to the total displacement. Consequence if right: the paper
  says "frequency/markedness displacement" (with the Greenberg note),
  and the load-bearing result remains the concept-physiology COUPLING
  (BALANCE~d_norm in Pythia +0.7 vs GloVe −0.07), which is untouched by
  this control either way.
  Decision rule:
    F1 frequency: pseudo-pairs >= 70% of morph Δnorm AND intercept
       indistinguishable from 0 (within bootstrap 95% CI).
    F2 markedness: pseudo-pairs <= 30% AND intercept clearly negative.
    F3 mixed: anything else — report both components with magnitudes.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from wordfreq import zipf_frequency

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML

# ---- exp138 lists ----
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
INFL = ["ER_comparative","EST_superlative","ING_progressive","ED_past","S_plural"]

# Leg C candidates — runtime-verified zipf(infl) >= zipf(base) + 0.3
REVERSED_CANDIDATES = [
    ("stun","stunning"),("gild","gilded"),("renown","renowned"),
    ("scissor","scissors"),("trouser","trousers"),("outskirt","outskirts"),
    ("disgruntle","disgruntled"),("dilapidate","dilapidated"),
    ("bedraggle","bedraggled"),("dishevel","disheveled"),
    ("emaciate","emaciated"),("exasperate","exasperated"),
    ("allege","alleged"),("gnarl","gnarled"),("parch","parched"),
    ("wizen","wizened"),("embattle","embattled"),("tatter","tattered"),
    ("belong","belongings"),("furnish","furnishings"),
]

ZIPF = lambda w: zipf_frequency(w, "en")
reversed_pairs = [(b, i) for b, i in REVERSED_CANDIDATES
                  if ZIPF(i) >= ZIPF(b) + 0.3 and ZIPF(b) > 0]
print(f"Reversed-frequency pairs verified: {len(reversed_pairs)}/{len(REVERSED_CANDIDATES)}")
for b, i in reversed_pairs:
    print(f"  {b} ({ZIPF(b):.2f}) -> {i} ({ZIPF(i):.2f})")

# ---- collect residuals ----
device = "mps"
print("\nLoading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
model.eval()
LAYERS = [4, 8, 12, 16, 20]
hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]

all_words = set(COMMON + RARE)
for pairs in SUFFIX_PAIRS.values():
    for b, i in pairs:
        all_words.add(b); all_words.add(i)
for b, i in reversed_pairs:
    all_words.add(b); all_words.add(i)
for sn in SCHEMA_NAMES:
    for p, n in LAKOFF_SCHEMAS_MML[sn]:
        all_words.add(p); all_words.add(n)
all_words = sorted(all_words)
zipfs = {w: ZIPF(w) for w in all_words}
no_freq = [w for w in all_words if zipfs[w] == 0]
if no_freq:
    print(f"\nWords with no frequency data (excluded from freq analyses): {no_freq}")

print(f"\nCollecting residuals for {len(all_words)} words at {LAYERS}...")
residuals = {}
for k, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    residuals[w] = {L: cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy()
                    for L in LAYERS}
    if (k + 1) % 100 == 0:
        print(f"  {k+1}/{len(all_words)}")


def corrf(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y)))


rng = np.random.default_rng(11)
pool = [w for w in all_words if zipfs[w] > 0]

print("\n" + "=" * 78)
print("exp155 — frequency vs markedness in the norm displacement")
print("=" * 78)

verdicts = {}
for L in LAYERS:
    norms = {w: float(np.linalg.norm(residuals[w][L])) for w in all_words}

    # strip machinery for BALANCE axis (exp138 protocol)
    arr = np.stack([residuals[w][L] for w in all_words], axis=0)
    aniso = arr.mean(axis=0); aniso /= np.linalg.norm(aniso)
    def mean_acts(ws): return np.mean([residuals[w][L] for w in ws], axis=0)
    freq_raw = mean_acts(COMMON) - mean_acts(RARE)
    fq = freq_raw / np.linalg.norm(freq_raw)
    fq_o = fq - (fq @ aniso) * aniso; fq_o /= np.linalg.norm(fq_o)
    def strip(d):
        d = d - (d @ aniso) * aniso
        d = d - (d @ fq_o) * fq_o
        return d / np.linalg.norm(d)
    bp = LAKOFF_SCHEMAS_MML["BALANCE"]
    bal = strip((mean_acts(sorted(set(p[0] for p in bp))) -
                 mean_acts(sorted(set(p[1] for p in bp)))))

    def unit(w): return residuals[w][L] / norms[w]
    def dpair(b, i):
        return (norms[i] - norms[b],
                zipfs[i] - zipfs[b],
                float((unit(i) - unit(b)) @ bal))

    # LEG A0: word-level coupling
    r_freq_norm = corrf([zipfs[w] for w in pool], [norms[w] for w in pool])
    r_freq_bal = corrf([zipfs[w] for w in pool], [float(unit(w) @ bal) for w in pool])

    # LEG A: regression over inflectional pairs
    morph = [dpair(b, i) for sn in INFL for b, i in SUFFIX_PAIRS[sn]]
    dn = np.array([m[0] for m in morph]); dz = np.array([m[1] for m in morph])
    db = np.array([m[2] for m in morph])
    A = np.vstack([dz, np.ones_like(dz)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, dn, rcond=None)
    pred = A @ np.array([slope, intercept])
    r2 = 1 - np.sum((dn - pred) ** 2) / np.sum((dn - dn.mean()) ** 2)
    # bootstrap CI on intercept
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(dn), len(dn))
        Ab = np.vstack([dz[idx], np.ones(len(idx))]).T
        coef, *_ = np.linalg.lstsq(Ab, dn[idx], rcond=None)
        boots.append(coef[1])
    ci = np.percentile(boots, [2.5, 97.5])

    # LEG B: Δzipf-matched pseudo-pairs (5 per morph pair)
    pseudo_dn, pseudo_db, matched = [], [], 0
    morph_words = set()
    for sn in SUFFIX_PAIRS.values():
        for b, i in sn:
            morph_words.add(b); morph_words.add(i)
    for m_dn, m_dz, m_db in morph:
        found = []
        tries = 0
        while len(found) < 5 and tries < 4000:
            w1, w2 = rng.choice(pool, 2, replace=False)
            if abs((zipfs[w2] - zipfs[w1]) - m_dz) < 0.15:
                found.append((norms[w2] - norms[w1],
                              float((unit(w2) - unit(w1)) @ bal)))
            tries += 1
        if found:
            matched += 1
            pseudo_dn.append(np.mean([f[0] for f in found]))
            pseudo_db.append(np.mean([f[1] for f in found]))
    pseudo_dn = np.array(pseudo_dn); pseudo_db = np.array(pseudo_db)
    frac_norm = float(pseudo_dn.mean() / dn.mean()) if abs(dn.mean()) > 1e-9 else float("nan")
    frac_bal = float(pseudo_db.mean() / db.mean()) if abs(db.mean()) > 1e-9 else float("nan")

    # LEG C: reversed pairs
    rev = [dpair(b, i) for b, i in reversed_pairs]
    rev_dn = np.array([r[0] for r in rev]); rev_db = np.array([r[2] for r in rev])

    print(f"\n--- Layer {L} ---")
    print(f"  A0 word-level: corr(zipf, norm) = {r_freq_norm:+.3f}; "
          f"corr(zipf, BAL-proj) = {r_freq_bal:+.3f}")
    print(f"  A  inflectional pairs (n={len(dn)}): mean Δnorm = {dn.mean():+.1f}, "
          f"mean Δzipf = {dz.mean():+.2f}")
    print(f"     Δnorm ~ Δzipf: slope = {slope:+.1f}/zipf, "
          f"intercept = {intercept:+.1f} [95% CI {ci[0]:+.1f}, {ci[1]:+.1f}], R² = {r2:.2f}")
    print(f"  B  Δzipf-matched pseudo-pairs ({matched}/{len(morph)} matched, 5 each): "
          f"mean Δnorm = {pseudo_dn.mean():+.1f} "
          f"({100*frac_norm:.0f}% of morphological)")
    print(f"     ΔBAL-proj: morph = {db.mean():+.4f}, pseudo = {pseudo_db.mean():+.4f} "
          f"({100*frac_bal:.0f}%)")
    print(f"  C  reversed-freq pairs (n={len(rev_dn)}): mean Δnorm = {rev_dn.mean():+.1f} "
          f"({np.sum(rev_dn < 0)}/{len(rev_dn)} negative); "
          f"mean ΔBAL-proj = {rev_db.mean():+.4f}")
    verdicts[L] = (frac_norm, ci, dn.mean(), pseudo_dn.mean(), rev_dn.mean())

print("\n" + "=" * 78)
print("VERDICT vs pre-registered decision rule")
print("=" * 78)
fracs = [verdicts[L][0] for L in LAYERS]
intercept_zero = [verdicts[L][1][0] <= 0 <= verdicts[L][1][1] for L in LAYERS]
print(f"  pseudo-pair fraction of morph Δnorm per layer: " +
      ", ".join(f"L{L}:{100*f:.0f}%" for L, f in zip(LAYERS, fracs)))
print(f"  intercept 95% CI includes 0: " +
      ", ".join(f"L{L}:{v}" for L, v in zip(LAYERS, intercept_zero)))
mf = float(np.mean(fracs))
if mf >= 0.70 and all(intercept_zero):
    print("  -> F1: FREQUENCY. The displacement is a frequency-norm coupling;")
    print("     paper says 'frequency/markedness' with the Greenberg note.")
    print("     The concept-physiology coupling result is unaffected.")
elif mf <= 0.30 and not any(intercept_zero):
    print("  -> F2: MARKEDNESS. Displacement is morphological beyond frequency.")
else:
    print("  -> F3: mixed — report both components with magnitudes.")
