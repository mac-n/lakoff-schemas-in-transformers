"""
exp153_embedding_layer_emergence.py — WHERE in depth does Pythia's
inflectional schema-positioning (the BALANCE markedness sink) emerge?

Context: exp150/150b/150c established that Pythia 410M's inflectional
suffix x schema geometry (ER x BALANCE = -0.38, universal-ish BALANCE sink)
exists in NO static space tested (GloVe, word2vec, fastText all ~0.00),
while the three static spaces share a consistent inflectional geometry
among themselves (r ~ 0.71-0.77) that only weakly resembles Pythia's
(r ~ 0.20-0.36). The transformer REORGANISES inflectional geometry.

Question: is that reorganisation (a) already in Pythia's learned embedding
matrix (= trained-into-the-static-embedding by transformer training), or
(b) computed across early layers (= contextual reorganisation in depth)?

Probe: exp138 protocol at hook_embed, L0, L1, L2, plus L4/L8/L12 as
continuity anchors (L4/L12 should reproduce exp138's parsed values).

PRE-REGISTRATION (2026-06-10, before running):
  Tracked quantities per probe point:
    - ER x BALANCE (the marquee number)
    - inflectional-rows (5x8) Pearson r vs GloVe-stripped (exp150)
    - inflectional-rows (5x8) Pearson r vs Pythia L12 (exp138)
  Outcomes:
    E1 (embedding already Pythia-like): sink full-strength at hook_embed.
       Reading: transformer TRAINING writes schema-anchored inflectional
       geometry into its static embedding — static-vs-deep contrast is
       about training regime, not about contextual computation.
    E2 (embedding GloVe-like, sink builds over L0-L4): inflectional
       geometry starts distributional-like and is computed into schema
       positioning in depth. Reading: contextual reorganisation;
       strongest version of "reconstructed in the network".
    E3 (embedding like NEITHER — low r to both): tokenization fragments
       dominate at the embedding layer; only the mean-pooled variant is
       interpretable there; rely on it and say so.
  Committed point prediction (this Claude): E2-leaning-E3 — last-token
  embedding-layer geometry will be tokenizer-dominated (E3 symptoms on
  the last-token variant), the mean-pooled variant will look weakly
  GloVe-like (r_GloVe > r_PythiaL12), and the sink will reach its
  exp138 plateau by ~L4 (it is already -0.47 at L4 for ED in exp138).

Tokenization caveat handled explicitly: at hook_embed, last-token
residuals of multi-token words carry no stem info. We run BOTH:
  - last-token (exp138 parity, comparable to L4+ numbers)
  - mean-pooled over word tokens (excluding BOS) at hook_embed/L0/L1/L2
and report the multi-token fraction of the vocabulary.
"""

import re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML

# ============================================================================
# Same lists as exp138/exp150
# ============================================================================
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
INFL = [0, 1, 2, 3, 4]   # ER, EST, ING, ED, S


# ============================================================================
# Reference matrices: Pythia L12 (exp138) and GloVe stripped (exp150)
# ============================================================================
def parse_pythia(path="results_exp138.txt"):
    mats, cur = {}, None
    for line in open(path):
        m = re.match(r"Layer (\d+) ", line)
        if m and "suffix" in line:
            cur = int(m.group(1)); mats[cur] = {}
        elif cur is not None:
            t = line.split()
            if t and t[0] in SUFFIX_PAIRS:
                mats[cur][t[0]] = [float(x) for x in t[1:9]]
    return {k: np.array([v[s] for s in SUFFIX_ORDER]) for k, v in mats.items() if len(v) == 7}


def parse_last_stripped(path):
    rows, cur = None, None
    for line in open(path, errors="ignore"):
        if "[stripped] suffix x schema" in line:
            cur = {}
        elif cur is not None:
            t = line.split()
            if t and t[0] in SUFFIX_PAIRS:
                cur[t[0]] = [float(x) for x in t[1:9]]
                if len(cur) == 7:
                    rows, cur = cur, None
    return np.array([rows[s] for s in SUFFIX_ORDER])


PYTHIA_REF = parse_pythia()
GLOVE_REF = parse_last_stripped("exp150_output.txt")


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y)))


# ============================================================================
# Collect residuals at early probe points
# ============================================================================
device = "mps"
print("Loading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
model.eval()

PROBE_LAYERS = [0, 1, 2, 4, 8, 12]
hook_names = ["hook_embed"] + [f"blocks.{L}.hook_resid_post" for L in PROBE_LAYERS]
PROBES = ["embed"] + PROBE_LAYERS  # display order

all_words = set(COMMON + RARE)
for pairs in SUFFIX_PAIRS.values():
    for b, i in pairs:
        all_words.add(b); all_words.add(i)
for sn in SCHEMA_NAMES:
    for p, n in LAKOFF_SCHEMAS_MML[sn]:
        all_words.add(p); all_words.add(n)
all_words = sorted(all_words)

print(f"Collecting residuals for {len(all_words)} words at {PROBES} "
      f"(last-token + mean-pooled)...")

# residuals[variant][word][probe] -> vector
residuals = {"last": {}, "mean": {}}
n_tokens = {}
for k, w in enumerate(all_words):
    toks = model.to_tokens(w)            # NOTE: Pythia config does NOT
    n_tokens[w] = toks.shape[1]          # prepend BOS (verified empirically:
    with torch.no_grad():                # to_tokens("big").shape == [1,1])
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    for probe, hname in zip(PROBES, hook_names):
        acts = cache[hname][0]           # [seq, d]
        last = acts[-1, :].cpu().numpy()
        mean = acts.mean(dim=0).cpu().numpy()   # all positions (no BOS present)
        residuals["last"].setdefault(w, {})[probe] = last
        residuals["mean"].setdefault(w, {})[probe] = mean
    if (k + 1) % 100 == 0:
        print(f"  {k+1}/{len(all_words)}")

multi = [w for w in all_words if n_tokens[w] > 1]
print(f"\nMulti-token words: {len(multi)}/{len(all_words)} "
      f"({100*len(multi)/len(all_words):.0f}%)")
print(f"  examples: {[(w, n_tokens[w]) for w in multi[:8]]}")


# ============================================================================
# exp138 protocol per (variant, probe)
# ============================================================================
def build_matrix(variant, probe):
    R = residuals[variant]
    arr = np.stack([R[w][probe] for w in all_words], axis=0)
    m = arr.mean(axis=0)
    aniso = m / np.linalg.norm(m)

    def mean_acts(words):
        return np.mean([R[w][probe] for w in words], axis=0)

    freq_raw = mean_acts(COMMON) - mean_acts(RARE)
    freq = freq_raw / np.linalg.norm(freq_raw)
    freq_orth = freq - (freq @ aniso) * aniso
    freq_orth = freq_orth / np.linalg.norm(freq_orth)

    def strip(d):
        d = d - (d @ aniso) * aniso
        d = d - (d @ freq_orth) * freq_orth
        return d / np.linalg.norm(d)

    def schema_dir(sn):
        pairs = LAKOFF_SCHEMAS_MML[sn]
        pos = sorted(set(p[0] for p in pairs))
        neg = sorted(set(p[1] for p in pairs))
        raw = mean_acts(pos) - mean_acts(neg)
        return strip(raw / np.linalg.norm(raw))

    def suffix_dir(sn):
        diffs = []
        for base, infl in SUFFIX_PAIRS[sn]:
            b = R[base][probe]; i = R[infl][probe]
            diffs.append(i / np.linalg.norm(i) - b / np.linalg.norm(b))
        raw = np.mean(diffs, axis=0)
        return strip(raw / np.linalg.norm(raw))

    sdirs = {sn: schema_dir(sn) for sn in SCHEMA_NAMES}
    fdirs = {sn: suffix_dir(sn) for sn in SUFFIX_ORDER}
    M = np.zeros((len(SUFFIX_ORDER), len(SCHEMA_NAMES)))
    for i, suf in enumerate(SUFFIX_ORDER):
        for j, sch in enumerate(SCHEMA_NAMES):
            M[i, j] = float(fdirs[suf] @ sdirs[sch])
    return M


print("\n" + "=" * 78)
print("Emergence trajectory — per probe point (stripped, exp138 protocol)")
print("=" * 78)
bal = SCHEMA_NAMES.index("BALANCE")
results = {}
for variant in ["last", "mean"]:
    print(f"\n--- variant: {variant}-token ---")
    print(f"  {'probe':>6}  {'ERxBAL':>7}  {'EDxBAL':>7}  {'infl r_GloVe':>12}  "
          f"{'infl r_PythiaL12':>16}  {'full r_PythiaL12':>16}")
    for probe in PROBES:
        M = build_matrix(variant, probe)
        results[(variant, probe)] = M
        r_glove = pearson(M[INFL].flatten(), GLOVE_REF[INFL].flatten())
        r_p12 = pearson(M[INFL].flatten(), PYTHIA_REF[12][INFL].flatten())
        r_full = pearson(M.flatten(), PYTHIA_REF[12].flatten())
        results[(variant, probe, "stats")] = (r_glove, r_p12, r_full)
        print(f"  {str(probe):>6}  {M[0, bal]:>+7.3f}  {M[3, bal]:>+7.3f}  "
              f"{r_glove:>+12.3f}  {r_p12:>+16.3f}  {r_full:>+16.3f}")

# Parity check: L4/L12 last-token should match exp138's parsed numbers
print("\nParity check vs exp138 (last-token):")
for L in [4, 12]:
    diff = float(np.abs(results[("last", L)] - PYTHIA_REF[L]).max())
    print(f"  L{L}: max |Δ| vs results_exp138.txt = {diff:.3f} "
          f"{'(ok)' if diff < 0.05 else '(MISMATCH — investigate before trusting)'}")

# Reference values
print(f"\nReference: GloVe ERxBAL = {GLOVE_REF[0, bal]:+.3f}, "
      f"Pythia L12 ERxBAL = {PYTHIA_REF[12][0, bal]:+.3f}")
print(f"Reference: r(GloVe infl, Pythia L12 infl) = "
      f"{pearson(GLOVE_REF[INFL].flatten(), PYTHIA_REF[12][INFL].flatten()):+.3f}")


# ============================================================================
# Plot the emergence trajectory
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
xticks = list(range(len(PROBES)))
for ax, variant in zip(axes, ["last", "mean"]):
    er = [results[(variant, p)][0, bal] for p in PROBES]
    ed = [results[(variant, p)][3, bal] for p in PROBES]
    rg = [results[(variant, p, "stats")][0] for p in PROBES]
    rp = [results[(variant, p, "stats")][1] for p in PROBES]
    ax.plot(xticks, er, "o-", label="ER × BALANCE")
    ax.plot(xticks, ed, "s-", label="ED × BALANCE")
    ax.plot(xticks, rg, "^--", label="infl rows: r vs GloVe")
    ax.plot(xticks, rp, "v--", label="infl rows: r vs Pythia L12")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axhline(-0.377, color="red", lw=0.5, ls=":", label="exp138 ER×BAL plateau")
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(p) for p in PROBES])
    ax.set_xlabel("probe point")
    ax.set_title(f"{variant}-token variant")
    ax.legend(fontsize=8)
fig.suptitle("exp153 — emergence of inflectional schema-positioning across early depth\n"
             "(does the BALANCE sink exist in the embedding, or get computed?)")
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp153_emergence.png", dpi=120)
print("\nSaved exp153_emergence.png")
