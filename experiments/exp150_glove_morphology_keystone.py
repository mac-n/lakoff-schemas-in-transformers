"""
exp150_glove_morphology_keystone.py — KEYSTONE: morphology-on-Lakoff in GloVe.

Replicates the exp138 protocol (suffix pair-difference directions x Lakoff
schema directions, cosine matrix) in static GloVe embeddings
(glove-wiki-gigaword-300), and compares against the Pythia 410M exp138 result
(parsed from results_exp138.txt on disk — no transcription).

Why this is load-bearing (paper Claim 3):
- If GloVe shows the same morphology-schema mappings -> the "transformers
  reconstruct embodied organisation" claim collapses; paper restructures
  around within-transformer findings.
- If GloVe doesn't -> Claim 3 lands and the paper has its punchline.

============================================================================
PRE-REGISTRATION (written 2026-06-10, BEFORE running)
============================================================================
Competing hypotheses on the table:

  H_paper (Claim 3): morphology-on-Lakoff positioning is transformer-
    specific. Predicts: GloVe suffix x schema matrix does NOT correlate
    with Pythia's beyond the permutation null; signatures absent.

  H_functionalization (pidgin/grammar doc, point 10): static substrates
    store metaphor "as fact" (static alignment), deep substrates "as
    function". Predicts: GloVe shows comparable or STRONGER static
    alignment than Pythia's within-layer matrix.

Primary metric: Pearson r between the GloVe 7x8 matrix and Pythia L12
  matrix (flattened), against a row+column label-permutation null
  (10,000 perms). Significant = r > 95th percentile of null.
  Also reported vs L4 and L20 (early/late checks).

Secondary signature checks (from exp138 / v5 readings):
  S1. BALANCE universal sink: all 7 suffixes negative on BALANCE.
  S2. UN_negation has the largest row signature (L2 norm across schemas).
  S3. LIGHT-DARK column null: mean |cos| < 0.10.
  (ER-vs-EST layer decay is untestable in a static space by definition —
  flagged as an inherently transformer-only signature either way.)

Committed point prediction (this Claude, before running): a PARTIAL
  outcome — the BALANCE-negative markedness sink partially reproduces in
  GloVe (markedness is close to a frequency/distributional regularity),
  but the suffix-SPECIFIC schema mappings do not, and the matrix
  correlation is driven mostly by the BALANCE column. Concretely:
  r(GloVe, PythiaL12) in the 0.2-0.5 range, dropping below the null
  threshold when the BALANCE column is excluded.

Decision rule:
  A. r significant AND S1-S3 reproduce -> Claim 3 collapses; restructure
     (functionalization framing becomes the fallback story).
  B. r not significant AND signatures absent -> Claim 3 lands as stated.
  C. r driven by BALANCE only (significant with BALANCE, not without)
     -> Claim 3 restructures: "suffix-specific schema positioning is
     transformer-specific; the markedness sink is distributional."
============================================================================

Method parity with exp138:
- Same SUFFIX_PAIRS, same 8 schemas, same MML vocabulary.
- Suffix directions: mean over pairs of (unit(inflected) - unit(base)),
  renormalised — identical to exp138's build_suffix_direction.
- Schema directions: mean(pos) - mean(neg) over unique pole words.
- Strip analogue: GloVe has no per-layer anisotropy, but it has a global
  mean offset and a documented frequency axis (Mu et al. "All-but-the-Top").
  We strip (1) the normalised mean of the full collected word set —
  the static analogue of the per-layer anisotropy direction — and
  (2) the COMMON-RARE frequency axis, Gram-Schmidt orthogonalised,
  exactly mirroring exp138's strip_aniso_freq. Matrices are reported
  BOTH raw and stripped (stripped is the headline, for parity).

Optional: --w2v also runs word2vec-google-news-300 (~1.6GB download).
"""

import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gensim.downloader as api

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML

RUN_W2V = "--w2v" in sys.argv
# fastText caveat: subword n-gram architecture means suffix directions share
# components BY CONSTRUCTION ("walking"/"running" literally contain `ing`
# n-grams). It cannot serve as a clean distributional control for morphology —
# supplementary only: tests whether architecturally-coherent suffix directions
# still fail to position on schema axes.
RUN_FASTTEXT = "--fasttext" in sys.argv

# ============================================================================
# Same lists as exp138 (copied verbatim)
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
PYTHIA_LAYERS = [4, 8, 12, 16, 20]


# ============================================================================
# Parse Pythia 410M exp138 matrices from results_exp138.txt
# ============================================================================
def parse_exp138_results(path="results_exp138.txt"):
    matrices = {}
    with open(path) as f:
        lines = f.readlines()
    current_layer = None
    for line in lines:
        m = re.match(r"Layer (\d+) ", line)
        if m and "suffix" in line and "schema" in line:
            current_layer = int(m.group(1))
            matrices[current_layer] = {}
            continue
        if current_layer is not None:
            toks = line.split()
            if toks and toks[0] in SUFFIX_PAIRS:
                vals = [float(t) for t in toks[1:1 + len(SCHEMA_NAMES)]]
                matrices[current_layer][toks[0]] = vals
            elif line.strip() == "" and len(matrices[current_layer]) == len(SUFFIX_PAIRS):
                current_layer = None
    out = {}
    for L, rows in matrices.items():
        if len(rows) != len(SUFFIX_PAIRS):
            raise ValueError(f"Layer {L}: parsed {len(rows)} suffix rows, expected {len(SUFFIX_PAIRS)}")
        out[L] = np.array([rows[s] for s in SUFFIX_ORDER])
    for L in PYTHIA_LAYERS:
        if L not in out:
            raise ValueError(f"Layer {L} missing from {path}")
    return out


print("Parsing Pythia 410M exp138 matrices from results_exp138.txt...")
pythia = parse_exp138_results()
# sanity anchor against known value (L12 ER x BALANCE = -0.377)
anchor = pythia[12][SUFFIX_ORDER.index("ER_comparative"), SCHEMA_NAMES.index("BALANCE")]
assert abs(anchor - (-0.377)) < 1e-6, f"parse sanity check failed: {anchor}"
print(f"  ok (anchor L12 ER x BALANCE = {anchor:+.3f})")


# ============================================================================
# Static-embedding harness (works for GloVe and word2vec)
# ============================================================================
def collect_words():
    words = set(COMMON + RARE)
    for pairs in SUFFIX_PAIRS.values():
        for b, i in pairs:
            words.add(b); words.add(i)
    for sn in SCHEMA_NAMES:
        for p, n in LAKOFF_SCHEMAS_MML[sn]:
            words.add(p); words.add(n)
    return sorted(words)


def run_static_substrate(wv, name):
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    all_words = collect_words()
    missing = [w for w in all_words if w not in wv.key_to_index]
    if missing:
        print(f"  MISSING from vocab ({len(missing)}): {missing}")
    vecs = {w: np.asarray(wv[w], dtype=np.float64)
            for w in all_words if w in wv.key_to_index}

    # Strip axes: global-mean (static anisotropy analogue) + freq
    mean_vec = np.mean([vecs[w] for w in vecs], axis=0)
    aniso = mean_vec / np.linalg.norm(mean_vec)
    freq_raw = (np.mean([vecs[w] for w in COMMON if w in vecs], axis=0)
                - np.mean([vecs[w] for w in RARE if w in vecs], axis=0))
    freq = freq_raw / np.linalg.norm(freq_raw)
    freq_orth = freq - (freq @ aniso) * aniso
    freq_orth = freq_orth / np.linalg.norm(freq_orth)

    def strip(d):
        d = d - (d @ aniso) * aniso
        d = d - (d @ freq_orth) * freq_orth
        return d / np.linalg.norm(d)

    def schema_dir(sn, stripped):
        pairs = LAKOFF_SCHEMAS_MML[sn]
        pos = sorted(set(p[0] for p in pairs) & set(vecs))
        neg = sorted(set(p[1] for p in pairs) & set(vecs))
        raw = np.mean([vecs[w] for w in pos], axis=0) - np.mean([vecs[w] for w in neg], axis=0)
        raw = raw / np.linalg.norm(raw)
        return strip(raw) if stripped else raw

    def suffix_dir(sn, stripped):
        diffs = []
        for base, infl in SUFFIX_PAIRS[sn]:
            if base not in vecs or infl not in vecs:
                continue
            b = vecs[base]; i = vecs[infl]
            diffs.append(i / np.linalg.norm(i) - b / np.linalg.norm(b))
        raw = np.mean(diffs, axis=0)
        raw = raw / np.linalg.norm(raw)
        return strip(raw) if stripped else raw

    mats = {}
    for stripped, label in [(False, "raw"), (True, "stripped")]:
        sdirs = {sn: schema_dir(sn, stripped) for sn in SCHEMA_NAMES}
        fdirs = {sn: suffix_dir(sn, stripped) for sn in SUFFIX_ORDER}
        M = np.zeros((len(SUFFIX_ORDER), len(SCHEMA_NAMES)))
        for i, suf in enumerate(SUFFIX_ORDER):
            for j, sch in enumerate(SCHEMA_NAMES):
                M[i, j] = float(fdirs[suf] @ sdirs[sch])
        mats[label] = M
        # suffix-suffix coherence for reference (exp138 reported this)
        fd = np.array([fdirs[s] for s in SUFFIX_ORDER])
        cos_mat = fd @ fd.T
        off = ~np.eye(len(SUFFIX_ORDER), dtype=bool)
        print(f"\n  [{label}] mean off-diag suffix-suffix cos = {cos_mat[off].mean():+.3f}")
        print(f"  [{label}] suffix x schema cosines:")
        print(f"    {'suffix':<18}  " + "  ".join(f"{s[:9]:>9}" for s in SCHEMA_NAMES))
        for i, suf in enumerate(SUFFIX_ORDER):
            row = f"    {suf:<18}  "
            for j in range(len(SCHEMA_NAMES)):
                row += f"{M[i, j]:>+9.3f}  "
            print(row)
    return mats


# ============================================================================
# Comparison statistics
# ============================================================================
def perm_null_r(A, B, n_perm=10000, seed=0):
    """Pearson r between flattened A and B, with a row+column
    label-permutation null applied to A."""
    rng = np.random.default_rng(seed)
    def pearson(x, y):
        x = x - x.mean(); y = y - y.mean()
        return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    r_obs = pearson(A.flatten(), B.flatten())
    null = np.empty(n_perm)
    for k in range(n_perm):
        P = A[rng.permutation(A.shape[0])][:, rng.permutation(A.shape[1])]
        null[k] = pearson(P.flatten(), B.flatten())
    p = float((np.sum(null >= r_obs) + 1) / (n_perm + 1))
    return r_obs, float(np.percentile(null, 95)), p


def signature_checks(M, label):
    bal = SCHEMA_NAMES.index("BALANCE")
    ld = SCHEMA_NAMES.index("LIGHT-DARK")
    un = SUFFIX_ORDER.index("UN_negation")
    s1 = bool(np.all(M[:, bal] < 0))
    row_norms = np.linalg.norm(M, axis=1)
    s2 = bool(np.argmax(row_norms) == un)
    s3_val = float(np.mean(np.abs(M[:, ld])))
    s3 = bool(s3_val < 0.10)
    print(f"\n  Signature checks [{label}]:")
    print(f"    S1 BALANCE universal sink (all 7 negative): {s1}  "
          f"(column: {np.array2string(M[:, bal], precision=2)})")
    print(f"    S2 UN_negation largest row norm: {s2}  "
          f"(row norms: " + ", ".join(f"{s.split('_')[0]}={n:.2f}" for s, n in zip(SUFFIX_ORDER, row_norms)) + ")")
    print(f"    S3 LIGHT-DARK null (mean |cos| < 0.10): {s3}  (mean |cos| = {s3_val:.3f})")
    return s1, s2, s3


def compare_to_pythia(M, substrate_label):
    print(f"\n  --- {substrate_label} vs Pythia 410M (exp138) ---")
    bal = SCHEMA_NAMES.index("BALANCE")
    keep = [j for j in range(len(SCHEMA_NAMES)) if j != bal]
    results = {}
    for L in PYTHIA_LAYERS:
        r, thr95, p = perm_null_r(M, pythia[L])
        r_nb, thr95_nb, p_nb = perm_null_r(M[:, keep], pythia[L][:, keep])
        sig = "SIG" if r > thr95 else "ns "
        sig_nb = "SIG" if r_nb > thr95_nb else "ns "
        print(f"    vs L{L:>2}: r = {r:+.3f} (null95 {thr95:+.3f}, p={p:.4f}) {sig}   "
              f"| BALANCE excluded: r = {r_nb:+.3f} (null95 {thr95_nb:+.3f}, p={p_nb:.4f}) {sig_nb}")
        results[L] = (r, p, r_nb, p_nb)
    return results


# ============================================================================
# Run
# ============================================================================
print("\nLoading GloVe (glove-wiki-gigaword-300)...")
glove = api.load("glove-wiki-gigaword-300")
print(f"  loaded: {len(glove.key_to_index)} words")

glove_mats = run_static_substrate(glove, "GloVe wiki-gigaword 300d")
sigs = signature_checks(glove_mats["stripped"], "GloVe stripped")
glove_cmp = compare_to_pythia(glove_mats["stripped"], "GloVe (stripped)")
print("\n  (raw, for reference:)")
_ = compare_to_pythia(glove_mats["raw"], "GloVe (raw)")

w2v_mats = None
if RUN_W2V:
    print("\nLoading word2vec (word2vec-google-news-300)...")
    w2v = api.load("word2vec-google-news-300")
    w2v_mats = run_static_substrate(w2v, "word2vec GoogleNews 300d")
    signature_checks(w2v_mats["stripped"], "word2vec stripped")
    compare_to_pythia(w2v_mats["stripped"], "word2vec (stripped)")

ft_mats = None
if RUN_FASTTEXT:
    print("\nLoading fastText (fasttext-wiki-news-subwords-300)...")
    print("  CAVEAT: subword architecture — suffix coherence is built in;")
    print("  supplementary evidence only, not a clean distributional control.")
    ft = api.load("fasttext-wiki-news-subwords-300")
    ft_mats = run_static_substrate(ft, "fastText wiki-news subwords 300d")
    signature_checks(ft_mats["stripped"], "fastText stripped")
    compare_to_pythia(ft_mats["stripped"], "fastText (stripped)")


# ============================================================================
# Verdict against pre-registered decision rule
# ============================================================================
print("\n" + "=" * 78)
print("VERDICT vs pre-registered decision rule (see docstring)")
print("=" * 78)
r12, p12, r12_nb, p12_nb = glove_cmp[12]
sig_full = p12 < 0.05
sig_nobal = p12_nb < 0.05
s1, s2, s3 = sigs
print(f"  Primary (vs Pythia L12): r = {r12:+.3f} (p={p12:.4f}), "
      f"BALANCE-excluded r = {r12_nb:+.3f} (p={p12_nb:.4f})")
print(f"  Signatures: S1 sink={s1}, S2 UN-largest={s2}, S3 LD-null={s3}")
if sig_full and sig_nobal and s1 and s2:
    print("  -> Outcome A: GloVe reproduces the pattern. Claim 3 COLLAPSES as")
    print("     stated; restructure (functionalization framing is the fallback).")
elif sig_full and not sig_nobal:
    print("  -> Outcome C: correlation is BALANCE-driven. Claim 3 restructures:")
    print("     suffix-specific schema positioning is transformer-specific;")
    print("     the markedness sink is distributional.")
elif not sig_full:
    print("  -> Outcome B: no significant correspondence. Claim 3 LANDS as stated.")
else:
    print("  -> Mixed outcome not covered by pre-registration — interpret")
    print("     cautiously, do NOT retrofit a story before a follow-up control.")

# ============================================================================
# Plot: GloVe (raw + stripped) beside Pythia L4/L12/L20
# ============================================================================
panels = [("GloVe raw", glove_mats["raw"]), ("GloVe stripped", glove_mats["stripped"])]
if w2v_mats is not None:
    panels.append(("word2vec stripped", w2v_mats["stripped"]))
if ft_mats is not None:
    panels.append(("fastText stripped (subword caveat)", ft_mats["stripped"]))
panels += [(f"Pythia 410M L{L}", pythia[L]) for L in [4, 12, 20]]

fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 5), sharey=True)
for ax, (title, M) in zip(axes, panels):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(SCHEMA_NAMES)))
    ax.set_xticklabels([s[:8] for s in SCHEMA_NAMES], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(SUFFIX_ORDER)))
    ax.set_yticklabels(SUFFIX_ORDER, fontsize=8)
    for i in range(len(SUFFIX_ORDER)):
        for j in range(len(SCHEMA_NAMES)):
            v = M[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.25 else "black", fontsize=7)
    ax.set_title(title, fontsize=11)
plt.colorbar(im, ax=axes[-1], fraction=0.04)
fig.suptitle("exp150 KEYSTONE — suffix × Lakoff schema cosines: static GloVe vs Pythia 410M (exp138)",
             fontsize=12)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp150_glove_vs_pythia.png", dpi=120)
print("\nSaved exp150_glove_vs_pythia.png")
