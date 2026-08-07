"""
exp154c_heldout_dnorm.py — held-out d_norm control for exp154's collapse.

FOLLOW-UP CONTROL to exp154 (designed 2026-06-11, after seeing exp154's
result — flagged as such per the post-hoc labelling convention — but with
its own pre-registration committed before THIS script was run).

Circularity risk in exp154: d_norm/d_disp were estimated from all 489
words, INCLUDING every suffix-pair word and the BALANCE vocabulary they
are then used to strip. With 489 words in 1024-dim space, the covariance
direction could partly absorb the test items themselves, manufacturing
the collapse.

Design (surgical change from exp154 — everything else identical):
  - est_words: all words EXCLUDING suffix-pair words (base + inflected)
    and BALANCE vocabulary. d_norm/d_disp estimated on est_words only
    (z-stats from est_words).
  - aniso/freq strip unchanged (full word set, exp138 protocol parity).
  - Report: cos(d_norm_heldout, d_norm_full) per layer; in-sample AND
    out-of-sample carrier correlations (does projection onto held-out
    d_norm predict z-norm of the EXCLUDED test words?); sink before ->
    after held-out strip; random-2D-strip band for parity.
  - Parity check: sink_before per layer is PARSED from exp154_output.txt
    on disk (never transcribed) and must match to 3 decimals.

Decision-rule fix vs exp154: exp154's N1 'above_band' check was
mis-specified (it required the stripped sink to be MORE negative than a
band centred on the UNSTRIPPED value — unreachable). Here 'survives'
means the stripped sink stays within the random-strip band (which sits
at the unstripped value); 'collapses' means |retained| small and the
stripped value near 0, far above the band.

PRE-REGISTRATION (2026-06-11, before running; this Claude):
  Committed prediction: the collapse is NOT an overfit artifact. The
  norm direction is global lexicon geometry (carrier sanity was
  0.83-0.94; corr(zipf, norm) ~ +0.71 is lexicon-wide), so removing the
  test words from its estimation should barely move it.
  Point predictions:
    - cos(d_norm_heldout, d_norm_full) >= +0.90 at every layer
    - out-of-sample carrier corr >= +0.75 at L8+
    - inflectional mean sink retained under held-out strip: |x| <= 15%
      at every layer
  Decision rule:
    H1 |mean retained| <= 25%: collapse holds out-of-sample. Circularity
       dismissed; exp154's N2 verdict stands as written.
    H2 mean retained >= 60% (sink survives held-out strip, stays at/near
       the random band): exp154's collapse was substantially overfit.
       The grounding claim must be weakened and Claim 3 re-examined
       before exp152.
    H3 intermediate (25-60%): partial overfit; report both, prefer
       held-out numbers in the paper.
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

# ---- held-out split ----
test_words = set()
for pairs in SUFFIX_PAIRS.values():
    for b, i in pairs:
        test_words.add(b); test_words.add(i)
for p, n in LAKOFF_SCHEMAS_MML["BALANCE"]:
    test_words.add(p); test_words.add(n)
test_words &= set(all_words)
est_words = [w for w in all_words if w not in test_words]
test_words = sorted(test_words)
print(f"Word split: {len(all_words)} total = {len(est_words)} estimation "
      f"+ {len(test_words)} held-out test (suffix pairs + BALANCE vocab)")

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


def corrf(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y)))


# ---- parse exp154 reference values from disk (parity check) ----
ref_before = {}
with open("/Users/macn/Documents/embeddingexp/exp154_output.txt") as f:
    cur = None
    for line in f:
        m = re.match(r"--- Layer (\d+) ---", line.strip())
        if m:
            cur = int(m.group(1)); continue
        m = re.match(r"inflectional mean sink:\s*([+-]\d+\.\d+)\s*->", line.strip())
        if m and cur is not None:
            ref_before[cur] = float(m.group(1))
print(f"exp154 reference (parsed): {ref_before}")
assert set(ref_before) == set(LAYERS), "failed to parse exp154 reference values"

print("\n" + "=" * 78)
print("exp154c — held-out d_norm control (circularity check on exp154)")
print("=" * 78)

summary = {}
rng_global = np.random.default_rng(7)

for L in LAYERS:
    # ---- strip machinery (exp138 protocol, FULL set — unchanged) ----
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

    # ---- d_norm / d_disp from HELD-OUT estimation set only ----
    def build_norm_dirs(words):
        nv = np.array([norms[w] for w in words])
        zz = {w: (norms[w] - nv.mean()) / nv.std() for w in words}
        dn = np.sum([zz[w] * units[w] for w in words], axis=0)
        dn = strip(dn / np.linalg.norm(dn))
        az = {w: abs(zz[w]) for w in words}
        am = np.mean(list(az.values())); asd = np.std(list(az.values()))
        dd = np.sum([((az[w] - am) / asd) * units[w] for w in words], axis=0)
        dd = strip(dd / np.linalg.norm(dd))
        return dn, dd, nv.mean(), nv.std()

    d_norm_ho, d_disp_ho, est_mean, est_std = build_norm_dirs(est_words)
    d_norm_full, d_disp_full, _, _ = build_norm_dirs(all_words)
    d_disp_ho_o = proj_out(d_disp_ho, d_norm_ho)

    cos_dirs = float(d_norm_ho @ d_norm_full)
    cos_disp_dirs = float(d_disp_ho @ d_disp_full)

    # carrier: in-sample (est words) and OUT-OF-SAMPLE (held-out test words)
    z_est = [(norms[w] - est_mean) / est_std for w in est_words]
    r_in = corrf([float(units[w] @ d_norm_ho) for w in est_words], z_est)
    z_test = [(norms[w] - est_mean) / est_std for w in test_words]
    r_out = corrf([float(units[w] @ d_norm_ho) for w in test_words], z_test)
    r_out_disp = corrf([float(units[w] @ d_disp_ho) for w in test_words],
                       [abs(z) for z in z_test])

    # ---- BALANCE + suffix directions (identical to exp154/exp138) ----
    def schema_dir(sn):
        pairs = LAKOFF_SCHEMAS_MML[sn]
        pos = sorted(set(p[0] for p in pairs)); neg = sorted(set(p[1] for p in pairs))
        raw = mean_acts(pos) - mean_acts(neg)
        return strip(raw / np.linalg.norm(raw))

    bal = schema_dir("BALANCE")
    cos_bn = float(bal @ d_norm_ho)

    def suffix_dir(sn):
        diffs = []
        for base, infl in SUFFIX_PAIRS[sn]:
            b = residuals[base][L]; i = residuals[infl][L]
            diffs.append(i / np.linalg.norm(i) - b / np.linalg.norm(b))
        raw = np.mean(diffs, axis=0)
        return strip(raw / np.linalg.norm(raw))

    sufs = {sn: suffix_dir(sn) for sn in SUFFIX_ORDER}
    sink_before = {sn: float(sufs[sn] @ bal) for sn in SUFFIX_ORDER}

    bal_ns = proj_out(proj_out(bal, d_norm_ho), d_disp_ho_o)
    sink_after = {sn: float(proj_out(proj_out(sufs[sn], d_norm_ho), d_disp_ho_o) @ bal_ns)
                  for sn in SUFFIX_ORDER}

    # ---- random-2D-strip band (parity with exp154) ----
    rand_means = []
    for _ in range(20):
        g1 = strip(rng_global.standard_normal(arr.shape[1]))
        g2 = proj_out(strip(rng_global.standard_normal(arr.shape[1])), g1)
        bal_r = proj_out(proj_out(bal, g1), g2)
        vals = [float(proj_out(proj_out(sufs[sn], g1), g2) @ bal_r) for sn in INFL]
        rand_means.append(np.mean(vals))
    rand_means = np.array(rand_means)

    infl_before = np.mean([sink_before[s] for s in INFL])
    infl_after = np.mean([sink_after[s] for s in INFL])

    # ---- report ----
    print(f"\n--- Layer {L} ---")
    parity = abs(infl_before - ref_before[L])
    print(f"  parity vs exp154 (inflectional mean before): "
          f"{infl_before:+.3f} vs {ref_before[L]:+.3f} (|Δ| = {parity:.4f}) "
          f"{'ok' if parity < 0.0005 else 'MISMATCH'}")
    print(f"  cos(d_norm_heldout, d_norm_full) = {cos_dirs:+.3f}   "
          f"cos(d_disp_heldout, d_disp_full) = {cos_disp_dirs:+.3f}")
    print(f"  carrier: in-sample = {r_in:+.3f}   OUT-OF-SAMPLE (test words) = {r_out:+.3f}   "
          f"(disp out-of-sample = {r_out_disp:+.3f})")
    print(f"  cos(BALANCE, d_norm_heldout) = {cos_bn:+.3f}")
    print(f"  suffix x BALANCE — before -> after HELD-OUT norm+disp strip:")
    for sn in SUFFIX_ORDER:
        frac = sink_after[sn] / sink_before[sn] if abs(sink_before[sn]) > 0.01 else float("nan")
        print(f"    {sn:<16} {sink_before[sn]:>+7.3f} -> {sink_after[sn]:>+7.3f}   "
              f"({100*frac:.0f}% retained)")
    print(f"  inflectional mean sink: {infl_before:+.3f} -> {infl_after:+.3f} "
          f"({100*infl_after/infl_before:.0f}% retained)")
    print(f"  random-2D-strip band (inflectional mean): "
          f"{rand_means.mean():+.3f} ± {rand_means.std():.3f} "
          f"[{rand_means.min():+.3f}, {rand_means.max():+.3f}]")
    summary[L] = dict(before=infl_before, after=infl_after, band=rand_means,
                      cos_dirs=cos_dirs, r_out=r_out, parity=parity)

# ---- verdict (corrected band logic) ----
print("\n" + "=" * 78)
print("VERDICT vs pre-registered decision rule")
print("=" * 78)
fracs = [summary[L]["after"] / summary[L]["before"] for L in LAYERS]
print("  inflectional sink retained per layer: " +
      ", ".join(f"L{L}:{100*f:.0f}%" for L, f in zip(LAYERS, fracs)))
mean_frac = float(np.mean(fracs))
abs_mean_frac = float(np.mean([abs(f) for f in fracs]))
# 'survives' = stripped sink stays within the random band (band sits at the
# unstripped value); 'collapses' = stripped sink far above (less negative
# than) the band, near zero.
in_band = [summary[L]["band"].min() <= summary[L]["after"] <= summary[L]["band"].max()
           for L in LAYERS]
print(f"  mean retained = {100*mean_frac:.0f}% (mean |retained| = {100*abs_mean_frac:.0f}%)")
print(f"  stripped sink within random band (= survival) per layer: "
      + ", ".join(f"L{L}:{'Y' if b else 'n'}" for L, b in zip(LAYERS, in_band)))
print(f"  cos(d_norm_heldout, d_norm_full) per layer: "
      + ", ".join(f"L{L}:{summary[L]['cos_dirs']:+.2f}" for L in LAYERS))
print(f"  out-of-sample carrier per layer: "
      + ", ".join(f"L{L}:{summary[L]['r_out']:+.2f}" for L in LAYERS))
if abs_mean_frac <= 0.25 and not any(in_band):
    print("  -> H1: collapse holds with held-out d_norm. Circularity dismissed;")
    print("     exp154's N2 verdict stands as written.")
elif mean_frac >= 0.60 or all(in_band):
    print("  -> H2: sink survives the held-out strip. exp154's collapse was")
    print("     substantially overfit — weaken the grounding claim and")
    print("     re-examine Claim 3 before exp152.")
else:
    print("  -> H3: partial overfit. Report both; prefer held-out numbers")
    print("     in the paper.")
