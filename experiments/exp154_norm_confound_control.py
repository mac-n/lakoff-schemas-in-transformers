"""
exp154_norm_confound_control.py — does the inflectional BALANCE sink reduce
to norm-displacement geometry?

The scariest hole in the "model grounds markedness in its own normalisation
physiology" chain (exp138 sink + exp141 BALANCE-norm r=0.85-0.97 + LN):
maybe the BALANCE schema axis is just a readout of residual-norm structure,
and the suffix x BALANCE sink is re-measuring norm displacement rather than
schema content.

Note: exp138 unit-normalises word vectors before building suffix directions,
so per-word norm MAGNITUDE is already removed. The live confound is the
norm-encoding DIRECTION: a direction d_norm such that proj(unit(r), d_norm)
predicts ||r||. If BALANCE ~ d_norm, the sink is norm geometry.

Design (per layer, exp138 word set and strip protocol):
  1. Δnorm check: per suffix, mean ||r(inflected)|| - ||r(base)|| — are
     inflected forms even at systematically different norms?
  2. Build d_norm: covariance direction sum_w z(||r_w||) * unit(r_w),
     aniso+freq stripped, normalised. Sanity: corr(proj onto d_norm, z-norm)
     across words should be substantial, else d_norm is meaningless and
     the confound has no carrier.
     Also d_disp: same with z(|z-norm|) (absolute displacement from
     norm-typical region — closer to exp141's "displacement" framing).
  3. cos(BALANCE_stripped, d_norm), cos(BALANCE_stripped, d_disp).
  4. Norm-strip: project d_norm AND d_disp out of both schema and suffix
     directions; recompute suffix x BALANCE. Sink before vs after.
  5. Control: same with 20 random stripped directions — stripping any one
     dimension shouldn't generically kill the sink. Report the random band.

PRE-REGISTRATION (2026-06-10, before running):
  Committed prediction (this Claude): cos(BALANCE, d_norm) will be
  substantial (|cos| ~ 0.5-0.8 at mid layers, consistent with exp141's
  correlations); the sink will ATTENUATE under norm-strip to roughly half
  its size but SURVIVE above the 20-seed random-strip band. I.e. partial:
  BALANCE carries both norm geometry and markedness content beyond it.
  Decision rule:
    N1 sink survives (>= ~60% magnitude, above random band): norm confound
       dismissed; BALANCE-as-schema carries markedness content beyond norm.
    N2 sink collapses toward random band (<= ~25% magnitude): the sink IS
       norm-displacement geometry. The grounding sentence becomes MORE
       literal ("markedness is implemented as norm-displacement, read out
       through BALANCE") but BALANCE-as-schema weakens to epiphenomenal
       readout. exp141's correspondence and exp138's sink become one
       finding, not two.
    N3 intermediate (25-60%): two-component story; report both components
       honestly; the paper's grounding claim needs the RMSNorm test before
       leaning hard either way.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


print("\n" + "=" * 78)
print("exp154 — norm-confound control on the inflectional BALANCE sink")
print("=" * 78)

summary = {}
rng_global = np.random.default_rng(7)

for L in LAYERS:
    # ---- strip machinery (exp138 protocol) ----
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

    # ---- 1. Δnorm per suffix ----
    norms = {w: float(np.linalg.norm(residuals[w][L])) for w in all_words}
    nvals = np.array([norms[w] for w in all_words])
    z = {w: (norms[w] - nvals.mean()) / nvals.std() for w in all_words}

    # ---- 2. norm-encoding directions ----
    units = {w: residuals[w][L] / norms[w] for w in all_words}
    d_norm = np.sum([z[w] * units[w] for w in all_words], axis=0)
    d_norm = strip(d_norm / np.linalg.norm(d_norm))
    absz = {w: abs(z[w]) for w in all_words}
    az_mean = np.mean(list(absz.values())); az_std = np.std(list(absz.values()))
    d_disp = np.sum([((absz[w] - az_mean) / az_std) * units[w] for w in all_words], axis=0)
    d_disp = strip(d_disp / np.linalg.norm(d_disp))
    # orthogonalise d_disp against d_norm for joint stripping
    d_disp_o = proj_out(d_disp, d_norm)

    # carrier sanity: does projection onto d_norm actually predict norm?
    proj_n = [float(strip(units[w] / np.linalg.norm(units[w])) @ d_norm) if False else float((units[w] @ d_norm)) for w in all_words]
    r_carrier = corrf(proj_n, [z[w] for w in all_words])
    proj_d = [float(units[w] @ d_disp) for w in all_words]
    r_carrier_disp = corrf(proj_d, [absz[w] for w in all_words])

    # ---- 3. BALANCE axis and its norm alignment ----
    def schema_dir(sn):
        pairs = LAKOFF_SCHEMAS_MML[sn]
        pos = sorted(set(p[0] for p in pairs)); neg = sorted(set(p[1] for p in pairs))
        raw = mean_acts(pos) - mean_acts(neg)
        return strip(raw / np.linalg.norm(raw))

    bal = schema_dir("BALANCE")
    cos_bn = float(bal @ d_norm); cos_bd = float(bal @ d_disp)

    # ---- 4. suffix directions, sink before/after norm-strip ----
    def suffix_dir(sn):
        diffs = []
        for base, infl in SUFFIX_PAIRS[sn]:
            b = residuals[base][L]; i = residuals[infl][L]
            diffs.append(i / np.linalg.norm(i) - b / np.linalg.norm(b))
        raw = np.mean(diffs, axis=0)
        return strip(raw / np.linalg.norm(raw))

    sufs = {sn: suffix_dir(sn) for sn in SUFFIX_ORDER}
    sink_before = {sn: float(sufs[sn] @ bal) for sn in SUFFIX_ORDER}

    bal_ns = proj_out(proj_out(bal, d_norm), d_disp_o)
    sink_after = {sn: float(proj_out(proj_out(sufs[sn], d_norm), d_disp_o) @ bal_ns)
                  for sn in SUFFIX_ORDER}

    # ---- 5. random-strip control band (strip 2 random dims for parity) ----
    rand_means = []
    for _ in range(20):
        g1 = strip(rng_global.standard_normal(arr.shape[1]))
        g2 = proj_out(strip(rng_global.standard_normal(arr.shape[1])), g1)
        bal_r = proj_out(proj_out(bal, g1), g2)
        vals = [float(proj_out(proj_out(sufs[sn], g1), g2) @ bal_r) for sn in INFL]
        rand_means.append(np.mean(vals))
    rand_means = np.array(rand_means)

    # ---- report ----
    print(f"\n--- Layer {L} ---")
    print(f"  carrier sanity: corr(proj d_norm, z-norm) = {r_carrier:+.3f}; "
          f"corr(proj d_disp, |z|) = {r_carrier_disp:+.3f}")
    print(f"  cos(BALANCE, d_norm) = {cos_bn:+.3f}   cos(BALANCE, d_disp) = {cos_bd:+.3f}")
    print(f"  Δnorm (inflected - base), mean ± sd per suffix:")
    for sn in SUFFIX_ORDER:
        dn = [norms[i] - norms[b] for b, i in SUFFIX_PAIRS[sn]]
        print(f"    {sn:<16} {np.mean(dn):>+8.2f} ± {np.std(dn):>6.2f}   "
              f"(mean word norm {nvals.mean():.1f})")
    print(f"  suffix x BALANCE — before -> after norm+disp strip:")
    for sn in SUFFIX_ORDER:
        frac = sink_after[sn] / sink_before[sn] if abs(sink_before[sn]) > 0.01 else float("nan")
        print(f"    {sn:<16} {sink_before[sn]:>+7.3f} -> {sink_after[sn]:>+7.3f}   "
              f"({100*frac:.0f}% retained)")
    infl_before = np.mean([sink_before[s] for s in INFL])
    infl_after = np.mean([sink_after[s] for s in INFL])
    print(f"  inflectional mean sink: {infl_before:+.3f} -> {infl_after:+.3f} "
          f"({100*infl_after/infl_before:.0f}% retained)")
    print(f"  random-2D-strip control band (inflectional mean): "
          f"{rand_means.mean():+.3f} ± {rand_means.std():.3f} "
          f"[{rand_means.min():+.3f}, {rand_means.max():+.3f}]")
    summary[L] = (infl_before, infl_after, rand_means, cos_bn, cos_bd)

# ---- verdict ----
print("\n" + "=" * 78)
print("VERDICT vs pre-registered decision rule")
print("=" * 78)
fracs = [summary[L][1] / summary[L][0] for L in LAYERS]
print(f"  inflectional sink retained per layer: " +
      ", ".join(f"L{L}:{100*f:.0f}%" for L, f in zip(LAYERS, fracs)))
mean_frac = float(np.mean(fracs))
above_band = all(summary[L][1] < summary[L][2].min() for L in LAYERS)  # sink is negative
print(f"  mean retained = {100*mean_frac:.0f}%; stripped sink below random band "
      f"at every layer: {above_band}")
if mean_frac >= 0.60 and above_band:
    print("  -> N1: sink survives norm-strip. Norm confound dismissed; BALANCE")
    print("     carries markedness content beyond norm geometry.")
elif mean_frac <= 0.25:
    print("  -> N2: sink collapses. The sink IS norm-displacement geometry —")
    print("     exp138 and exp141 are one finding. Grounding more literal,")
    print("     BALANCE-as-schema epiphenomenal. Rewrite accordingly.")
else:
    print("  -> N3: two-component story. Report both; RMSNorm test (exp152)")
    print("     before leaning on either component.")

# ---- plot ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(LAYERS, [summary[L][0] for L in LAYERS], "o-", label="inflectional sink (before)")
ax.plot(LAYERS, [summary[L][1] for L in LAYERS], "s-", label="after norm+disp strip")
band_lo = [summary[L][2].min() for L in LAYERS]
band_hi = [summary[L][2].max() for L in LAYERS]
ax.fill_between(LAYERS, band_lo, band_hi, alpha=0.2, color="gray",
                label="random-2D-strip band (20 seeds)")
ax2 = ax.twinx()
ax2.plot(LAYERS, [summary[L][3] for L in LAYERS], "^--", color="tab:red",
         label="cos(BALANCE, d_norm)")
ax2.plot(LAYERS, [summary[L][4] for L in LAYERS], "v--", color="tab:orange",
         label="cos(BALANCE, d_disp)")
ax2.set_ylabel("cos(BALANCE, norm axes)")
ax.set_xlabel("layer"); ax.set_ylabel("mean inflectional suffix × BALANCE")
ax.axhline(0, color="gray", lw=0.5)
ax.legend(loc="lower left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
ax.set_title("exp154 — does the BALANCE sink survive removal of norm-encoding directions?")
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp154_norm_confound.png", dpi=120)
print("\nSaved exp154_norm_confound.png")
