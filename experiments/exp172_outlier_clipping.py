"""
exp172_outlier_clipping.py — does the BALANCE–norm coupling survive
clipping massive/outlier dimensions? See PREREG_exp172.md (frozen).

Attack being tested (Opus injector via CONSULT_LOG 23:55[real ~21:54]
item 3): exp155's Δnorm scale jump (−7 → −535, L4→L8) smells of rogue
dims; maybe cos(BALANCE, d_norm) ≈ +0.7 is a few outlier dimensions.

Per layer (Pythia 410M, {4,8,12,16,20}, exp154 word set/strip):
  1. outlier dims: mean |activation| across words > mean + 5·SD of the
     per-dim distribution; ZERO them in all residuals first.
  2. rebuild BALANCE, d_norm, suffix dirs on clipped residuals;
     measure cos(BALANCE_clip, d_norm_clip) and the inflectional sink
     before/after clip.
  3. control: clip equal COUNT of random dims, 20 seeds → band.

Run AFTER exp170/171 on a free bench: nice -n 10
./lakoff/bin/python3 exp172_outlier_clipping.py
--selftest-only runs the synthetic discrimination test and exits.
"""

import re
import sys

import numpy as np

OUT_Z = 5.0
SURVIVE = 0.60
N_RAND = 20
PREREG = open("/Users/macn/Documents/embeddingexp/PREREG_exp172.md").read()
assert "OUT_Z = 5.0" in PREREG and "SURVIVE = 0.60" in PREREG \
    and "N_RAND = 20" in PREREG, "rule constants missing from prereg"

LAYERS = [4, 8, 12, 16, 20]
SRC = open("/Users/macn/Documents/embeddingexp/exp154_norm_confound_control.py").read()
SUFFIX_PAIRS = eval(re.search(r"SUFFIX_PAIRS = (\{.*?\n\})", SRC, re.S).group(1))
SCHEMA_NAMES = ["UP-DOWN", "IN-OUT_CLEAN", "FORWARD-BACK", "PATH-MOTION",
                "LIGHT-DARK", "FORCE", "BALANCE", "DIFFICULTY-BURDEN"]
COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]
SUFFIX_ORDER = list(SUFFIX_PAIRS.keys())
INFL = ["ER_comparative", "EST_superlative", "ING_progressive", "ED_past",
        "S_plural"]


def corrf(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float((x @ y) / den) if den > 0 else float("nan")


def outlier_dims(R, z=OUT_Z):
    """R: (n_words, dim). Dims whose mean |act| exceeds a ROBUST threshold:
    median + z * (1.4826*MAD) of the per-dim mean-|act| distribution.
    (Prereg amendment, pre-data: mean+z*SD self-masks — the outliers
    inflate the SD used to detect them; caught by the synthetic test.)"""
    m = np.abs(R).mean(axis=0)
    med = np.median(m)
    mad = np.median(np.abs(m - med)) * 1.4826
    return np.where(m > med + z * mad)[0]


def analyse_layer(R, words, lakoff, clip_dims=None):
    """Full exp154-style measurement on residual matrix R (rows=words).
    clip_dims: indices zeroed BEFORE anything else. Returns
    (cos_bal_dnorm, sink_infl_mean, share_of_norm_variance_clipped)."""
    R = R.copy()
    share = 0.0
    if clip_dims is not None and len(clip_dims):
        tot = (R ** 2).sum()
        share = float((R[:, clip_dims] ** 2).sum() / tot)
        R[:, clip_dims] = 0.0
    vec = {w: R[k] for k, w in enumerate(words)}

    aniso = R.mean(axis=0); aniso = aniso / np.linalg.norm(aniso)
    freq_raw = (np.mean([vec[w] for w in COMMON], axis=0)
                - np.mean([vec[w] for w in RARE], axis=0))
    freq = freq_raw / np.linalg.norm(freq_raw)
    freq_orth = freq - (freq @ aniso) * aniso
    freq_orth = freq_orth / np.linalg.norm(freq_orth)

    def strip(d):
        d = d - (d @ aniso) * aniso
        d = d - (d @ freq_orth) * freq_orth
        return d / np.linalg.norm(d)

    norms = {w: float(np.linalg.norm(vec[w])) for w in words}
    nv = np.array([norms[w] for w in words])
    z = {w: (norms[w] - nv.mean()) / nv.std() for w in words}
    units = {w: vec[w] / norms[w] for w in words}
    d_norm = np.sum([z[w] * units[w] for w in words], axis=0)
    d_norm = strip(d_norm / np.linalg.norm(d_norm))

    pairs = lakoff["BALANCE"]
    pos = sorted(set(p[0] for p in pairs) & set(vec))
    neg = sorted(set(p[1] for p in pairs) & set(vec))
    raw = (np.mean([vec[w] for w in pos], axis=0)
           - np.mean([vec[w] for w in neg], axis=0))
    bal = strip(raw / np.linalg.norm(raw))
    cos_bn = float(bal @ d_norm)

    sinks = []
    for sn in INFL:
        diffs = []
        for b, i in SUFFIX_PAIRS[sn]:
            if b in vec and i in vec:
                diffs.append(vec[i] / np.linalg.norm(vec[i])
                             - vec[b] / np.linalg.norm(vec[b]))
        d = np.mean(diffs, axis=0)
        d = strip(d / np.linalg.norm(d))
        sinks.append(float(d @ bal))
    return cos_bn, float(np.mean(sinks)), share


# ---------------- synthetic self-test ----------------
def synth(kind, seed=5, n_words=300, dim=64, n_out=3):
    """(a) coupling carried BY outlier dims -> ARTIFACT;
    (b) coupling distributed, outliers present but orthogonal -> ROBUST."""
    rng = np.random.default_rng(seed)
    words = [f"w{k}" for k in range(n_words)]
    lakoff = {sn: [(f"p{sn}{i}", f"n{sn}{i}") for i in range(5)]
              for sn in SCHEMA_NAMES}
    # give schema words real vectors too
    all_w = list(words)
    for sn in SCHEMA_NAMES:
        for p, n in lakoff[sn]:
            all_w += [p, n]
    for sn_ in [("x", "y")]:
        pass
    suffix_words = set()
    for prs in SUFFIX_PAIRS.values():
        for b, i in prs:
            suffix_words.add(b); suffix_words.add(i)
    all_w += sorted(suffix_words) + COMMON + RARE
    all_w = list(dict.fromkeys(all_w))
    out_dims = np.arange(n_out)                      # first dims are outliers
    bal_axis = np.zeros(dim)
    if kind == "a":
        bal_axis[out_dims] = rng.standard_normal(n_out)   # BAL lives in outliers
    else:
        bal_axis[n_out:] = rng.standard_normal(dim - n_out)  # distributed
    bal_axis /= np.linalg.norm(bal_axis)
    R = []
    balset = set()
    for p, n in lakoff["BALANCE"]:
        balset.add(p); balset.add(n)
    for w in all_w:
        v = rng.standard_normal(dim)
        v[out_dims] += rng.standard_normal(n_out) * 12.0   # massive dims
        s = rng.standard_normal()
        v += 2.5 * s * bal_axis                            # norm-coupled comp
        v *= (1 + 0.25 * s)
        if w in balset:
            sign = 1 if w.startswith("pBAL") else -1
            v += 4.0 * sign * bal_axis
        R.append(v)
    return np.stack(R), all_w, lakoff


def self_test():
    print("=" * 78)
    print("SYNTHETIC SELF-TEST (a: coupling in outlier dims; b: distributed)")
    print("=" * 78)
    for kind, want in [("a", "ARTIFACT"), ("b", "ROBUST")]:
        R, words, lak = synth(kind)
        od = outlier_dims(R)
        c0, s0, _ = analyse_layer(R, words, lak, clip_dims=None)
        c1, s1, share = analyse_layer(R, words, lak, clip_dims=od)
        frac = abs(c1) / abs(c0) if abs(c0) > 1e-9 else float("nan")
        print(f"  world {kind}: n_outlier={len(od)}  cos {c0:+.3f} -> {c1:+.3f} "
              f"({100*frac:.0f}% retained; clipped {100*share:.1f}% of energy)  want {want}")
        if kind == "a":
            ok_a = frac < 0.30
        else:
            ok_b = frac >= 0.60
    print(f"  discriminates a (artifact): {ok_a}  b (robust): {ok_b}")
    assert ok_a and ok_b, "SELF-TEST FAILED — do not run on model"
    print("  SELF-TEST PASS")


if __name__ == "__main__":
    self_test()
    if "--selftest-only" in sys.argv:
        sys.exit(0)

    sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
    from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML
    import torch
    from transformer_lens import HookedTransformer

    print("\nLoading Pythia 410M...")
    model = HookedTransformer.from_pretrained("pythia-410m", device="mps")
    model.eval()
    hooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]

    words = set(COMMON + RARE)
    for prs in SUFFIX_PAIRS.values():
        for b, i in prs:
            words.add(b); words.add(i)
    for sn in SCHEMA_NAMES:
        for p, n in LAKOFF_SCHEMAS_MML[sn]:
            words.add(p); words.add(n)
    words = sorted(words)
    print(f"Collecting residuals for {len(words)} words at {LAYERS}...")
    store = {L: [] for L in LAYERS}
    for k, w in enumerate(words):
        toks = model.to_tokens(w)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=hooks)
        for L in LAYERS:
            store[L].append(cache[f"blocks.{L}.hook_resid_post"][0, -1, :]
                            .cpu().numpy().astype(np.float64))
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(words)}")

    print("\n" + "=" * 78)
    print("exp172 — outlier-dimension robustness")
    print("=" * 78)
    rng = np.random.default_rng(11)
    fracs_c, fracs_s, ok_rand = [], [], []
    for L in LAYERS:
        R = np.stack(store[L])
        od = outlier_dims(R)
        c0, s0, _ = analyse_layer(R, words, LAKOFF_SCHEMAS_MML, None)
        c1, s1, share = analyse_layer(R, words, LAKOFF_SCHEMAS_MML, od)
        rc, rs = [], []
        for _ in range(N_RAND):
            rd = rng.choice(R.shape[1], size=max(len(od), 1), replace=False)
            cr, sr, _ = analyse_layer(R, words, LAKOFF_SCHEMAS_MML, rd)
            rc.append(cr); rs.append(sr)
        rc = np.array(rc); rs = np.array(rs)
        fc = abs(c1) / abs(c0) if abs(c0) > 1e-9 else float("nan")
        fs = s1 / s0 if abs(s0) > 1e-9 else float("nan")
        fracs_c.append(fc); fracs_s.append(fs)
        ok_rand.append(bool(abs(rc.mean()) >= 0.9 * abs(c0)))
        print(f"\n--- Layer {L} ---")
        print(f"  outlier dims: {len(od)} {list(od[:12])}  "
              f"({100*share:.1f}% of residual energy)")
        print(f"  cos(BALANCE, d_norm): {c0:+.3f} -> clipped {c1:+.3f} "
              f"({100*fc:.0f}% retained)  random-clip band "
              f"{rc.mean():+.3f} ± {rc.std():.3f}")
        print(f"  infl mean sink:       {s0:+.3f} -> clipped {s1:+.3f} "
              f"({100*fs:.0f}% retained)  random band {rs.mean():+.3f} ± {rs.std():.3f}")

    print("\n" + "=" * 78)
    print("VERDICT vs PREREG_exp172.md")
    print("=" * 78)
    n_c = sum(f >= SURVIVE for f in fracs_c)
    n_s = sum(f >= SURVIVE for f in fracs_s)
    n_art = sum(f < 0.30 for f in fracs_c)
    print(f"  coupling retained >= {SURVIVE}: {n_c}/5 layers; "
          f"sink retained >= {SURVIVE}: {n_s}/5; coupling < 0.30: {n_art}/5")
    print(f"  random-clip preserved coupling per layer: {ok_rand}")
    if n_c >= 4 and n_s >= 4:
        print("  -> ROBUST: coupling and sink are distributed, not outlier-dim artifacts.")
    elif n_art >= 3:
        print("  -> ARTIFACT: coupling rides outlier dims; demote 'norm physiology' wording.")
    else:
        print("  -> MIXED: report per-layer; keep the blog's concession sentence.")
