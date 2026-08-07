"""
markedness_norm_protocol.py — shared protocol for the exp154-family
norm/markedness experiments (exp151, exp152, and any later ports).

Extracted 2026-06-11 after three copy-paste generations of this machinery
(exp154, exp154c, exp154d) bred two sign bugs. One implementation, one
place to fix it. Behaviour matches exp154c exactly (held-out d_norm is
the STANDARD here — exp154d showed full-set estimation is circular).

Provides:
  - the exp138 vocabulary (suffix pairs, schemas, COMMON/RARE)
  - residual collection (last-token; BOS-safe — works whether or not the
    tokenizer prepends BOS, cf. Pythia-no-BOS vs Llama-BOS)
  - per-layer analysis: sink before, full-set strip, HELD-OUT strip,
    random-2D band, coupling cosines, carrier stats, Δnorm table
  - verdict() as a pure function of the summary + selftest() for the
    synthetic-verdict-test convention (adopted 2026-06-11)
"""

import numpy as np
import torch

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML

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


def build_word_lists():
    """Returns (all_words, est_words, test_words). test = suffix-pair words
    + BALANCE vocab; est = everything else (held-out d_norm estimation)."""
    all_words = set(COMMON + RARE)
    for pairs in SUFFIX_PAIRS.values():
        for b, i in pairs:
            all_words.add(b); all_words.add(i)
    for sn in SCHEMA_NAMES:
        for p, n in LAKOFF_SCHEMAS_MML[sn]:
            all_words.add(p); all_words.add(n)
    all_words = sorted(all_words)
    test_words = set()
    for pairs in SUFFIX_PAIRS.values():
        for b, i in pairs:
            test_words.add(b); test_words.add(i)
    for p, n in LAKOFF_SCHEMAS_MML["BALANCE"]:
        test_words.add(p); test_words.add(n)
    test_words &= set(all_words)
    est_words = [w for w in all_words if w not in test_words]
    return all_words, est_words, sorted(test_words)


def collect_residuals(model, layers, words, log_every=100):
    """Last-token resid_post per word per layer. Last-token indexing is
    BOS-safe (BOS, if any, sits at position 0)."""
    hook_names = [f"blocks.{L}.hook_resid_post" for L in layers]
    residuals = {}
    for k, w in enumerate(words):
        toks = model.to_tokens(w)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=hook_names)
        residuals[w] = {L: cache[f"blocks.{L}.hook_resid_post"][0, -1, :]
                        .float().cpu().numpy() for L in layers}
        if log_every and (k + 1) % log_every == 0:
            print(f"  {k+1}/{len(words)}")
    return residuals


def corrf(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def analyze_layer(residuals, L, all_words, est_words, test_words,
                  rng, n_rand=20):
    """Full exp154c analysis for one layer. Returns a dict of everything
    the verdicts and reports need."""
    aniso = np.stack([residuals[w][L] for w in all_words]).mean(axis=0)
    aniso = aniso / np.linalg.norm(aniso)

    def mean_acts(words):
        return np.mean([residuals[w][L] for w in words], axis=0)

    freq = mean_acts(COMMON) - mean_acts(RARE)
    freq = freq / np.linalg.norm(freq)
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
        return dn, dd, nv.mean(), nv.std()

    d_norm_ho, d_disp_ho, est_mean, est_std = build_norm_dirs(est_words)
    d_norm_full, d_disp_full, _, _ = build_norm_dirs(all_words)
    d_disp_ho_o = proj_out(d_disp_ho, d_norm_ho)
    d_disp_full_o = proj_out(d_disp_full, d_norm_full)

    z_test = [(norms[w] - est_mean) / est_std for w in test_words]
    carrier_out = corrf([float(units[w] @ d_norm_ho) for w in test_words], z_test)

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

    def strip2(v, a1, a2):
        return proj_out(proj_out(v, a1), a2)

    sink_before = {sn: float(sufs[sn] @ bal) for sn in SUFFIX_ORDER}
    bal_ho = strip2(bal, d_norm_ho, d_disp_ho_o)
    sink_ho = {sn: float(strip2(sufs[sn], d_norm_ho, d_disp_ho_o) @ bal_ho)
               for sn in SUFFIX_ORDER}
    bal_full = strip2(bal, d_norm_full, d_disp_full_o)
    sink_full = {sn: float(strip2(sufs[sn], d_norm_full, d_disp_full_o) @ bal_full)
                 for sn in SUFFIX_ORDER}

    dim = aniso.shape[0]
    rand_means = []
    for _ in range(n_rand):
        g1 = strip(rng.standard_normal(dim))
        g2 = proj_out(strip(rng.standard_normal(dim)), g1)
        bal_r = strip2(bal, g1, g2)
        rand_means.append(np.mean([float(strip2(sufs[sn], g1, g2) @ bal_r)
                                   for sn in INFL]))

    dnorm_suffix = {sn: (float(np.mean([norms[i] - norms[b] for b, i in SUFFIX_PAIRS[sn]])),
                         float(np.std([norms[i] - norms[b] for b, i in SUFFIX_PAIRS[sn]])))
                    for sn in SUFFIX_ORDER}

    return {
        "sink_before": sink_before, "sink_ho": sink_ho, "sink_full": sink_full,
        "infl_before": float(np.mean([sink_before[s] for s in INFL])),
        "infl_ho": float(np.mean([sink_ho[s] for s in INFL])),
        "infl_full": float(np.mean([sink_full[s] for s in INFL])),
        "rand_band": np.array(rand_means),
        "cos_bal_dnorm": float(bal @ d_norm_ho),
        "cos_dnorm_ho_full": float(d_norm_ho @ d_norm_full),
        "carrier_out": carrier_out,
        "dnorm_suffix": dnorm_suffix,
        "mean_norm": float(np.mean([norms[w] for w in all_words])),
        "est_after_ho": sink_ho["EST_superlative"],
    }


def report_layer(L, r):
    print(f"\n--- Layer {L} ---")
    print(f"  cos(BALANCE, d_norm_heldout) = {r['cos_bal_dnorm']:+.3f}   "
          f"cos(d_norm_ho, d_norm_full) = {r['cos_dnorm_ho_full']:+.3f}   "
          f"out-of-sample carrier = {r['carrier_out']:+.3f}")
    print(f"  Δnorm (inflected - base), mean ± sd per suffix "
          f"(mean word norm {r['mean_norm']:.1f}):")
    for sn in SUFFIX_ORDER:
        m, s = r["dnorm_suffix"][sn]
        print(f"    {sn:<16} {m:>+10.2f} ± {s:>8.2f}")
    print(f"  suffix x BALANCE — before / after FULL strip / after HELD-OUT strip:")
    for sn in SUFFIX_ORDER:
        print(f"    {sn:<16} {r['sink_before'][sn]:>+7.3f}  {r['sink_full'][sn]:>+7.3f}  "
              f"{r['sink_ho'][sn]:>+7.3f}")
    b, f, h = r["infl_before"], r["infl_full"], r["infl_ho"]
    print(f"  inflectional mean: {b:+.3f} -> full {f:+.3f} ({100*f/b:.0f}%) "
          f"-> held-out {h:+.3f} ({100*h/b:.0f}%)")
    rb = r["rand_band"]
    print(f"  random-2D band: {rb.mean():+.3f} ± {rb.std():.3f} "
          f"[{rb.min():+.3f}, {rb.max():+.3f}]")


# ---------------------------------------------------------------------------
# Verdict logic — pure functions of the summary, selftest-able.
# summary: {layer: analyze_layer-dict}; decision_layers: subset of layers.
# ---------------------------------------------------------------------------

def verdict_components(summary, decision_layers,
                       sink_thresh=-0.15, coupling_thresh=0.5,
                       residue_lo=0.15, residue_hi=0.60):
    """Three independently-reported components:
    Q1 sink exists: mean inflectional sink (before strip) <= sink_thresh
       at every decision layer.
    Q2 coupling exists: cos(BALANCE, d_norm_heldout) >= coupling_thresh
       at every decision layer.
    Q3 held-out residue: retained fraction (sink_ho/sink_before, valid
       only if Q1) within [residue_lo, residue_hi] on average — i.e. a
       real partial residue, neither full collapse nor full survival.
    Plus q3_mean_retained for reporting, and est_flip: EST x BALANCE
    positive after held-out strip at a majority of decision layers."""
    q1 = all(summary[L]["infl_before"] <= sink_thresh for L in decision_layers)
    q2 = all(summary[L]["cos_bal_dnorm"] >= coupling_thresh for L in decision_layers)
    if q1:
        fracs = [summary[L]["infl_ho"] / summary[L]["infl_before"]
                 for L in decision_layers]
        mean_ret = float(np.mean(fracs))
    else:
        mean_ret = float("nan")
    q3 = q1 and residue_lo <= mean_ret <= residue_hi
    est_flip = (sum(1 for L in decision_layers if summary[L]["est_after_ho"] > 0)
                > len(decision_layers) / 2)
    return {"q1_sink": q1, "q2_coupling": q2, "q3_residue": q3,
            "mean_retained": mean_ret, "est_flip": est_flip}


def selftest_verdict(decision_layers=(8, 12, 16, 20)):
    """Synthetic-verdict-test convention (2026-06-11): push fabricated
    summaries through verdict_components and assert each branch fires."""
    def fake(before, ho, cos_bn, est_after):
        return {L: {"infl_before": before, "infl_ho": ho,
                    "cos_bal_dnorm": cos_bn, "est_after_ho": est_after}
                for L in decision_layers}

    # clear replicate-with-residue (the exp154c-like outcome)
    v = verdict_components(fake(-0.30, -0.10, 0.70, +0.10), decision_layers)
    assert v["q1_sink"] and v["q2_coupling"] and v["q3_residue"] and v["est_flip"], v
    assert abs(v["mean_retained"] - 1/3) < 1e-9, v
    # clear no-sink
    v = verdict_components(fake(-0.05, -0.01, 0.70, -0.10), decision_layers)
    assert not v["q1_sink"] and v["q2_coupling"] and not v["q3_residue"], v
    # sink but no coupling, full collapse, no EST flip
    v = verdict_components(fake(-0.30, -0.01, 0.10, -0.10), decision_layers)
    assert v["q1_sink"] and not v["q2_coupling"] and not v["q3_residue"], v
    assert not v["est_flip"], v
    # sink survives fully (no residue window match: retained ~ 0.9)
    v = verdict_components(fake(-0.30, -0.27, 0.70, -0.10), decision_layers)
    assert v["q1_sink"] and not v["q3_residue"] and v["mean_retained"] > 0.85, v
    print("selftest_verdict: all branches fire correctly.")


if __name__ == "__main__":
    selftest_verdict()
