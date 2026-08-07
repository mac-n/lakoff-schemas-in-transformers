"""
exp170_dnorm_purity.py — d_norm TOKEN-COUNT PURITY (the owed control).

See PREREG_exp170.md (frozen, incl. 22:20 pre-data amendment). Short form:
exp157b showed corr(zipf, norm) is ~90% tokenization. d_norm (exp154) is
built from full-vocab norm variation, so it is partly a token-count
direction. Does cos(BALANCE, d_norm) survive when d_norm is rebuilt clean?

Variants per layer (exp154 word set, strip protocol, layers {4,8,12,16,20}):
  d_norm_orig   — exp154 as-is (replication gate: must give +0.70..+0.78)
  d_norm_single — covariance sum over SINGLE-TOKEN words only
  d_norm_resid  — all words, z(norm) residualized on token count (OLS)
  d_tokcount    — covariance direction of z(token_count): the confound

Plus SINK_MATCHED: the sink recomputed on the POOLED token-count-matched
inflectional pairs (N=27 per prereg amendment; binding on pooled value).

RULE PARAMETERS asserted against PREREG_exp170.md at runtime.
Run: nice -n 10 ./lakoff/bin/python3 exp170_dnorm_purity.py
ONLY after the shared GPU bench is free (foreground ps check first).
"""

import re
import sys

import numpy as np
import torch

# ---------------- rule constants (asserted vs prereg file) ----------------
COUPLE_HI = 0.50
COUPLE_LO = 0.20
LAYERS_MAJ = 3
SINK_SURVIVE = 0.60
CARRIER_MIN = 0.30
MIN_MATCHED_N = 15

PREREG = open("/Users/macn/Documents/embeddingexp/PREREG_exp170.md").read()
for name, val in [("COUPLE_HI", "0.50"), ("COUPLE_LO", "0.20"),
                  ("LAYERS_MAJ = 3", ""), ("SINK_SURVIVE = 0.60", ""),
                  ("CARRIER_MIN = 0.30", "")]:
    assert (f"{name} = {val}".strip() if val else name) in PREREG, \
        f"rule constant {name} missing from PREREG_exp170.md"
assert "N were ever < 15" in PREREG, "MIN_MATCHED_N rule missing from prereg"

LAYERS = [4, 8, 12, 16, 20]

# exp154's word machinery, imported by execution to guarantee identity
SRC = open("/Users/macn/Documents/embeddingexp/exp154_norm_confound_control.py").read()
SUFFIX_PAIRS = eval(re.search(r"SUFFIX_PAIRS = (\{.*?\n\})", SRC, re.S).group(1))
SCHEMA_NAMES = ["UP-DOWN", "IN-OUT_CLEAN", "FORWARD-BACK", "PATH-MOTION",
                "LIGHT-DARK", "FORCE", "BALANCE", "DIFFICULTY-BURDEN"]
COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]
SUFFIX_ORDER = list(SUFFIX_PAIRS.keys())
INFL = ["ER_comparative", "EST_superlative", "ING_progressive", "ED_past", "S_plural"]


def corrf(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float((x @ y) / den) if den > 0 else float("nan")


def analyse(residuals, tokcounts, all_words, lakoff_schemas, layers, label=""):
    """Full per-layer analysis. residuals: {word: {L: vec}}; tokcounts: {word: int}.
    Returns dict for verdict; prints the report."""
    out = {}
    rng = np.random.default_rng(7)
    for L in layers:
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
        nvals = np.array([norms[w] for w in all_words])
        z = {w: (norms[w] - nvals.mean()) / nvals.std() for w in all_words}
        units = {w: residuals[w][L] / norms[w] for w in all_words}
        tc = np.array([tokcounts[w] for w in all_words], float)
        zn = np.array([z[w] for w in all_words])

        def cov_dir(weights):
            d = np.sum([wt * units[w] for wt, w in zip(weights, all_words)], axis=0)
            return strip(d / np.linalg.norm(d))

        # d_norm_orig — exp154 construction
        d_orig = cov_dir(zn)
        # d_tokcount — the confound direction itself
        ztc = (tc - tc.mean()) / tc.std()
        d_tok = cov_dir(ztc)
        # d_norm_resid — z(norm) with token count regressed out (OLS)
        beta = float((ztc @ zn) / (ztc @ ztc))
        zres = zn - beta * ztc
        var_ratio = float(np.var(zres) / np.var(zn))
        if var_ratio < 0.05:
            print(f"  WARNING L{L}: residual variance ratio {var_ratio:.3f} < 0.05 — "
                  f"d_resid UNINFORMATIVE (norm ≈ pure token count)")
        d_resid = cov_dir(zres)
        # d_norm_single — single-token words only
        single = [w for w in all_words if tokcounts[w] == 1]
        sn_norms = np.array([norms[w] for w in single])
        zs = (sn_norms - sn_norms.mean()) / sn_norms.std()
        d_single = np.sum([zs[k] * units[w] for k, w in enumerate(single)], axis=0)
        d_single = strip(d_single / np.linalg.norm(d_single))

        # carrier sanity
        def carrier(d, words, target):
            return corrf([float(units[w] @ d) for w in words], target)
        car_orig = carrier(d_orig, all_words, zn)
        car_tok = corrf([float(units[w] @ d_tok) for w in all_words], ztc)
        car_resid = carrier(d_resid, all_words, zres)
        car_single = corrf([float(units[w] @ d_single) for w in single], zs)

        # BALANCE axis
        def schema_dir(sname):
            pairs = lakoff_schemas[sname]
            pos = sorted(set(p[0] for p in pairs)); neg = sorted(set(p[1] for p in pairs))
            raw = mean_acts(pos) - mean_acts(neg)
            return strip(raw / np.linalg.norm(raw))

        bal = schema_dir("BALANCE")
        cos_orig = float(bal @ d_orig); cos_single = float(bal @ d_single)
        cos_resid = float(bal @ d_resid); cos_tok = float(bal @ d_tok)
        cos_orig_tok = float(d_orig @ d_tok)

        # suffix directions + sink
        def suffix_dir_from(pairs):
            diffs = []
            for base, infl in pairs:
                b = residuals[base][L]; i = residuals[infl][L]
                diffs.append(i / np.linalg.norm(i) - b / np.linalg.norm(b))
            raw = np.mean(diffs, axis=0)
            return strip(raw / np.linalg.norm(raw))

        sufs = {s: suffix_dir_from(SUFFIX_PAIRS[s]) for s in SUFFIX_ORDER}
        sink_before = {s: float(sufs[s] @ bal) for s in SUFFIX_ORDER}
        infl_before = float(np.mean([sink_before[s] for s in INFL]))

        def strip_pair(dvec, a1, a2):
            return proj_out(proj_out(dvec, a1), a2)

        def sink_after_strip(a1):
            a2 = None
            # d_disp analogue for the given d_norm flavour: abs-z covariance
            # (kept identical in spirit to exp154's d_disp, orthogonalised)
            az = np.abs(zn); azz = (az - az.mean()) / az.std()
            dd = cov_dir(azz)
            a2 = proj_out(dd, a1)
            bal_s = strip_pair(bal, a1, a2)
            return float(np.mean([strip_pair(sufs[s], a1, a2) @ bal_s for s in INFL]))

        infl_after_orig = sink_after_strip(d_orig)
        infl_after_single = sink_after_strip(d_single)
        infl_after_resid = sink_after_strip(d_resid)

        # random band (20 seeds, 2-dir strip parity)
        rand_means = []
        for _ in range(20):
            g1 = strip(rng.standard_normal(arr.shape[1]))
            g2 = proj_out(strip(rng.standard_normal(arr.shape[1])), g1)
            bal_r = strip_pair(bal, g1, g2)
            rand_means.append(float(np.mean(
                [strip_pair(sufs[s], g1, g2) @ bal_r for s in INFL])))
        rand_means = np.array(rand_means)

        # SINK_MATCHED — pooled token-count-matched inflectional pairs
        matched = [(b, i) for s in INFL for (b, i) in SUFFIX_PAIRS[s]
                   if tokcounts[b] == tokcounts[i]]
        n_matched = len(matched)
        pooled_dir = suffix_dir_from(matched)
        sink_matched_pooled = float(pooled_dir @ bal)
        per_pair = [float(strip((residuals[i][L] / np.linalg.norm(residuals[i][L]))
                                - (residuals[b][L] / np.linalg.norm(residuals[b][L])))
                          @ bal) for b, i in matched]
        sink_matched_meanpp = float(np.mean(per_pair))

        print(f"\n--- Layer {L} {label}---")
        print(f"  carriers: orig {car_orig:+.3f}  single {car_single:+.3f}  "
              f"resid {car_resid:+.3f}  tokcount {car_tok:+.3f}")
        print(f"  cos(BALANCE, .): orig {cos_orig:+.3f}  SINGLE {cos_single:+.3f}  "
              f"RESID {cos_resid:+.3f}  tokcount {cos_tok:+.3f}")
        print(f"  confound anatomy: cos(d_orig, d_tok) = {cos_orig_tok:+.3f}")
        print(f"  infl mean sink before {infl_before:+.3f} -> after strip: "
              f"orig {infl_after_orig:+.3f}  single {infl_after_single:+.3f}  "
              f"resid {infl_after_resid:+.3f}")
        print(f"  random-2D band: {rand_means.mean():+.3f} ± {rand_means.std():.3f}")
        print(f"  SINK_MATCHED (N={n_matched} pooled pairs): pooled-dir "
              f"{sink_matched_pooled:+.3f}  mean-per-pair {sink_matched_meanpp:+.3f}")
        out[L] = dict(cos_orig=cos_orig, cos_single=cos_single, cos_resid=cos_resid,
                      cos_tok=cos_tok, cos_orig_tok=cos_orig_tok,
                      car_orig=car_orig, car_single=car_single, car_resid=car_resid,
                      infl_before=infl_before, after_orig=infl_after_orig,
                      after_single=infl_after_single, after_resid=infl_after_resid,
                      rand=rand_means, n_matched=n_matched,
                      matched_pooled=sink_matched_pooled,
                      matched_meanpp=sink_matched_meanpp)
    return out


def verdict(out, layers):
    print("\n" + "=" * 78)
    print("VERDICT vs PREREG_exp170.md decision rule")
    print("=" * 78)
    # replication gate
    gate = all(0.65 <= out[L]["cos_orig"] <= 0.83 for L in layers)
    print(f"  replication gate (cos_orig in ~[0.70,0.78], tol ±0.05): "
          f"{'PASS' if gate else 'FAIL — STOP, protocol drift'}")
    if not gate:
        return "INVALID"
    for v, cname in [("cos_single", "car_single"), ("cos_resid", "car_resid")]:
        weak = [L for L in layers if abs(out[L][cname]) < CARRIER_MIN]
        if weak:
            print(f"  NOTE {v}: carrier < {CARRIER_MIN} at layers {weak} — "
                  f"UNINFORMATIVE there, not evidence of purity")
    hi_single = sum(out[L]["cos_single"] >= COUPLE_HI for L in layers)
    hi_resid = sum(out[L]["cos_resid"] >= COUPLE_HI for L in layers)
    lo_single = sum(abs(out[L]["cos_single"]) < COUPLE_LO for L in layers)
    lo_resid = sum(abs(out[L]["cos_resid"]) < COUPLE_LO for L in layers)
    n_matched = out[layers[0]]["n_matched"]
    matched_ok = (n_matched >= MIN_MATCHED_N and
                  all(out[L]["matched_pooled"] < 0 for L in layers))
    # sink-survival: clean-strip collapse vs orig-strip collapse
    def collapse(key):
        return np.mean([1 - out[L][key] / out[L]["infl_before"] for L in layers])
    surv = (collapse("after_single") >= SINK_SURVIVE * collapse("after_orig")
            and collapse("after_resid") >= SINK_SURVIVE * collapse("after_orig"))
    print(f"  cos>=HI: single {hi_single}/5, resid {hi_resid}/5 | "
          f"|cos|<LO: single {lo_single}/5, resid {lo_resid}/5")
    print(f"  clean-strip collapse >= {SINK_SURVIVE}x orig collapse: {surv}")
    print(f"  SINK_MATCHED pooled negative every layer (N={n_matched}): {matched_ok}")
    if hi_single >= LAYERS_MAJ and hi_resid >= LAYERS_MAJ and surv and matched_ok:
        r = "PURITY_CONFIRMED — magnitude stands; Part 2 wording unchanged."
    elif lo_single >= 4 and lo_resid >= 4:
        r = "PURITY_KILLED — magnitude was tokenization; retire the +0.7 number."
    else:
        r = "PURITY_PARTIAL — two-component; clean numbers become primary."
    print(f"  -> {r}")
    return r


# ---------------- synthetic self-test (must pass BEFORE model) ----------------
def synthetic_world(kind, n_words=400, dim=64, seed=0):
    """Plant residuals with known structure. Returns (residuals, tokcounts,
    words, schemas) shaped like the real pipeline expects."""
    rng = np.random.default_rng(seed)
    words = [f"w{k}" for k in range(n_words)]
    tokc = {w: int(rng.integers(1, 4)) for w in words}
    bal_axis = rng.standard_normal(dim); bal_axis /= np.linalg.norm(bal_axis)
    ind_axis = rng.standard_normal(dim)
    ind_axis -= (ind_axis @ bal_axis) * bal_axis; ind_axis /= np.linalg.norm(ind_axis)
    res = {}
    for w in words:
        base = rng.standard_normal(dim)
        tc_component = (tokc[w] - 2.0)
        if kind == "A":      # BALANCE-norm alignment purely token-count-driven
            # norm has token-independent noise too (realistic; keeps zres
            # non-degenerate) but the BALANCE component rides ONLY tokcount
            scale = 8.0 + 2.0 * tc_component + 0.8 * rng.standard_normal()
            vec = base + 3.0 * tc_component * bal_axis
        elif kind == "B":    # token-independent norm component aligned w/ BALANCE
            s_ind = rng.standard_normal()
            scale = 8.0 + 2.0 * s_ind
            vec = base + 3.0 * s_ind * bal_axis
        else:                # C: mixture
            s_ind = rng.standard_normal()
            scale = 8.0 + 1.4 * tc_component + 1.4 * s_ind
            vec = base + 2.0 * tc_component * bal_axis + 2.0 * s_ind * bal_axis
        vec = vec / np.linalg.norm(vec) * scale
        res[w] = {L: vec for L in LAYERS}
    return res, tokc, words, bal_axis


def self_test():
    print("=" * 78)
    print("SYNTHETIC SELF-TEST (three planted worlds; harness must discriminate)")
    print("=" * 78)
    results = {}
    for kind in ["A", "B", "C"]:
        res, tokc, words, bal_axis = synthetic_world(kind)
        # minimal fake vocab structures reusing real word lists is impossible
        # here; instead measure the core discriminator directly: cos(planted
        # BALANCE axis proxy, d_norm variants). We recompute variants inline.
        L = LAYERS[0]
        norms = {w: float(np.linalg.norm(res[w][L])) for w in words}
        nv = np.array([norms[w] for w in words])
        zn = (nv - nv.mean()) / nv.std()
        units = np.stack([res[w][L] / norms[w] for w in words])
        tc = np.array([tokc[w] for w in words], float)
        ztc = (tc - tc.mean()) / tc.std()
        d_orig = (zn[:, None] * units).sum(0); d_orig /= np.linalg.norm(d_orig)
        beta = float((ztc @ zn) / (ztc @ ztc))
        zres = zn - beta * ztc
        d_resid = (zres[:, None] * units).sum(0); d_resid /= np.linalg.norm(d_resid)
        sing = tc == 1
        zs = (nv[sing] - nv[sing].mean()) / nv[sing].std()
        d_single = (zs[:, None] * units[sing]).sum(0); d_single /= np.linalg.norm(d_single)
        c = dict(orig=float(bal_axis @ d_orig), single=float(bal_axis @ d_single),
                 resid=float(bal_axis @ d_resid))
        results[kind] = c
        print(f"  world {kind}: cos(planted BAL, d): orig {c['orig']:+.3f}  "
              f"single {c['single']:+.3f}  resid {c['resid']:+.3f}")
    okA = abs(results["A"]["single"]) < 0.25 and abs(results["A"]["resid"]) < 0.25 \
        and abs(results["A"]["orig"]) > 0.5
    okB = results["B"]["single"] > 0.5 and results["B"]["resid"] > 0.5
    okC = 0.2 < results["C"]["single"] and results["C"]["single"] < results["C"]["orig"] + 0.05
    print(f"  discriminates A (killed): {okA}  B (confirmed): {okB}  C (partial): {okC}")
    assert okA and okB and okC, "SELF-TEST FAILED — do not run on model"
    print("  SELF-TEST PASS")


if __name__ == "__main__":
    self_test()
    if "--selftest-only" in sys.argv:
        sys.exit(0)

    sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
    from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML
    from transformer_lens import HookedTransformer

    device = "mps"
    print("\nLoading Pythia 410M...")
    model = HookedTransformer.from_pretrained("pythia-410m", device=device)
    model.eval()
    hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]

    all_words = set(COMMON + RARE)
    for pairs in SUFFIX_PAIRS.values():
        for b, i in pairs:
            all_words.add(b); all_words.add(i)
    for s in SCHEMA_NAMES:
        for p, n in LAKOFF_SCHEMAS_MML[s]:
            all_words.add(p); all_words.add(n)
    all_words = sorted(all_words)
    print(f"Collecting residuals for {len(all_words)} words at {LAYERS}...")

    residuals, tokcounts = {}, {}
    for k, w in enumerate(all_words):
        toks = model.to_tokens(w)
        tokcounts[w] = int(toks.shape[1] - 1)  # minus BOS
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=hook_names)
        residuals[w] = {L: cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy()
                        for L in LAYERS}
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(all_words)}")

    single_n = sum(1 for w in all_words if tokcounts[w] == 1)
    print(f"single-token words: {single_n}/{len(all_words)}")

    print("\n" + "=" * 78)
    print("exp170 — d_norm token-count purity")
    print("=" * 78)
    out = analyse(residuals, tokcounts, all_words, LAKOFF_SCHEMAS_MML, LAYERS)
    verdict(out, LAYERS)
