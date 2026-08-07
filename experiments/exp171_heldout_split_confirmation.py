"""
exp171_heldout_split_confirmation.py — HELD-OUT confirmation of the
derivational/inflectional split (paper Claim 3 keystone; blog Finding 4).

See PREREG_exp171.md (frozen). Short form: exp150's clean split was
post-hoc on the original pairs. This runs the identical protocol on a
FROZEN fresh pair set (exp171_pairs_heldout.json, checksummed) and —
new content — actually significance-tests the derivational-vs-
inflectional DIFFERENCE via joint bootstrap over word pairs (T2).

Tests (declared in prereg):
  T1 ordering: r_deriv > r_infl (static vs Pythia), per binding substrate
  T2 difference: bootstrap CI95 on Δr = r_deriv − r_infl excludes 0
  T3 static-static inflectional agreement r >= +0.50
  T4 sink: all 5 inflectional suffixes BALANCE-negative in Pythia L12

Run: nice -n 10 ./lakoff/bin/python3 exp171_heldout_split_confirmation.py
ONLY after the shared GPU bench is free (foreground ps check first).
--selftest-only runs the synthetic harness test and exits.
"""

import gc
import hashlib
import json
import re
import sys

import numpy as np

# ---------------- frozen rule constants (asserted vs prereg) ----------------
BOOT_N = 2000
CI = 95
MIN_PAIRS = 8
T3_MIN = 0.50
PAIRS_FILE = "/Users/macn/Documents/embeddingexp/exp171_pairs_heldout.json"
PAIRS_SHA256 = "c6bbd7f0ddd75d0e802f6e631eba9568f704f5ae7c7e66ee9d1c5e979293c2bc"

PREREG = open("/Users/macn/Documents/embeddingexp/PREREG_exp171.md").read()
assert "BOOT_N = 2000" in PREREG and "MIN_PAIRS = 8" in PREREG \
    and "T3_MIN = 0.50" in PREREG, "rule constants missing from prereg"
assert PAIRS_SHA256 in PREREG, "pairs checksum missing from prereg"

SUFFIX_ORDER = ["ER_comparative", "EST_superlative", "ING_progressive",
                "ED_past", "S_plural", "UN_negation", "RE_repetition"]
INFL = SUFFIX_ORDER[:5]
DERIV = SUFFIX_ORDER[5:]
SCHEMA_NAMES = ["UP-DOWN", "IN-OUT_CLEAN", "FORWARD-BACK", "PATH-MOTION",
                "LIGHT-DARK", "FORCE", "BALANCE", "DIFFICULTY-BURDEN"]
COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]
PYTHIA_LAYERS = [4, 12, 20]
L_PRIMARY = 12


def pearson(x, y):
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    x = x - x.mean(); y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float((x @ y) / den) if den > 0 else float("nan")


# ============================================================================
# Core machinery, substrate-agnostic: works on {word: vector} dicts.
# ============================================================================
def build_axes(vecs, abtt=False):
    """abtt=True: All-But-The-Top top-3-PC strip (exp102 precedent) — the
    prereg's frozen commitment FOR STATIC SUBSTRATES. abtt=False: exp138
    mean+freq strip (used for Pythia; was erroneously applied to static in
    the first run — injector #2 catch, corrective run uses abtt for static)."""
    if abtt:
        M = np.stack(list(vecs.values()))
        mu = M.mean(axis=0)
        Mc = M - mu
        # top-3 principal components
        _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
        pcs = Vt[:3]

        def strip(d):
            d = d - (d @ (mu / np.linalg.norm(mu))) * (mu / np.linalg.norm(mu))
            for pc in pcs:
                d = d - (d @ pc) * pc
            return d / np.linalg.norm(d)
        return strip
    mean_vec = np.mean(list(vecs.values()), axis=0)
    aniso = mean_vec / np.linalg.norm(mean_vec)
    cm = [vecs[w] for w in COMMON if w in vecs]
    rr = [vecs[w] for w in RARE if w in vecs]
    freq_raw = np.mean(cm, axis=0) - np.mean(rr, axis=0)
    freq = freq_raw / np.linalg.norm(freq_raw)
    freq_orth = freq - (freq @ aniso) * aniso
    freq_orth = freq_orth / np.linalg.norm(freq_orth)

    def strip(d):
        d = d - (d @ aniso) * aniso
        d = d - (d @ freq_orth) * freq_orth
        return d / np.linalg.norm(d)
    return strip


def schema_dirs(vecs, strip, lakoff):
    out = {}
    for sn in SCHEMA_NAMES:
        pairs = lakoff[sn]
        pos = sorted(set(p[0] for p in pairs) & set(vecs))
        neg = sorted(set(p[1] for p in pairs) & set(vecs))
        raw = (np.mean([vecs[w] for w in pos], axis=0)
               - np.mean([vecs[w] for w in neg], axis=0))
        out[sn] = strip(raw / np.linalg.norm(raw))
    return out


def pair_diffs(vecs, pairs):
    """Per-pair normalized difference vectors (the bootstrap unit)."""
    diffs = []
    for base, infl in pairs:
        if base in vecs and infl in vecs:
            b = vecs[base]; i = vecs[infl]
            diffs.append(i / np.linalg.norm(i) - b / np.linalg.norm(b))
    return diffs


def matrix_from_diffs(diffs_by_suffix, sdirs, strip):
    M = np.zeros((len(SUFFIX_ORDER), len(SCHEMA_NAMES)))
    for i, suf in enumerate(SUFFIX_ORDER):
        raw = np.mean(diffs_by_suffix[suf], axis=0)
        d = strip(raw / np.linalg.norm(raw))
        for j, sch in enumerate(SCHEMA_NAMES):
            M[i, j] = float(d @ sdirs[sch])
    return M


def split_r(M_static, M_pythia):
    infl_idx = [SUFFIX_ORDER.index(s) for s in INFL]
    der_idx = [SUFFIX_ORDER.index(s) for s in DERIV]
    r_infl = pearson(M_static[infl_idx], M_pythia[infl_idx])
    r_deriv = pearson(M_static[der_idx], M_pythia[der_idx])
    return r_deriv, r_infl


def bootstrap_delta(diffs_static, diffs_pythia, sdirs_s, strip_s,
                    sdirs_p, strip_p, n_boot=BOOT_N, seed=17):
    """Joint bootstrap over word pairs (same resample indices applied to
    both substrates — the pairs are shared objects). Returns CI on
    Δr = r_deriv − r_infl."""
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        bs_s, bs_p = {}, {}
        for suf in SUFFIX_ORDER:
            n = len(diffs_static[suf])
            idx = rng.integers(0, n, n)
            bs_s[suf] = [diffs_static[suf][i] for i in idx]
            bs_p[suf] = [diffs_pythia[suf][i] for i in idx]
        Ms = matrix_from_diffs(bs_s, sdirs_s, strip_s)
        Mp = matrix_from_diffs(bs_p, sdirs_p, strip_p)
        rd, ri = split_r(Ms, Mp)
        deltas[k] = rd - ri
    lo, hi = np.percentile(deltas, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    return float(lo), float(hi), float(np.mean(deltas))


# ============================================================================
# Synthetic self-test — harness must discriminate three planted worlds
# ============================================================================
def synth_world(kind, seed=3, dim=50, n_pairs=12, noise=0.6):
    rng = np.random.default_rng(seed)
    lakoff = {sn: [(f"p{sn}{k}", f"n{sn}{k}") for k in range(6)]
              for sn in SCHEMA_NAMES}
    # planted schema axes shared across substrates
    axes = np.linalg.qr(rng.standard_normal((dim, 8)))[0].T
    # per-suffix "true content": deriv rows shared across families;
    # infl rows: static family shares one geometry, pythia has another
    shared = {s: rng.standard_normal(8) for s in DERIV}
    infl_static = {s: rng.standard_normal(8) for s in INFL}
    infl_pythia = {s: rng.standard_normal(8) for s in INFL}

    def make_substrate(family, nse):
        vecs = {}
        for sn in SCHEMA_NAMES:
            for k, (p, n) in enumerate(lakoff[sn]):
                a = axes[SCHEMA_NAMES.index(sn)]
                vecs[p] = a * 3 + rng.standard_normal(dim) * 0.5
                vecs[n] = -a * 3 + rng.standard_normal(dim) * 0.5
        for w in COMMON + RARE:
            vecs[w] = rng.standard_normal(dim)
        diffs = {}
        for s in SUFFIX_ORDER:
            if s in DERIV:
                load = shared[s]
            elif kind == "ii":            # world ii: infl ALSO shared
                load = infl_static[s]
            else:
                load = infl_static[s] if family == "static" else infl_pythia[s]
            base_dir = (load[None, :] @ axes).ravel()
            diffs[s] = [base_dir + rng.standard_normal(dim) * nse
                        for _ in range(n_pairs)]
        return vecs, diffs

    nse = noise * (3.0 if kind == "iii" else 1.0)
    return make_substrate("static", nse), make_substrate("pythia", nse), lakoff


def self_test():
    print("=" * 78)
    print("SYNTHETIC SELF-TEST (worlds: i=split real, ii=no split, iii=split+noise)")
    print("=" * 78)
    verdicts = {}
    for kind in ["i", "ii", "iii"]:
        (v_s, d_s), (v_p, d_p), lak = synth_world(kind)
        strip_s = build_axes(v_s); strip_p = build_axes(v_p)
        sd_s = schema_dirs(v_s, strip_s, lak)
        sd_p = schema_dirs(v_p, strip_p, lak)
        Ms = matrix_from_diffs(d_s, sd_s, strip_s)
        Mp = matrix_from_diffs(d_p, sd_p, strip_p)
        rd, ri = split_r(Ms, Mp)
        lo, hi, mu = bootstrap_delta(d_s, d_p, sd_s, strip_s, sd_p, strip_p,
                                     n_boot=300)  # light for self-test
        verdicts[kind] = (rd, ri, lo, hi)
        print(f"  world {kind}: r_deriv {rd:+.3f}  r_infl {ri:+.3f}  "
              f"Δr CI [{lo:+.3f}, {hi:+.3f}]")
    ok_i = verdicts["i"][0] > verdicts["i"][1] and verdicts["i"][2] > 0
    ok_ii = not (verdicts["ii"][2] > 0)      # CI must NOT exclude 0 upward
    ok_iii = verdicts["iii"][0] > verdicts["iii"][1]  # ordering holds, CI may be wide
    print(f"  discriminates i (confirm): {ok_i}  ii (no-split): {ok_ii}  "
          f"iii (ordering under noise): {ok_iii}")
    assert ok_i and ok_ii and ok_iii, "SELF-TEST FAILED — do not run on models"
    print("  SELF-TEST PASS")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    self_test()
    if "--selftest-only" in sys.argv:
        sys.exit(0)

    # ---- frozen materials ----
    raw = open(PAIRS_FILE, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    assert got == PAIRS_SHA256, f"pairs file checksum mismatch: {got}"
    heldout = {k: [tuple(p) for p in v] for k, v in json.loads(raw).items()
               if not k.startswith("_")}
    assert sorted(heldout) == sorted(SUFFIX_ORDER)

    sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
    from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML

    # ---- runtime exclusions ----
    old_src = open("/Users/macn/Documents/embeddingexp/exp138_morphology_anisotropy_strip.py").read()
    OLD_PAIRS = eval(re.search(r"SUFFIX_PAIRS = (\{.*?\n\})", old_src, re.S).group(1))
    burned = set()
    for prs in OLD_PAIRS.values():
        for b, i in prs:
            burned.add(b); burned.add(i)
    anchors = set()
    for sn in SCHEMA_NAMES:
        for p, n in LAKOFF_SCHEMAS_MML[sn]:
            anchors.add(p); anchors.add(n)
    excluded_report = {}
    for suf in SUFFIX_ORDER:
        kept, dropped = [], []
        for b, i in heldout[suf]:
            (dropped if (b in burned or i in burned or b in anchors
                         or i in anchors) else kept).append((b, i))
        excluded_report[suf] = dropped
        heldout[suf] = kept
    print("Runtime exclusions (burned/anchor collisions):")
    for suf in SUFFIX_ORDER:
        print(f"  {suf:<18} kept {len(heldout[suf]):>2}  "
              f"dropped {excluded_report[suf]}")

    # ---- collect word set ----
    words = set(COMMON + RARE) | anchors
    for prs in heldout.values():
        for b, i in prs:
            words.add(b); words.add(i)
    words = sorted(words)

    results = {}

    # ---- Pythia 410M ----
    import torch
    from transformer_lens import HookedTransformer
    print("\nLoading Pythia 410M...")
    model = HookedTransformer.from_pretrained("pythia-410m", device="mps")
    model.eval()
    hooks = [f"blocks.{L}.hook_resid_post" for L in PYTHIA_LAYERS]
    pythia_vecs = {L: {} for L in PYTHIA_LAYERS}
    for k, w in enumerate(words):
        toks = model.to_tokens(w)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=hooks)
        for L in PYTHIA_LAYERS:
            pythia_vecs[L][w] = cache[f"blocks.{L}.hook_resid_post"][0, -1, :]\
                .cpu().numpy().astype(np.float64)
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(words)}")
    del model, cache
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass

    pythia_side = {}
    for L in PYTHIA_LAYERS:
        v = pythia_vecs[L]
        strip_p = build_axes(v)
        sd_p = schema_dirs(v, strip_p, LAKOFF_SCHEMAS_MML)
        d_p = {s: pair_diffs(v, heldout[s]) for s in SUFFIX_ORDER}
        Mp = matrix_from_diffs(d_p, sd_p, strip_p)
        pythia_side[L] = (v, strip_p, sd_p, d_p, Mp)
    # T4 at L12
    Mp12 = pythia_side[L_PRIMARY][4]
    bal_j = SCHEMA_NAMES.index("BALANCE")
    t4_col = Mp12[[SUFFIX_ORDER.index(s) for s in INFL], bal_j]
    T4 = bool(np.all(t4_col < 0))
    print(f"\nT4 sink on held-out pairs (Pythia L12 BALANCE column, INFL): "
          f"{np.array2string(t4_col, precision=3)}  -> all negative: {T4}")

    # ---- static substrates, sequential ----
    import gensim.downloader as api
    static_mats = {}
    binding = ["glove-wiki-gigaword-300", "word2vec-google-news-300"]
    supplementary = ["fasttext-wiki-news-subwords-300"]
    for name in binding + supplementary:
        print(f"\nLoading {name} ...")
        wv = api.load(name)
        vecs = {w: np.asarray(wv[w], np.float64) for w in words
                if w in wv.key_to_index}
        miss = [w for w in words if w not in vecs]
        print(f"  vocab hit {len(vecs)}/{len(words)}; missing e.g. {miss[:8]}")
        use_abtt = "--abtt" in sys.argv
        if use_abtt:
            print("  [static strip: ABTT top-3 PCs, per frozen prereg commitment]")
        strip_s = build_axes(vecs, abtt=use_abtt)
        sd_s = schema_dirs(vecs, strip_s, LAKOFF_SCHEMAS_MML)
        d_s = {s: pair_diffs(vecs, heldout[s]) for s in SUFFIX_ORDER}
        ns = {s: len(d_s[s]) for s in SUFFIX_ORDER}
        nonbind = [s for s in SUFFIX_ORDER if ns[s] < MIN_PAIRS]
        if nonbind:
            print(f"  NOTE: suffixes below MIN_PAIRS={MIN_PAIRS}: "
                  f"{[(s, ns[s]) for s in nonbind]} — reported, non-binding")
        Ms = matrix_from_diffs(d_s, sd_s, strip_s)
        static_mats[name] = (Ms, d_s, sd_s, strip_s, ns)

        v_p, strip_p, sd_p, d_p_full, Mp = pythia_side[L_PRIMARY]
        # align pair lists: use pairs present in BOTH (joint bootstrap needs
        # index parity)
        d_p = {}
        for s in SUFFIX_ORDER:
            present = [(b, i) for (b, i) in heldout[s]
                       if b in vecs and i in vecs]
            d_p[s] = pair_diffs(v_p, present)
            d_s[s] = pair_diffs(vecs, present)
        Ms = matrix_from_diffs(d_s, sd_s, strip_s)
        Mp_al = matrix_from_diffs(d_p, sd_p, strip_p)
        rd, ri = split_r(Ms, Mp_al)
        lo, hi, mu = bootstrap_delta(d_s, d_p, sd_s, strip_s, sd_p, strip_p)
        results[name] = dict(r_deriv=rd, r_infl=ri, dlo=lo, dhi=hi, dmu=mu)
        print(f"  vs Pythia L12: r_deriv {rd:+.3f}  r_infl {ri:+.3f}  "
              f"Δr plug-in {rd-ri:+.3f} (boot mean {mu:+.3f}) CI95 [{lo:+.3f}, {hi:+.3f}]")
        # L4/L20 secondary report (frozen prereg: "L4, L20 reported")
        for Lsec in [4, 20]:
            v_ps, strip_ps, sd_ps, _, _ = pythia_side[Lsec]
            d_ps = {s: pair_diffs(v_ps, [(b, i) for (b, i) in heldout[s]
                                         if b in vecs and i in vecs])
                    for s in SUFFIX_ORDER}
            Mp_sec = matrix_from_diffs(d_ps, sd_ps, strip_ps)
            rds, ris = split_r(Ms, Mp_sec)
            print(f"    [L{Lsec} secondary] r_deriv {rds:+.3f}  r_infl {ris:+.3f}  "
                  f"Δr plug-in {rds-ris:+.3f}")
        del wv, vecs
        gc.collect()

    # T3 static-static inflectional agreement (binding pair)
    infl_idx = [SUFFIX_ORDER.index(s) for s in INFL]
    Ma = static_mats[binding[0]][0][infl_idx]
    Mb = static_mats[binding[1]][0][infl_idx]
    t3_r = pearson(Ma, Mb)
    T3 = bool(t3_r >= T3_MIN)
    print(f"\nT3 static-static inflectional agreement "
          f"({binding[0].split('-')[0]}↔{binding[1].split('-')[0]}): "
          f"r = {t3_r:+.3f}  (>= {T3_MIN}: {T3})")

    # ---- verdict ----
    print("\n" + "=" * 78)
    print("VERDICT vs PREREG_exp171.md decision rule")
    print("=" * 78)
    t1 = {n: results[n]["r_deriv"] > results[n]["r_infl"] for n in binding}
    t2 = {n: results[n]["dlo"] > 0 for n in binding}
    T1 = all(t1.values()); n_T2 = sum(t2.values())
    print(f"  T1 ordering per binding substrate: {t1}  -> {T1}")
    print(f"  T2 Δr CI95 excludes 0: {t2}  -> {n_T2}/2")
    print(f"  T3 {T3} (r={t3_r:+.3f})   T4 {T4}")
    if not T1:
        v = "SPLIT_REFUTED — post-hoc split fails on held-out pairs; rewrite Claim 3 keystone prominently."
    elif n_T2 == 2 and T3 and T4:
        v = "SPLIT_CONFIRMED — upgrade blog/paper to held-out confirmed."
    elif n_T2 >= 1 or (n_T2 == 2 and not (T3 and T4)):
        v = "SPLIT_PARTIAL — directionally confirmed; difference test mixed/underpowered; keep 'descriptive' language where failed."
    else:
        v = "SPLIT_NOT_CONFIRMED — ordering replicates, difference not distinguishable at this N; say so plainly."
    print(f"  -> {v}")
