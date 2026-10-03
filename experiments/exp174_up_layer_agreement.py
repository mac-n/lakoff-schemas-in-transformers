"""
exp174_up_layer_agreement.py — is UP grounded in layer agreement?

Frozen rules: PREREG_exp174.md (commit f2e30a4, 2026-10-02 16:43 IST),
including its FREEZE AMENDMENTS section. Written AFTER the freeze.

Part A (words): give every word a layer-agreement score C_adj, build the
direction along which it varies (single-token words only = primary), and
ask whether the UP axis lines up with it more than the other schemas do.

Part B (texts): happy / sad / neutral sentences, intact and scrambled,
each presented twice. Does the UP readout follow agreement or valence?

Order of operations: frozen-constant assertions -> synthetic self-tests
(real verdict paths) -> model load -> replication gate (exp170's
cos(BALANCE, d_norm_orig)) -> Part A -> Part B.

Results are written to disk as they are produced and the run resumes
from what is there (word residual cache, centring mean, per-text rows).

Usage:
  ./lakoff/bin/python3 exp174_up_layer_agreement.py --selftest-only
  ./lakoff/bin/python3 exp174_up_layer_agreement.py
"""

import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = "/Users/macn/Documents/embeddingexp"

# ---- frozen rule parameters (asserted against the prereg below) -------------
COUPLE_HI = 0.30
COUPLE_LO = 0.10
LAYERS_MAJ = 3
CARRIER_MIN = 0.30
NULL_PCT = 95
N_NULL = 100
GATE_D = 0.50
EFFECT_D = 0.30
N_SHUFFLES = 5
REPL_TOL = 0.02
N_PERM = 200
N_BOOT = 2000
FLAT_D = 0.20

LAYERS = [4, 8, 12, 16, 20]
N_LAYERS = 24
N_SPLITS = 20
REPL_TARGET = {4: 0.784, 8: 0.729, 12: 0.701, 16: 0.718, 20: 0.729}

PREREG = open(f"{ROOT}/PREREG_exp174.md").read()
for _name, _val in [("COUPLE_HI", COUPLE_HI), ("COUPLE_LO", COUPLE_LO),
                    ("LAYERS_MAJ", LAYERS_MAJ), ("CARRIER_MIN", CARRIER_MIN),
                    ("NULL_PCT", NULL_PCT), ("N_NULL", N_NULL),
                    ("GATE_D", GATE_D), ("EFFECT_D", EFFECT_D),
                    ("N_SHUFFLES", N_SHUFFLES), ("REPL_TOL", REPL_TOL),
                    ("N_PERM", N_PERM), ("N_BOOT", N_BOOT), ("FLAT_D", FLAT_D)]:
    _m = re.search(rf"\b{_name} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_val), f"{_name} drifted from prereg"
for _L, _t in REPL_TARGET.items():
    assert f"L{_L} +{_t:.3f}" in PREREG, f"replication target L{_L} drifted"

STIM_PATH = f"{ROOT}/exp174_stimuli.json"
_stim_blob = open(STIM_PATH, "rb").read()
STIM_SHA = hashlib.sha256(_stim_blob).hexdigest()
assert STIM_SHA in PREREG, "stimuli file does not match the sha256 in the prereg"


# ---- word lists, taken from the source files that defined them --------------
def _grab_list(src, name):
    m = re.search(rf"^{name}\s*=\s*(\[.*?\])", src, re.S | re.M)
    assert m, name
    return eval(m.group(1))


_SRC170 = open(f"{ROOT}/exp170_dnorm_purity.py").read()
_SRC154 = open(f"{ROOT}/exp154_norm_confound_control.py").read()
_SRC141 = open(f"{ROOT}/exp141_substrate_primitives.py").read()
COMMON = _grab_list(_SRC170, "COMMON")
RARE = _grab_list(_SRC170, "RARE")
SCHEMA_NAMES = _grab_list(_SRC170, "SCHEMA_NAMES")
SUFFIX_PAIRS = eval(re.search(r"SUFFIX_PAIRS = (\{.*?\n\})", _SRC154, re.S).group(1))
LAKOFF_UP = _grab_list(_SRC141, "LAKOFF_UP")
LAKOFF_DOWN = _grab_list(_SRC141, "LAKOFF_DOWN")
VALENCE_POS = _grab_list(_SRC141, "VALENCE_POS")
VALENCE_NEG = _grab_list(_SRC141, "VALENCE_NEG")
assert SCHEMA_NAMES[0] == "UP-DOWN" and len(SCHEMA_NAMES) == 8
assert len(LAKOFF_UP) == 14 and len(LAKOFF_DOWN) == 13


# ---- small statistics -------------------------------------------------------
def corrf(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float((x @ y) / den) if den > 0 else float("nan")


def zscore(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0.0


def d_z(x, y):
    """Paired effect size: mean difference over sd of differences."""
    d = np.asarray(x, float) - np.asarray(y, float)
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 0 else 0.0


def cohen_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


# ---- the carrier ------------------------------------------------------------
def compute_C(R, mu=None):
    """Layer agreement for each row of R.

    R: [n, n_layers, d] residuals (resid_post at every layer).
    mu: [n_layers, d] centring mean; defaults to the mean over the n rows.
    Returns C_adj [n], path length [n], C_straight [n].
    """
    R = np.asarray(R, np.float64)
    if mu is None:
        mu = R.mean(axis=0)
    X = R - mu[None, :, :]
    a, b = X[:, :-1, :], X[:, 1:, :]
    num = np.einsum("nld,nld->nl", a, b)
    den = np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)
    c_adj = (num / np.maximum(den, 1e-12)).mean(axis=1)
    steps = np.linalg.norm(R[:, 1:, :] - R[:, :-1, :], axis=2)
    path = steps.sum(axis=1)
    straight = np.linalg.norm(R[:, -1, :] - R[:, 0, :], axis=1) / np.maximum(path, 1e-12)
    return c_adj, path, straight


# ---- Part A analysis --------------------------------------------------------
def analyse_A(R_layers, C, tok, words, lists, layers, seed=174, quiet=False):
    """R_layers: {L: [n, d]} raw last-token residuals; C: [n] layer agreement;
    tok: [n] token counts; words: list of n words; lists: dict with keys
    common, rare, schemas {name: (pos, neg)} (the 7 non-UP schemas),
    up_literal (pos, neg), up_bundle (pos, neg), val (pos, neg)."""
    idx = {w: i for i, w in enumerate(words)}
    tok = np.asarray(tok, float)
    C = np.asarray(C, float)
    single = np.where(tok == 1)[0]
    axes_def = {"UP_literal": lists["up_literal"], "UP_bundle": lists["up_bundle"]}
    axes_def.update(lists["schemas"])
    axes_def["VAL"] = lists["val"]
    rank_names = ["UP_literal"] + list(lists["schemas"].keys())
    up_anchor = set(lists["up_literal"][0]) | set(lists["up_literal"][1]) \
        | set(lists["up_bundle"][0]) | set(lists["up_bundle"][1])
    held = np.array([i for i in single if words[i] not in up_anchor])
    pool = sorted({w for name in rank_names + ["UP_bundle"]
                   for side in axes_def[name] for w in side})
    n_pos, n_neg = len(lists["up_literal"][0]), len(lists["up_literal"][1])

    out = {"layers": list(layers), "n_words": len(words), "n_single": int(len(single)),
           "n_heldout": int(len(held))}
    saved_axes = {}
    for L in layers:
        rng = np.random.default_rng(seed + L)
        arr = np.asarray(R_layers[L], np.float64)
        aniso = arr.mean(axis=0); aniso /= np.linalg.norm(aniso)

        def mean_acts(ws):
            return arr[[idx[w] for w in ws]].mean(axis=0)

        freq = mean_acts(lists["common"]) - mean_acts(lists["rare"])
        freq /= np.linalg.norm(freq)
        freq_orth = freq - (freq @ aniso) * aniso
        freq_orth /= np.linalg.norm(freq_orth)

        def strip(d):
            d = d - (d @ aniso) * aniso
            d = d - (d @ freq_orth) * freq_orth
            return d / np.linalg.norm(d)

        norms = np.linalg.norm(arr, axis=1)
        units = arr / norms[:, None]

        def cov_dir(weights, rows):
            d = np.asarray(weights, float) @ units[rows]
            return strip(d / np.linalg.norm(d))

        def axis(pos, neg):
            raw = mean_acts(sorted(set(pos))) - mean_acts(sorted(set(neg)))
            return strip(raw / np.linalg.norm(raw))

        allrows = np.arange(len(words))
        zC = zscore(C)
        ztc = zscore(tok)
        zn = zscore(norms)
        # residualise on token count; on token count and norm
        beta = float((ztc @ zC) / (ztc @ ztc)) if ztc.std() > 0 else 0.0
        zres = zC - beta * ztc
        Xr = np.stack([ztc, zn], axis=1)
        coef, *_ = np.linalg.lstsq(Xr, zC, rcond=None)
        zres2 = zC - Xr @ coef
        zs = zscore(C[single])
        zn_s = zscore(norms[single])
        zh = zscore(C[held])

        dirs = {
            "single": cov_dir(zs, single),
            "resid": cov_dir(zres, allrows),
            "resid2": cov_dir(zres2, allrows),
            "orig": cov_dir(zC, allrows),
            "tokcount": cov_dir(ztc, allrows) if ztc.std() > 0 else None,
            "norm_single": cov_dir(zn_s, single),
            "heldout": cov_dir(zh, held),
        }
        axes = {name: axis(*pn) for name, pn in axes_def.items()}
        saved_axes[L] = {k: axes[k] for k in ("UP_literal", "UP_bundle", "VAL")}

        def cv_carrier(rows, target):
            vals = []
            n = len(rows)
            for _ in range(N_SPLITS):
                perm = rng.permutation(n)
                A, B = perm[: n // 2], perm[n // 2:]
                for tr, te in ((A, B), (B, A)):
                    d = cov_dir(zscore(target[tr]), rows[tr])
                    vals.append(corrf(units[rows[te]] @ d, target[te]))
            return float(np.mean(vals))

        res = {
            "cv_carrier_single": cv_carrier(single, C[single]),
            "cv_carrier_resid": cv_carrier(allrows, zres),
            "insample_carrier_single": corrf(units[single] @ dirs["single"], zs),
            "insample_carrier_resid": corrf(units @ dirs["resid"], zres),
            "corr_C_norm_all": corrf(C, norms),
            "corr_C_norm_single": corrf(C[single], norms[single]),
            "resid_var_ratio": float(np.var(zres) / np.var(zC)),
            "cos": {}, "confound": {},
        }
        for name, ax in axes.items():
            res["cos"][name] = {k: (float(ax @ d) if d is not None else float("nan"))
                                for k, d in dirs.items()}
        if dirs["tokcount"] is not None:
            res["confound"]["cos_orig_tok"] = float(dirs["orig"] @ dirs["tokcount"])
            res["confound"]["cos_single_tok"] = float(dirs["single"] @ dirs["tokcount"])
        res["confound"]["cos_single_normsingle"] = float(dirs["single"] @ dirs["norm_single"])

        # rank of UP among the eight schemas by |cos with d_coh_single|
        mags = {n_: abs(res["cos"][n_]["single"]) for n_ in rank_names}
        order = sorted(rank_names, key=lambda n_: -mags[n_])
        res["up_rank"] = order.index("UP_literal") + 1
        res["rank_order"] = order

        # null 1: random-partition axes, pole sizes matched to UP_literal
        part = []
        for _ in range(N_NULL):
            pick = rng.choice(len(pool), size=n_pos + n_neg, replace=False)
            pw = [pool[i] for i in pick]
            part.append(abs(float(axis(pw[:n_pos], pw[n_pos:]) @ dirs["single"])))
        res["partnull95"] = float(np.percentile(part, NULL_PCT))
        # null 2: shuffled carrier
        perm_lit, perm_bun = [], []
        for _ in range(N_PERM):
            dperm = cov_dir(rng.permutation(zs), single)
            perm_lit.append(abs(float(axes["UP_literal"] @ dperm)))
            perm_bun.append(abs(float(axes["UP_bundle"] @ dperm)))
        res["perm95"] = float(np.percentile(perm_lit, NULL_PCT))
        res["perm95_bundle"] = float(np.percentile(perm_bun, NULL_PCT))

        # valence anatomy: UP with VAL projected out
        up_nv = axes["UP_literal"] - (axes["UP_literal"] @ axes["VAL"]) * axes["VAL"]
        up_nv /= np.linalg.norm(up_nv)
        res["cos_UPnoVAL_single"] = float(up_nv @ dirs["single"])
        res["cos_UP_VAL"] = float(axes["UP_literal"] @ axes["VAL"])
        # EXPLORATORY (added post-freeze, pre-data; non-binding): across the
        # held-out single-token words, does a word's position on the UP axis
        # track its layer agreement? Permutation p, two-sided.
        proj = units[held] @ axes["UP_literal"]
        r_obs = corrf(proj, C[held])
        r_perm = np.array([corrf(proj, rng.permutation(C[held])) for _ in range(2000)])
        res["explore_heldout_corr"] = r_obs
        res["explore_heldout_p"] = float((np.sum(np.abs(r_perm) >= abs(r_obs)) + 1) / (len(r_perm) + 1))
        out[L] = res

    # layer-free instrument checks
    path = lists.get("path")
    zipf = lists.get("zipf")
    out["instrument"] = {
        "C_mean": float(C.mean()), "C_sd": float(C.std()),
        "C_single_mean": float(C[single].mean()), "C_single_sd": float(C[single].std()),
        "corr_C_tok": corrf(C, tok) if tok.std() > 0 else float("nan"),
        "corr_C_path_all": corrf(C, path) if path is not None else float("nan"),
        "corr_C_path_single": corrf(C[single], np.asarray(path)[single]) if path is not None else float("nan"),
        "corr_C_zipf_all": corrf(C, zipf) if zipf is not None else float("nan"),
        "corr_C_zipf_single": corrf(C[single], np.asarray(zipf)[single]) if zipf is not None else float("nan"),
    }
    sset = set(single.tolist())
    up_s = [C[idx[w]] for w in lists["up_literal"][0] if idx[w] in sset]
    dn_s = [C[idx[w]] for w in lists["up_literal"][1] if idx[w] in sset]
    out["plain"] = {"n_up": len(up_s), "n_down": len(dn_s),
                    "mean_C_up": float(np.mean(up_s)) if up_s else float("nan"),
                    "mean_C_down": float(np.mean(dn_s)) if dn_s else float("nan"),
                    "cohen_d": cohen_d(up_s, dn_s)}
    # EXPLORATORY (post-freeze, pre-data; non-binding): permutation p for the
    # plain UP-vs-DOWN difference in C, shuffling the pole labels.
    if len(up_s) >= 2 and len(dn_s) >= 2:
        rng_p = np.random.default_rng(seed)
        both = np.array(up_s + dn_s); k = len(up_s)
        obs = both[:k].mean() - both[k:].mean()
        hits = 0
        for _ in range(10000):
            q = rng_p.permutation(both)
            hits += abs(q[:k].mean() - q[k:].mean()) >= abs(obs)
        out["plain"]["perm_p"] = float((hits + 1) / 10001)
    else:
        out["plain"]["perm_p"] = float("nan")
    out["_axes"] = saved_axes
    return out


def verdict_A(out):
    """Frozen Part A rule (prereg FREEZE AMENDMENT 3). Primary: UP_literal vs
    d_coh_single."""
    Ls = out["layers"]
    if sum(out[L]["cv_carrier_single"] < CARRIER_MIN for L in Ls) >= 3:
        return "UNINFORMATIVE"
    cs = {L: out[L]["cos"]["UP_literal"]["single"] for L in Ls}
    cr = {L: out[L]["cos"]["UP_literal"]["resid"] for L in Ls}
    sig = {L: abs(cs[L]) > out[L]["perm95"] and abs(cs[L]) > out[L]["partnull95"] for L in Ls}
    if sum(cs[L] <= -COUPLE_HI and sig[L] for L in Ls) >= LAYERS_MAJ:
        return "WRONG_SIGN"
    if sum(cs[L] >= COUPLE_HI and sig[L] for L in Ls) >= LAYERS_MAJ:
        if sum(out[L]["up_rank"] == 1 for L in Ls) >= LAYERS_MAJ:
            return "COUPLED_SPECIFIC"
        return "COUPLED_GENERIC"
    both_lo = sum(abs(cs[L]) < COUPLE_LO and abs(cr[L]) < COUPLE_LO for L in Ls) >= 4
    no_beat = sum(abs(cs[L]) <= out[L]["perm95"] for L in Ls) >= 3
    if both_lo or no_beat:
        return "NULL"
    return "WEAK"


def report_A(out):
    Ls = out["layers"]
    ins = out["instrument"]
    print(f"words {out['n_words']}  single-token {out['n_single']}  "
          f"held-out single-token (no UP/DOWN anchors) {out['n_heldout']}")
    print(f"C_adj all words: mean {ins['C_mean']:+.3f} sd {ins['C_sd']:.3f} | "
          f"single-token: mean {ins['C_single_mean']:+.3f} sd {ins['C_single_sd']:.3f}")
    print("INSTRUMENT CHECKS (what else C_adj tracks)")
    print(f"  corr(C, token count)        all {ins['corr_C_tok']:+.3f}")
    print(f"  corr(C, path length)        all {ins['corr_C_path_all']:+.3f}   single {ins['corr_C_path_single']:+.3f}"
          "   (strongly negative = agreement by inaction)")
    print(f"  corr(C, zipf frequency)     all {ins['corr_C_zipf_all']:+.3f}   single {ins['corr_C_zipf_single']:+.3f}")
    for L in Ls:
        r = out[L]
        print(f"  L{L:<2} corr(C, norm) all {r['corr_C_norm_all']:+.3f} single {r['corr_C_norm_single']:+.3f}"
              f" | resid var ratio {r['resid_var_ratio']:.3f}")
    pl = out["plain"]
    print("PLAIN VERSION: do UP words have more layer agreement than DOWN words?")
    print(f"  single-token UP words n={pl['n_up']} mean C {pl['mean_C_up']:+.4f} | "
          f"DOWN words n={pl['n_down']} mean C {pl['mean_C_down']:+.4f} | Cohen d {pl['cohen_d']:+.2f}"
          f" | label-shuffle p {pl['perm_p']:.3f} (exploratory)")
    for L in Ls:
        r = out[L]
        c = r["cos"]
        print(f"--- Layer {L} ---")
        print(f"  carrier d_coh_single: cross-validated {r['cv_carrier_single']:+.3f}"
              f" (in-sample {r['insample_carrier_single']:+.3f}) | d_coh_resid: cv "
              f"{r['cv_carrier_resid']:+.3f} (in-sample {r['insample_carrier_resid']:+.3f})")
        print(f"  HEADLINE cos(UP_literal, d_coh_single) = {c['UP_literal']['single']:+.3f}"
              f"   nulls: shuffled-carrier 95% {r['perm95']:.3f}, random-partition 95% {r['partnull95']:.3f}")
        print(f"  UP_literal vs: resid {c['UP_literal']['resid']:+.3f}  resid2 {c['UP_literal']['resid2']:+.3f}"
              f"  orig {c['UP_literal']['orig']:+.3f}  heldout {c['UP_literal']['heldout']:+.3f}"
              f"  tokcount {c['UP_literal']['tokcount']:+.3f}  norm_single {c['UP_literal']['norm_single']:+.3f}")
        print(f"  UP_bundle  vs: single {c['UP_bundle']['single']:+.3f} (shuffled-carrier 95% {r['perm95_bundle']:.3f})"
              f"  resid {c['UP_bundle']['resid']:+.3f}  heldout {c['UP_bundle']['heldout']:+.3f}")
        others = "  ".join(f"{n_} {c[n_]['single']:+.3f}" for n_ in r["rank_order"])
        print(f"  all eight vs d_coh_single, by |cos|: {others}   (UP rank {r['up_rank']})")
        print(f"  valence: cos(VAL, d_coh_single) {c['VAL']['single']:+.3f}  cos(UP, VAL) {r['cos_UP_VAL']:+.3f}"
              f"  UP with VAL removed vs d_coh_single {r['cos_UPnoVAL_single']:+.3f}")
        print(f"  exploratory: held-out words, corr(position on UP axis, C) {r['explore_heldout_corr']:+.3f}"
              f"  permutation p {r['explore_heldout_p']:.3f}")
        print(f"  confound: cos(d_single, d_tok) {r['confound'].get('cos_single_tok', float('nan')):+.3f}"
              f"  cos(d_orig, d_tok) {r['confound'].get('cos_orig_tok', float('nan')):+.3f}"
              f"  cos(d_single, d_norm_single) {r['confound']['cos_single_normsingle']:+.3f}")


# ---- Part B analysis --------------------------------------------------------
def to_cells(rows, layers):
    """rows: list of dicts (item, triple, valence, order ('intact' or 'shufK'),
    pass (1/2), C, surp, UP_literal {L}, UP_bundle {L}, VAL {L}).
    Returns per-sentence cells with shuffles averaged."""
    by = {}
    for r in rows:
        key = (r["item"], "intact" if r["order"] == "intact" else "shuf", r["pass"])
        by.setdefault(key, []).append(r)
    items = sorted({r["item"] for r in rows})
    meta = {r["item"]: (r["triple"], r["valence"]) for r in rows}
    cells = {}
    for it in items:
        for order in ("intact", "shuf"):
            for p in (1, 2):
                rs = by[(it, order, p)]
                if order == "shuf":
                    assert len(rs) == N_SHUFFLES, (it, len(rs))
                else:
                    assert len(rs) == 1
                cell = {"C": float(np.mean([r["C"] for r in rs])),
                        "surp": float(np.mean([r["surp"] for r in rs]))}
                for k in ("UP_literal", "UP_bundle", "VAL"):
                    cell[k] = {L: float(np.mean([r[k][str(L)] if str(L) in r[k] else r[k][L]
                                                 for r in rs])) for L in layers}
                cells[(it, order, p)] = cell
    return items, meta, cells


def analyse_B(rows, layers, seed=174, axis_key="UP_literal"):
    items, meta, cells = to_cells(rows, layers)
    val_items = {v: [it for it in items if meta[it][1] == v] for v in ("happy", "sad", "neutral")}
    # pair happy/sad by triple
    tri = {}
    for it in items:
        tri.setdefault(meta[it][0], {})[meta[it][1]] = it
    triples = sorted(t for t in tri if "happy" in tri[t] and "sad" in tri[t])

    def col(itlist, order, p, key, L=None):
        if L is None:
            return np.array([cells[(it, order, p)][key] for it in itlist])
        return np.array([cells[(it, order, p)][key][L] for it in itlist])

    out = {"layers": list(layers), "axis": axis_key, "n_sentences": len(items),
           "n_triples": len(triples)}
    # G1
    g1 = d_z(col(items, "intact", 1, "C"), col(items, "shuf", 1, "C"))
    out["G1_dz"] = g1
    out["G1"] = abs(g1) >= GATE_D
    direction = 1.0 if g1 >= 0 else -1.0
    out["higher_agreement_order"] = "intact" if direction > 0 else "shuffled"
    out["C_means"] = {f"{o}_p{p}": float(col(items, o, p, "C").mean())
                      for o in ("intact", "shuf") for p in (1, 2)}
    # G3
    s1 = float(col(items, "shuf", 1, "surp").mean())
    s2 = float(col(items, "shuf", 2, "surp").mean())
    out["surp_means"] = {f"{o}_p{p}": float(col(items, o, p, "surp").mean())
                         for o in ("intact", "shuf") for p in (1, 2)}
    out["G3"] = s2 <= 0.5 * s1
    hap = [tri[t]["happy"] for t in triples]
    sad = [tri[t]["sad"] for t in triples]

    rng = np.random.default_rng(seed)
    per = {}
    for L in layers:
        r = {}
        r["G2_dz"] = d_z(col(hap, "intact", 1, "VAL", L), col(sad, "intact", 1, "VAL", L))
        for v in ("happy", "sad", "neutral"):
            r[f"T1_{v}"] = direction * d_z(col(val_items[v], "intact", 1, axis_key, L),
                                          col(val_items[v], "shuf", 1, axis_key, L))
        if direction > 0:
            r["T2_dz"] = d_z(col(sad, "intact", 1, axis_key, L), col(hap, "shuf", 1, axis_key, L))
        else:
            r["T2_dz"] = d_z(col(sad, "shuf", 1, axis_key, L), col(hap, "intact", 1, axis_key, L))
        r["T0_dz"] = d_z(col(hap, "intact", 1, axis_key, L), col(sad, "intact", 1, axis_key, L))
        # T3: text-level regression over every (sentence x order x pass) cell
        ncell = 4
        Y = np.zeros((len(items), ncell)); Cm = np.zeros_like(Y); Sm = np.zeros_like(Y)
        H = np.zeros_like(Y); S_ = np.zeros_like(Y)
        for i, it in enumerate(items):
            for j, (o, p) in enumerate((("intact", 1), ("intact", 2), ("shuf", 1), ("shuf", 2))):
                c = cells[(it, o, p)]
                Y[i, j] = c[axis_key][L]; Cm[i, j] = c["C"]; Sm[i, j] = c["surp"]
                H[i, j] = 1.0 if meta[it][1] == "happy" else 0.0
                S_[i, j] = 1.0 if meta[it][1] == "sad" else 0.0

        def fit(rowsel):
            y = zscore(Y[rowsel].ravel()); c_ = zscore(Cm[rowsel].ravel()); s_ = zscore(Sm[rowsel].ravel())
            X = np.stack([np.ones_like(y), c_, s_, H[rowsel].ravel(), S_[rowsel].ravel()], axis=1)
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            return b[1], b[2]

        bC, bS = fit(np.arange(len(items)))
        boot = np.array([fit(rng.integers(0, len(items), len(items))) for _ in range(N_BOOT)])
        r["T3_beta_C"] = float(bC); r["T3_beta_surp"] = float(bS)
        r["T3_C_ci"] = [float(np.percentile(boot[:, 0], 2.5)), float(np.percentile(boot[:, 0], 97.5))]
        r["T3_surp_ci"] = [float(np.percentile(boot[:, 1], 2.5)), float(np.percentile(boot[:, 1], 97.5))]
        r["corr_C_surp_cells"] = corrf(Cm.ravel(), Sm.ravel())
        per[L] = r
    out["per_layer"] = per
    out["G2"] = sum(per[L]["G2_dz"] >= GATE_D for L in layers) >= LAYERS_MAJ
    out["T1"] = sum(per[L]["T1_happy"] >= EFFECT_D and per[L]["T1_sad"] >= EFFECT_D
                    for L in layers) >= LAYERS_MAJ
    out["T2"] = sum(per[L]["T2_dz"] >= EFFECT_D for L in layers) >= LAYERS_MAJ
    out["T3"] = sum(per[L]["T3_C_ci"][0] > 0 for L in layers) >= LAYERS_MAJ
    out["flat"] = sum(abs(per[L]["T1_happy"]) < FLAT_D and abs(per[L]["T1_sad"]) < FLAT_D
                      for L in layers) >= LAYERS_MAJ
    return out


def verdict_B(out):
    """Frozen Part B rule (prereg + FREEZE AMENDMENT 5)."""
    if not out["G1"]:
        return "UNINFORMATIVE (G1 failed: scrambling did not move layer agreement)"
    if not out["G2"]:
        return "UNINFORMATIVE (G2 failed: valence manipulation not visible)"
    if out["T1"] and out["T3"]:
        return "STRONG_DISSOCIATION" if out["T2"] else "UP_FOLLOWS_COHERENCE"
    if out["T1"] and not out["T3"]:
        return "PERPLEXITY_NOT_COHERENCE"
    if out["flat"]:
        return "UP_FOLLOWS_VALENCE_ONLY"
    return "MIXED"


def report_B(out):
    Ls = out["layers"]
    print(f"[axis: {out['axis']}]  sentences {out['n_sentences']}  matched triples {out['n_triples']}")
    cm, sm = out["C_means"], out["surp_means"]
    print(f"  mean C_adj: intact p1 {cm['intact_p1']:+.4f}  scrambled p1 {cm['shuf_p1']:+.4f}"
          f"  intact p2 {cm['intact_p2']:+.4f}  scrambled p2 {cm['shuf_p2']:+.4f}")
    print(f"  mean surprisal (nats): intact p1 {sm['intact_p1']:.2f}  scrambled p1 {sm['shuf_p1']:.2f}"
          f"  intact p2 {sm['intact_p2']:.2f}  scrambled p2 {sm['shuf_p2']:.2f}")
    print(f"  G1 scrambling moves agreement: d_z {out['G1_dz']:+.2f} -> {'PASS' if out['G1'] else 'FAIL'}"
          f"  (higher agreement in: {out['higher_agreement_order']})")
    print(f"  G2 valence visible: {'PASS' if out['G2'] else 'FAIL'}   "
          f"G3 repeat halves scrambled surprisal: {'PASS' if out['G3'] else 'FAIL'}")
    for L in Ls:
        r = out["per_layer"][L]
        print(f"  L{L:<2} G2 d_z {r['G2_dz']:+.2f} | T0 happy-vs-sad on UP {r['T0_dz']:+.2f} | "
              f"T1 happy {r['T1_happy']:+.2f} sad {r['T1_sad']:+.2f} neutral {r['T1_neutral']:+.2f} | "
              f"T2 {r['T2_dz']:+.2f} | T3 beta_C {r['T3_beta_C']:+.3f} "
              f"[{r['T3_C_ci'][0]:+.3f},{r['T3_C_ci'][1]:+.3f}] beta_surp {r['T3_beta_surp']:+.3f} "
              f"[{r['T3_surp_ci'][0]:+.3f},{r['T3_surp_ci'][1]:+.3f}]")
    print(f"  T1 {out['T1']}  T2 {out['T2']}  T3 {out['T3']}  flat {out['flat']}")


# ---- synthetic self-tests ---------------------------------------------------
def synthetic_A(kind, n=700, dim=256, seed=0):
    """Planted worlds for the Part A verdict path.

    The carrier is deliberately MODEST (loading 0.1 against unit noise) and
    the planted pole difference in C is VERY LARGE (3 latent sd for UP). Two
    earlier versions (2026-10-02, post-freeze, pre-data) showed the frozen
    rule returning NULL / WEAK for worlds with a planted UP-vs-DOWN difference
    of about 2 sd: the random-partition null reaches |cos| ~0.5 there. So the
    frozen rule only says COUPLED for very large effects; a NULL or WEAK from
    it does not rule out a moderate one. A
    first version (2026-10-02, post-freeze, pre-data) planted a carrier that
    was the dominant direction of the whole word cloud; there, any randomly
    weighted sum of unit vectors points along that direction, the
    shuffled-carrier null becomes as large as the real cosine, and the frozen
    rule returns NULL even for a truly coupled world. That is a real
    limitation of the cosine statistic (low power when the carrier is the top
    direction), logged in PREREG_exp174.md, not something tuned away: world E
    keeps the dominant-carrier case and requires that it never read COUPLED
    by accident when nothing is coupled.

    A: C is purely token count, UP poles differ in token count
    B: UP-pole words have higher C than DOWN-pole words (UP only)
    C: every schema's poles differ in C, BALANCE most, UP second
    D: C has a carrier, no schema is coupled
    E: as D, but the carrier is the dominant direction of the cloud
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(dim, 16)))
    m, fdir, c_dir, t_dir = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    names = ["UP_literal", "S1", "S2", "S3", "S4", "S5", "BALANCE", "S7"]
    s = {n_: Q[:, 4 + k] for k, n_ in enumerate(names)}
    s_val = Q[:, 12]
    words = [f"w{i}" for i in range(n)]
    widx = {w: i for i, w in enumerate(words)}
    tok = rng.choice([1, 2, 3], size=n, p=[0.5, 0.3, 0.2]).astype(float)
    latent = rng.normal(size=n)
    f = rng.normal(size=n)
    order = np.argsort(f)
    cursor = [0]
    free = rng.permutation(n).tolist()

    def take(k):
        got = free[cursor[0]: cursor[0] + k]; cursor[0] += k
        return [words[i] for i in got]

    poles = {n_: (take(13), take(13)) for n_ in names}
    poles["UP_literal"] = (poles["UP_literal"][0] + take(1), poles["UP_literal"][1])
    bundle = (take(20), take(20))
    val = (take(8), take(8))
    shift = {n_: 0.0 for n_ in names}
    if kind == "B":
        shift["UP_literal"] = 3.0
    elif kind == "C":
        shift = {n_: 0.75 for n_ in names}; shift["UP_literal"] = 3.0; shift["BALANCE"] = 4.5
    load = np.zeros((n, dim))
    for n_ in names:
        for sign, side in ((+1, poles[n_][0]), (-1, poles[n_][1])):
            for w in side:
                load[widx[w]] += sign * 0.3 * s[n_]
                latent[widx[w]] += sign * shift[n_]
    for sign, side in ((+1, bundle[0]), (-1, bundle[1])):
        for w in side:
            load[widx[w]] += sign * 0.3 * s["UP_literal"]
            latent[widx[w]] += sign * shift["UP_literal"]
    for sign, side in ((+1, val[0]), (-1, val[1])):
        for w in side:
            load[widx[w]] += sign * 0.3 * s_val
    if kind == "A":
        for w in poles["UP_literal"][0] + bundle[0]:
            tok[widx[w]] = 3.0
        for w in poles["UP_literal"][1] + bundle[1]:
            tok[widx[w]] = 1.0
    ztok = zscore(tok)
    C = (ztok + 0.05 * rng.normal(size=n)) if kind == "A" else latent.copy()
    a = 3.0 if kind == "E" else 0.1
    R_layers = {}
    for L in LAYERS:
        noise = rng.normal(size=(n, dim)) / np.sqrt(dim)
        R = (5.0 * m[None, :] + f[:, None] * fdir[None, :] + load + noise
             + 0.3 * ztok[:, None] * t_dir[None, :])
        if kind != "A":
            R = R + a * latent[:, None] * c_dir[None, :]
        R_layers[L] = R
    lists = {"common": [words[i] for i in order[-20:]], "rare": [words[i] for i in order[:5]],
             "schemas": {n_: poles[n_] for n_ in names[1:]},
             "up_literal": poles["UP_literal"], "up_bundle": bundle, "val": val}
    return R_layers, C, tok, words, lists


def synthetic_B(kind, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    code = {"happy": 1.0, "sad": -1.0, "neutral": 0.0}
    item = 0
    for t in range(40):
        for v in ("happy", "sad", "neutral"):
            base = rng.normal(scale=0.2)
            for order in ["intact"] + [f"shuf{k}" for k in range(N_SHUFFLES)]:
                intact = order == "intact"
                for p in (1, 2):
                    C = base + (1.0 if (intact and kind != "B4") else 0.0) + rng.normal(scale=0.3)
                    if p == 1:
                        S = (3.0 if intact else 7.0) + rng.normal(scale=0.3)
                    else:
                        S = (0.5 if intact else 1.0) + rng.normal(scale=0.3)
                    row = {"item": item, "triple": t, "valence": v, "order": order, "pass": p,
                           "C": C, "surp": S, "UP_literal": {}, "UP_bundle": {}, "VAL": {}}
                    for L in LAYERS:
                        if kind in ("B1", "B4"):
                            up = 1.0 * C
                        elif kind == "B2":
                            up = 1.0 * code[v]
                        else:
                            up = -0.5 * S
                        row["UP_literal"][L] = up + rng.normal(scale=0.3)
                        row["UP_bundle"][L] = up + rng.normal(scale=0.3)
                        row["VAL"][L] = code[v] + rng.normal(scale=0.3)
                    rows.append(row)
            item += 1
    return rows


def self_test():
    print("=" * 78)
    print("SYNTHETIC SELF-TEST (real verdict paths)")
    print("=" * 78)
    ok = True
    # compute_C on planted trajectories
    rng = np.random.default_rng(1)
    n, d = 300, 64
    fixed = rng.normal(size=(n, 1, d)) + 0.1 * rng.normal(size=(n, N_LAYERS, d))
    wander = rng.normal(size=(n, N_LAYERS, d))
    R = np.concatenate([fixed, wander], axis=0) + 3.0 * rng.normal(size=(1, N_LAYERS, d))
    c, path, straight = compute_C(R)
    good = c[:n].mean() > 0.9 and abs(c[n:].mean()) < 0.1
    print(f"  compute_C: stable tokens {c[:n].mean():+.3f} (want > 0.9), wandering tokens "
          f"{c[n:].mean():+.3f} (want ~0) -> {'ok' if good else 'FAIL'}")
    ok &= good
    expect_A = {"A": {"UNINFORMATIVE", "NULL"}, "B": {"COUPLED_SPECIFIC"},
                "C": {"COUPLED_GENERIC"}, "D": {"NULL"}, "E": {"NULL", "WEAK"}}
    for kind in "ABCDE":
        Rl, C, tok, words, lists = synthetic_A(kind)
        out = analyse_A(Rl, C, tok, words, lists, LAYERS)
        v = verdict_A(out)
        cs = [out[L]["cos"]["UP_literal"]["single"] for L in LAYERS]
        cv = [out[L]["cv_carrier_single"] for L in LAYERS]
        co = [out[L]["cos"]["UP_literal"]["orig"] for L in LAYERS]
        good = v in expect_A[kind]
        print(f"  Part A world {kind}: cos(UP, single) {np.mean(cs):+.3f}  orig {np.mean(co):+.3f}"
              f"  perm95 {np.mean([out[L]['perm95'] for L in LAYERS]):.3f}"
              f"  part95 {np.mean([out[L]['partnull95'] for L in LAYERS]):.3f}"
              f"  cv carrier {np.mean(cv):+.3f}  UP rank {[out[L]['up_rank'] for L in LAYERS]}"
              f" -> {v}  (want {'/'.join(sorted(expect_A[kind]))}) {'ok' if good else 'FAIL'}")
        ok &= good
    expect_B = {"B1": "STRONG_DISSOCIATION", "B2": "UP_FOLLOWS_VALENCE_ONLY",
                "B3": "PERPLEXITY_NOT_COHERENCE", "B4": "UNINFORMATIVE"}
    for kind in ("B1", "B2", "B3", "B4"):
        out = analyse_B(synthetic_B(kind), LAYERS)
        v = verdict_B(out)
        good = v.startswith(expect_B[kind])
        print(f"  Part B world {kind}: G1 d_z {out['G1_dz']:+.2f}  T1 {out['T1']}  T2 {out['T2']}"
              f"  T3 {out['T3']} -> {v}  (want {expect_B[kind]}) {'ok' if good else 'FAIL'}")
        ok &= good
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return ok


# ---- model runs -------------------------------------------------------------
def collect_words(model, words, hook_all):
    import torch
    cache_path = f"{ROOT}/exp174_cache_words.npz"
    R = np.zeros((len(words), N_LAYERS, model.cfg.d_model), np.float32)
    tokc = np.zeros(len(words), int)
    done = 0
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        if list(z["words"]) == list(words):
            done = int(z["done"]); R[:done] = z["R"][:done]; tokc[:done] = z["tokc"][:done]
            print(f"  resuming word residuals from cache: {done}/{len(words)}")
    for k in range(done, len(words)):
        toks = model.to_tokens(words[k])
        tokc[k] = int(toks.shape[1] - 1)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=hook_all)
        R[k] = np.stack([cache[h][0, -1, :].float().cpu().numpy() for h in hook_all])
        if (k + 1) % 100 == 0 or k + 1 == len(words):
            np.savez(cache_path, R=R, tokc=tokc, words=np.array(words, dtype=object), done=k + 1)
            print(f"  {k + 1}/{len(words)} (saved)")
    return R.astype(np.float64), tokc


def replication_gate(R, tokc, words):
    """exp170's cos(BALANCE, d_norm_orig), on exp170's exact word set."""
    from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML
    set170 = set(COMMON + RARE)
    for pairs in SUFFIX_PAIRS.values():
        for b, i in pairs:
            set170.add(b); set170.add(i)
    for s_ in SCHEMA_NAMES:
        for p, n_ in LAKOFF_SCHEMAS_MML[s_]:
            set170.add(p); set170.add(n_)
    w170 = sorted(set170)
    idx = {w: i for i, w in enumerate(words)}
    rows = [idx[w] for w in w170]
    print(f"REPLICATION GATE on exp170's word set ({len(w170)} words; exp170 had 489, 219 single-token;"
          f" here single-token {int((tokc[rows] == 1).sum())})")
    ok = len(w170) == 489
    loc = {w: i for i, w in enumerate(w170)}
    for L in LAYERS:
        arr = R[rows, L, :]
        aniso = arr.mean(axis=0); aniso /= np.linalg.norm(aniso)
        ma = lambda ws: arr[[loc[w] for w in ws]].mean(axis=0)
        freq = ma(COMMON) - ma(RARE); freq /= np.linalg.norm(freq)
        fo = freq - (freq @ aniso) * aniso; fo /= np.linalg.norm(fo)

        def strip(d):
            d = d - (d @ aniso) * aniso
            d = d - (d @ fo) * fo
            return d / np.linalg.norm(d)

        norms = np.linalg.norm(arr, axis=1)
        units = arr / norms[:, None]
        zn = (norms - norms.mean()) / norms.std()
        d = zn @ units
        d_orig = strip(d / np.linalg.norm(d))
        pairs = LAKOFF_SCHEMAS_MML["BALANCE"]
        pos = sorted(set(p[0] for p in pairs)); neg = sorted(set(p[1] for p in pairs))
        raw = ma(pos) - ma(neg)
        bal = strip(raw / np.linalg.norm(raw))
        c = float(bal @ d_orig)
        hit = abs(c - REPL_TARGET[L]) <= REPL_TOL
        ok &= hit
        print(f"  L{L:<2} cos(BALANCE, d_norm_orig) {c:+.3f}  target {REPL_TARGET[L]:+.3f}"
              f"  {'ok' if hit else 'OUT OF TOLERANCE'}")
    print("REPLICATION GATE", "PASS" if ok else "FAIL")
    return ok


def run_part_B(model, hook_all, axes):
    import torch
    stim = json.loads(_stim_blob)
    assert stim["n_shuffles"] == N_SHUFFLES
    tk = model.tokenizer
    bos = int(model.to_tokens("a")[0, 0])
    texts = []   # (key, item index, triple, valence, order, ids)
    for it, item in enumerate(stim["items"]):
        words = item["sentence"].split(" ")
        groups = [tk.encode(" " + w, add_special_tokens=False) for w in words]
        intact = [t for g in groups for t in g]
        assert len(intact) == item["n_tokens"], "tokenisation drifted from the frozen stimuli"
        texts.append((f"{it}|intact", it, item["triple"], item["valence"], "intact", intact))
        for k, perm in enumerate(item["shuffles"]):
            ids = [t for j in perm for t in groups[j]]
            assert sorted(ids) == sorted(intact), "scramble changed the token multiset"
            assert ids != intact
            texts.append((f"{it}|shuf{k}", it, item["triple"], item["valence"], f"shuf{k}", ids))
    print(f"Part B: {len(texts)} texts ({len(stim['items'])} sentences x (1 intact + {N_SHUFFLES} scrambles)),"
          " each presented twice in a row; token-identity assertions passed")

    def forward(ids):
        toks = torch.tensor([[bos] + ids + ids], device=model.cfg.device)
        with torch.no_grad():
            logits, cache = model.run_with_cache(toks, names_filter=hook_all)
        R = np.stack([cache[h][0].float().cpu().numpy() for h in hook_all]).astype(np.float64)
        logp = torch.log_softmax(logits[0].float(), dim=-1).cpu().numpy()
        return toks[0].cpu().numpy(), R, logp

    mu_path = f"{ROOT}/exp174_cache_mu.npz"
    if os.path.exists(mu_path) and str(np.load(mu_path)["sha"]) == STIM_SHA:
        mu = np.load(mu_path)["mu"]
        print("  centring mean loaded from cache")
    else:
        tot = np.zeros((N_LAYERS, model.cfg.d_model)); cnt = 0
        for n_, (_, _, _, _, _, ids) in enumerate(texts):
            _, R, _ = forward(ids)
            tot += R[:, 1:, :].sum(axis=1); cnt += R.shape[1] - 1
            if (n_ + 1) % 180 == 0:
                print(f"  centring pass {n_ + 1}/{len(texts)}")
        mu = tot / cnt
        np.savez(mu_path, mu=mu, sha=STIM_SHA, n_tokens=cnt)
        print(f"  centring mean over {cnt} stimulus tokens (saved)")

    rows_path = f"{ROOT}/exp174_rows.jsonl"
    rows, have = [], set()
    if os.path.exists(rows_path):
        for line in open(rows_path):
            r = json.loads(line)
            rows.append(r); have.add((r["key"], r["pass"]))
        print(f"  resuming: {len(rows)} rows already on disk")
    with open(rows_path, "a") as fh:
        for n_, (key, it, triple, valence, order, ids) in enumerate(texts):
            if (key, 1) in have and (key, 2) in have:
                continue
            toks, R, logp = forward(ids)
            T = R.shape[1]
            Rt = np.transpose(R[:, 1:, :], (1, 0, 2))          # [tokens, layers, d]
            c_adj, path, straight = compute_C(Rt, mu=mu)
            surp = np.array([-logp[p - 1, toks[p]] for p in range(1, T)])
            n_tok = len(ids)
            for p, sl in ((1, slice(0, n_tok)), (2, slice(n_tok, 2 * n_tok))):
                row = {"key": key, "item": it, "triple": triple, "valence": valence,
                       "order": order, "pass": p, "n_tokens": n_tok,
                       "C": float(c_adj[sl].mean()), "path": float(path[sl].mean()),
                       "straight": float(straight[sl].mean()), "surp": float(surp[sl].mean()),
                       "UP_literal": {}, "UP_bundle": {}, "VAL": {}, "norm": {}}
                for L in LAYERS:
                    X = Rt[sl, L, :]
                    nr = np.linalg.norm(X, axis=1)
                    U = X / nr[:, None]
                    row["norm"][str(L)] = float(nr.mean())
                    for name in ("UP_literal", "UP_bundle", "VAL"):
                        row[name][str(L)] = float((U @ axes[L][name]).mean())
                fh.write(json.dumps(row) + "\n"); fh.flush()
                rows.append(row)
            if (n_ + 1) % 120 == 0:
                print(f"  texts {n_ + 1}/{len(texts)} (rows saved as they finish)")
    return rows


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if k != "_axes"}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


if __name__ == "__main__":
    print("exp174 — is UP grounded in layer agreement?  (prereg frozen at commit f2e30a4)")
    print(f"stimuli sha256 {STIM_SHA} matches prereg; rule constants match prereg")
    if not self_test():
        print("Self-test failed: NOT running the model.")
        sys.exit(1)
    if "--selftest-only" in sys.argv:
        sys.exit(0)

    import torch
    from wordfreq import zipf_frequency
    sys.path.insert(0, ROOT)
    from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML
    from transformer_lens import HookedTransformer

    device = "mps"
    print("\nLoading Pythia 410M...")
    model = HookedTransformer.from_pretrained("pythia-410m", device=device)
    model.eval()
    assert model.cfg.n_layers == N_LAYERS
    hook_all = [f"blocks.{L}.hook_resid_post" for L in range(N_LAYERS)]

    all_words = set(COMMON + RARE + LAKOFF_UP + LAKOFF_DOWN + VALENCE_POS + VALENCE_NEG)
    for pairs in SUFFIX_PAIRS.values():
        for b, i in pairs:
            all_words.add(b); all_words.add(i)
    for s_ in SCHEMA_NAMES:
        for p, n_ in LAKOFF_SCHEMAS_MML[s_]:
            all_words.add(p); all_words.add(n_)
    all_words = sorted(all_words)
    print(f"Collecting residuals for {len(all_words)} words at all {N_LAYERS} layers...")
    R, tokc = collect_words(model, all_words, hook_all)

    print("\n" + "=" * 78)
    if not replication_gate(R, tokc, all_words):
        print("STOP: protocol drift, not a finding. Nothing below was computed.")
        sys.exit(2)

    print("\n" + "=" * 78)
    print("PART A — word-level coupling of UP to layer agreement")
    print("=" * 78)
    C, path, straight = compute_C(R)

    def poles(name):
        pairs = LAKOFF_SCHEMAS_MML[name]
        return (sorted(set(p[0] for p in pairs)), sorted(set(p[1] for p in pairs)))

    lists = {"common": COMMON, "rare": RARE,
             "schemas": {n_: poles(n_) for n_ in SCHEMA_NAMES[1:]},
             "up_literal": (LAKOFF_UP, LAKOFF_DOWN), "up_bundle": poles("UP-DOWN"),
             "val": (VALENCE_POS, VALENCE_NEG), "path": path,
             "zipf": np.array([zipf_frequency(w, "en") for w in all_words])}
    R_layers = {L: R[:, L, :] for L in LAYERS}
    outA = analyse_A(R_layers, C, tokc, all_words, lists, LAYERS)
    report_A(outA)
    vA = verdict_A(outA)
    print(f"\nPART A VERDICT (frozen rule, UP_literal vs d_coh_single): {vA}")
    # secondary carrier, reported whatever it says, never promoted
    outA2 = analyse_A(R_layers, straight, tokc, all_words, lists, LAYERS)
    print("SECONDARY carrier C_straight (non-binding): cos(UP_literal, d_single) by layer: "
          + "  ".join(f"L{L} {outA2[L]['cos']['UP_literal']['single']:+.3f} (perm95 {outA2[L]['perm95']:.3f},"
                      f" cv {outA2[L]['cv_carrier_single']:+.2f})" for L in LAYERS))
    axes = outA["_axes"]
    results = {"partA": _jsonable(outA), "partA_verdict": vA,
               "partA_straight": _jsonable(outA2), "stimuli_sha": STIM_SHA}
    json.dump(results, open(f"{ROOT}/exp174_results.json", "w"), indent=1)
    print("  (Part A results saved to exp174_results.json)")

    print("\n" + "=" * 78)
    print("PART B — the happy / incoherent dissociation")
    print("=" * 78)
    rows = run_part_B(model, hook_all, axes)
    outB = analyse_B(rows, LAYERS, axis_key="UP_literal")
    report_B(outB)
    vB = verdict_B(outB)
    print(f"\nPART B VERDICT (frozen rule, UP_literal): {vB}")
    if not outB["G3"]:
        print("  NOTE: G3 failed, so the perplexity control rests on the regression alone (weaker).")
    outB2 = analyse_B(rows, LAYERS, axis_key="UP_bundle")
    print("\nReported beside it, never substituting:")
    report_B(outB2)
    print(f"  (UP_bundle would read: {verdict_B(outB2)})")
    results.update({"partB": _jsonable(outB), "partB_verdict": vB,
                    "partB_bundle": _jsonable(outB2), "partB_bundle_verdict": verdict_B(outB2)})
    json.dump(results, open(f"{ROOT}/exp174_results.json", "w"), indent=1)
    print("\nAll results saved to exp174_results.json; per-text rows in exp174_rows.jsonl")
