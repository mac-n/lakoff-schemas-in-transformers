"""
exp175_holding_up.py — is UP grounded in "holding against gravity / effecting"?

Frozen rules: PREREG_exp175.md (commit 1cbdb60, 2026-10-02 17:08 IST).
Written AFTER the freeze.

The measure: at each layer, let one token fall (replace its state with the
layer's average state) and see how much the model's output changes. Summed
over all 24 layers. Two readouts: the next word (E_next) and the rest of the
sentence after it (E_reach, PRIMARY).

Part A: do spatial UP words leave a bigger hole than spatial DOWN words?
Part B: fresh happy / sad / neutral sentences, intact and scrambled. Does the
UP readout follow the size of the hole, or follow happy?

Order: frozen-constant assertions -> synthetic self-tests (real verdict paths)
-> model load -> mechanics checks (STOP on failure) -> Part A -> Part B.
Results are written to disk as they are produced; the run resumes.

Usage:
  ./lakoff/bin/python3 exp175_holding_up.py --selftest-only
  ./lakoff/bin/python3 exp175_holding_up.py
"""

import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = "/Users/macn/Documents/embeddingexp"

D_HI = 0.50
D_LO = 0.20
ALPHA = 0.05
N_PERM = 10000
GATE_D = 0.50
EFFECT_D = 0.30
FLAT_D = 0.20
LAYERS_MAJ = 3
N_BOOT = 2000
N_SHUFFLES = 3
IDENT_TOL = 0.001

LAYERS = [4, 8, 12, 16, 20]
N_LAYERS = 24
CHUNK = 72

PREREG = open(f"{ROOT}/PREREG_exp175.md").read()
for _name, _val in [("D_HI", D_HI), ("D_LO", D_LO), ("ALPHA", ALPHA), ("N_PERM", N_PERM),
                    ("GATE_D", GATE_D), ("EFFECT_D", EFFECT_D), ("FLAT_D", FLAT_D),
                    ("LAYERS_MAJ", LAYERS_MAJ), ("N_BOOT", N_BOOT),
                    ("N_SHUFFLES", N_SHUFFLES), ("IDENT_TOL", IDENT_TOL)]:
    _m = re.search(rf"\b{_name} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_val), f"{_name} drifted from prereg"
_stim_blob = open(f"{ROOT}/exp175_stimuli.json", "rb").read()
STIM_SHA = hashlib.sha256(_stim_blob).hexdigest()
assert STIM_SHA in PREREG, "stimuli file does not match the sha256 in the prereg"
STIM = json.loads(_stim_blob)
SCHEMA_NAMES = list(STIM["words"]["schemas"].keys())
assert len(SCHEMA_NAMES) == 7 and STIM["n_shuffles"] == N_SHUFFLES

# exp141's literal lists, for the secondary bare-word axis of exp174
LAKOFF_UP = ["up", "rise", "rose", "rising", "ascend", "raise", "climb", "lift",
             "above", "over", "top", "high", "higher", "upward"]
LAKOFF_DOWN = ["down", "fall", "fell", "falling", "descend", "drop", "sink",
               "below", "under", "bottom", "low", "lower", "downward"]


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
    d = np.asarray(x, float) - np.asarray(y, float)
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 0 else 0.0


def cohen_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def perm_p(a, b, rng):
    """Two-sided label-shuffle p for the difference in means."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    both = np.concatenate([a, b]); k = len(a)
    obs = abs(a.mean() - b.mean())
    X = rng.permuted(np.tile(both, (N_PERM, 1)), axis=1)
    diff = np.abs(X[:, :k].mean(axis=1) - X[:, k:].mean(axis=1))
    return float((np.sum(diff >= obs - 1e-15) + 1) / (N_PERM + 1))


def residualise(y, covs):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in covs])
    b, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    return np.asarray(y, float) - X @ b


# ---- Part A analysis --------------------------------------------------------
def analyse_A(value, value_next, dist, zipf, lists, seed=175):
    """value / value_next / dist / zipf: dicts word -> number (ln E_reach,
    ln E_next, mean ln distance from the average state, zipf frequency).
    lists: {"UP": (pos, neg), schema name: (pos, neg), ..., "VAL": (pos, neg)}."""
    rng = np.random.default_rng(seed)
    words = sorted({w for pos, neg in lists.values() for w in pos + neg})
    ix = {w: i for i, w in enumerate(words)}
    v = np.array([value[w] for w in words])
    vn = np.array([value_next[w] for w in words])
    z = np.array([zipf[w] for w in words])
    dd = np.array([dist[w] for w in words])
    versions = {"raw": v, "ctrl": residualise(v, [z]), "ctrl_dist": residualise(v, [z, dd]),
                "next_raw": vn, "next_ctrl": residualise(vn, [z])}
    out = {"n_words": len(words), "contrasts": {},
           "corr_value_zipf": corrf(v, z), "corr_value_dist": corrf(v, dd),
           "value_mean": float(v.mean()), "value_sd": float(v.std())}
    for name, (pos, neg) in lists.items():
        pi = [ix[w] for w in pos]; ni = [ix[w] for w in neg]
        c = {"n_pos": len(pi), "n_neg": len(ni)}
        for key, arr in versions.items():
            c[f"d_{key}"] = cohen_d(arr[pi], arr[ni])
            c[f"p_{key}"] = perm_p(arr[pi], arr[ni], rng)
        c["mean_pos_raw"] = float(v[pi].mean()); c["mean_neg_raw"] = float(v[ni].mean())
        c["zipf_pos"] = float(z[pi].mean()); c["zipf_neg"] = float(z[ni].mean())
        out["contrasts"][name] = c
    eight = ["UP"] + [n for n in lists if n not in ("UP", "VAL")]
    order = sorted(eight, key=lambda n: -abs(out["contrasts"][n]["d_ctrl"]))
    out["rank_order"] = order
    out["up_rank"] = order.index("UP") + 1
    return out


def verdict_A(out):
    """Frozen Part A rule (PREREG_exp175.md)."""
    c = out["contrasts"]["UP"]
    sig_raw = c["p_raw"] < ALPHA; sig_ctrl = c["p_ctrl"] < ALPHA
    if c["d_raw"] >= D_HI and c["d_ctrl"] >= D_HI and sig_raw and sig_ctrl:
        return "HOLDS_SPECIFIC" if out["up_rank"] == 1 else "HOLDS_GENERIC"
    if c["d_raw"] <= -D_HI and c["d_ctrl"] <= -D_HI and sig_raw and sig_ctrl:
        return "WRONG_SIGN"
    if (not sig_ctrl) or abs(c["d_ctrl"]) < D_LO:
        return "NULL"
    return "WEAK"


def report_A(out):
    print(f"analysis words {out['n_words']}  word value = mean over 3 frames of ln E_reach:"
          f" mean {out['value_mean']:+.3f} sd {out['value_sd']:.3f}")
    print(f"  what else the value tracks: corr with word frequency {out['corr_value_zipf']:+.3f},"
          f" with distance from the average state {out['corr_value_dist']:+.3f}")
    c = out["contrasts"]["UP"]
    print(f"BINDING: spatial UP ({c['n_pos']} words) minus spatial DOWN ({c['n_neg']} words)")
    print(f"  raw ln E_reach: UP {c['mean_pos_raw']:+.3f}  DOWN {c['mean_neg_raw']:+.3f}"
          f"  d {c['d_raw']:+.2f}  p {c['p_raw']:.4f}")
    print(f"  frequency-controlled: d {c['d_ctrl']:+.2f}  p {c['p_ctrl']:.4f}"
          f"   (mean zipf UP {c['zipf_pos']:.2f}, DOWN {c['zipf_neg']:.2f})")
    print(f"  also controlling distance from average: d {c['d_ctrl_dist']:+.2f}  p {c['p_ctrl_dist']:.4f}")
    print(f"  next-word readout instead of reach: raw d {c['d_next_raw']:+.2f} p {c['p_next_raw']:.4f}"
          f" | controlled d {c['d_next_ctrl']:+.2f} p {c['p_next_ctrl']:.4f}")
    print("  every contrast, frequency-controlled (positive pole minus negative pole):")
    for name in out["rank_order"] + ["VAL"]:
        k = out["contrasts"][name]
        print(f"    {name:18s} n {k['n_pos']:>2}/{k['n_neg']:<2}  d_ctrl {k['d_ctrl']:+.2f}  p {k['p_ctrl']:.4f}"
              f"   raw d {k['d_raw']:+.2f}")
    print(f"  UP rank among the eight by |d_ctrl|: {out['up_rank']}")


# ---- Part B analysis --------------------------------------------------------
def to_cells(rows, layers, axis_key):
    by = {}
    for r in rows:
        by.setdefault((r["item"], "intact" if r["order"] == "intact" else "shuf"), []).append(r)
    items = sorted({r["item"] for r in rows})
    meta = {r["item"]: (r["triple"], r["valence"]) for r in rows}
    cells = {}
    for it in items:
        for order in ("intact", "shuf"):
            rs = by[(it, order)]
            assert len(rs) == (1 if order == "intact" else N_SHUFFLES), (it, order, len(rs))
            cell = {"E": float(np.mean([r["E_reach"] for r in rs])),
                    "E_next": float(np.mean([r["E_next"] for r in rs])),
                    "surp": float(np.mean([r["surp"] for r in rs]))}
            for k in (axis_key, "VAL"):
                cell[k] = {L: float(np.mean([r[k][str(L)] for r in rs])) for L in layers}
            cells[(it, order)] = cell
    return items, meta, cells


def analyse_B(rows, layers, axis_key="UP_spatial", seed=175):
    items, meta, cells = to_cells(rows, layers, axis_key)
    val_items = {v: [it for it in items if meta[it][1] == v] for v in ("happy", "sad", "neutral")}
    tri = {}
    for it in items:
        tri.setdefault(meta[it][0], {})[meta[it][1]] = it
    triples = sorted(t for t in tri if "happy" in tri[t] and "sad" in tri[t])
    hap = [tri[t]["happy"] for t in triples]; sad = [tri[t]["sad"] for t in triples]

    def col(itlist, order, key, L=None):
        if L is None:
            return np.array([cells[(it, order)][key] for it in itlist])
        return np.array([cells[(it, order)][key][L] for it in itlist])

    out = {"layers": list(layers), "axis": axis_key, "n_sentences": len(items), "n_triples": len(triples)}
    g1 = d_z(col(items, "intact", "E"), col(items, "shuf", "E"))
    out["G1_dz"] = g1; out["G1"] = abs(g1) >= GATE_D
    direction = 1.0 if g1 >= 0 else -1.0
    out["bigger_hole_order"] = "intact" if direction > 0 else "scrambled"
    out["E_means"] = {o: float(col(items, o, "E").mean()) for o in ("intact", "shuf")}
    out["Enext_means"] = {o: float(col(items, o, "E_next").mean()) for o in ("intact", "shuf")}
    out["surp_means"] = {o: float(col(items, o, "surp").mean()) for o in ("intact", "shuf")}
    out["G1_next_dz"] = d_z(col(items, "intact", "E_next"), col(items, "shuf", "E_next"))
    out["T4_dz"] = d_z(col(hap, "intact", "E"), col(sad, "intact", "E"))
    out["corr_E_surp_cells"] = corrf(np.concatenate([col(items, o, "E") for o in ("intact", "shuf")]),
                                     np.concatenate([col(items, o, "surp") for o in ("intact", "shuf")]))
    rng = np.random.default_rng(seed)
    per = {}
    for L in layers:
        r = {"G2_dz": d_z(col(hap, "intact", "VAL", L), col(sad, "intact", "VAL", L))}
        for v in ("happy", "sad", "neutral"):
            r[f"T1_{v}"] = direction * d_z(col(val_items[v], "intact", axis_key, L),
                                          col(val_items[v], "shuf", axis_key, L))
        if direction > 0:
            r["T2_dz"] = d_z(col(sad, "intact", axis_key, L), col(hap, "shuf", axis_key, L))
        else:
            r["T2_dz"] = d_z(col(sad, "shuf", axis_key, L), col(hap, "intact", axis_key, L))
        r["T2b_dz"] = d_z(col(sad, "intact", axis_key, L), col(hap, "shuf", axis_key, L))
        r["T0_dz"] = d_z(col(hap, "intact", axis_key, L), col(sad, "intact", axis_key, L))
        n = len(items)
        Y = np.zeros((n, 2)); Em = np.zeros((n, 2)); Sm = np.zeros((n, 2))
        H = np.zeros((n, 2)); Sd = np.zeros((n, 2))
        for i, it in enumerate(items):
            for j, o in enumerate(("intact", "shuf")):
                c = cells[(it, o)]
                Y[i, j] = c[axis_key][L]; Em[i, j] = c["E"]; Sm[i, j] = c["surp"]
                H[i, j] = 1.0 if meta[it][1] == "happy" else 0.0
                Sd[i, j] = 1.0 if meta[it][1] == "sad" else 0.0

        def fit(sel):
            y = zscore(Y[sel].ravel())
            X = np.column_stack([np.ones_like(y), zscore(Em[sel].ravel()), zscore(Sm[sel].ravel()),
                                 H[sel].ravel(), Sd[sel].ravel()])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            return b[1], b[2]

        bE, bS = fit(np.arange(n))
        boot = np.array([fit(rng.integers(0, n, n)) for _ in range(N_BOOT)])
        r["T3_beta_E"] = float(bE); r["T3_beta_surp"] = float(bS)
        r["T3_E_ci"] = [float(np.percentile(boot[:, 0], 2.5)), float(np.percentile(boot[:, 0], 97.5))]
        r["T3_surp_ci"] = [float(np.percentile(boot[:, 1], 2.5)), float(np.percentile(boot[:, 1], 97.5))]
        per[L] = r
    out["per_layer"] = per
    out["G2"] = sum(per[L]["G2_dz"] >= GATE_D for L in layers) >= LAYERS_MAJ
    out["T1"] = sum(per[L]["T1_happy"] >= EFFECT_D and per[L]["T1_sad"] >= EFFECT_D
                    for L in layers) >= LAYERS_MAJ
    out["T2"] = sum(per[L]["T2_dz"] >= EFFECT_D for L in layers) >= LAYERS_MAJ
    out["T3"] = sum(per[L]["T3_E_ci"][0] > 0 for L in layers) >= LAYERS_MAJ
    out["flat"] = sum(abs(per[L]["T1_happy"]) < FLAT_D and abs(per[L]["T1_sad"]) < FLAT_D
                      for L in layers) >= LAYERS_MAJ
    return out


def verdict_B(out):
    """Frozen Part B rule (PREREG_exp175.md)."""
    if not out["G1"]:
        return "UNINFORMATIVE (G1 failed: scrambling did not move the size of the hole)"
    if not out["G2"]:
        return "UNINFORMATIVE (G2 failed: valence manipulation not visible)"
    if out["T1"] and out["T3"]:
        return "STRONG_DISSOCIATION" if out["T2"] else "UP_FOLLOWS_EFFECT"
    if out["T1"] and not out["T3"]:
        return "PERPLEXITY_NOT_EFFECT"
    if out["flat"]:
        return "UP_FOLLOWS_VALENCE_ONLY"
    return "MIXED"


def report_B(out):
    print(f"[axis: {out['axis']}]  sentences {out['n_sentences']}  matched triples {out['n_triples']}")
    print(f"  mean ln E_reach: intact {out['E_means']['intact']:+.3f}  scrambled {out['E_means']['shuf']:+.3f}"
          f" | ln E_next: intact {out['Enext_means']['intact']:+.3f}  scrambled {out['Enext_means']['shuf']:+.3f}"
          f" | surprisal: intact {out['surp_means']['intact']:.2f}  scrambled {out['surp_means']['shuf']:.2f}")
    print(f"  G1 scrambling moves the hole (reach): d_z {out['G1_dz']:+.2f} -> {'PASS' if out['G1'] else 'FAIL'}"
          f"  (bigger hole in: {out['bigger_hole_order']})   [next-word version d_z {out['G1_next_dz']:+.2f}]")
    print(f"  G2 valence visible: {'PASS' if out['G2'] else 'FAIL'}"
          f"   T4 does valence move the hole (happy vs sad, intact): d_z {out['T4_dz']:+.2f}"
          f"   corr(E, surprisal) over cells {out['corr_E_surp_cells']:+.2f}")
    for L in out["layers"]:
        r = out["per_layer"][L]
        print(f"  L{L:<2} G2 {r['G2_dz']:+.2f} | T0 happy-vs-sad on UP {r['T0_dz']:+.2f} | "
              f"T1 happy {r['T1_happy']:+.2f} sad {r['T1_sad']:+.2f} neutral {r['T1_neutral']:+.2f} | "
              f"T2 {r['T2_dz']:+.2f} | T2b sad-intact vs happy-scrambled {r['T2b_dz']:+.2f} | "
              f"T3 beta_E {r['T3_beta_E']:+.3f} [{r['T3_E_ci'][0]:+.3f},{r['T3_E_ci'][1]:+.3f}]"
              f" beta_surp {r['T3_beta_surp']:+.3f} [{r['T3_surp_ci'][0]:+.3f},{r['T3_surp_ci'][1]:+.3f}]")
    print(f"  T1 {out['T1']}  T2 {out['T2']}  T3 {out['T3']}  flat {out['flat']}")


# ---- synthetic self-tests ---------------------------------------------------
def synthetic_A(kind, seed=0):
    rng = np.random.default_rng(seed)
    names = ["UP", "S1", "S2", "S3", "S4", "S5", "BALANCE", "S7", "VAL"]
    sizes = {"UP": (35, 39), "VAL": (22, 19)}
    lists, value, vnext, dist, zipf = {}, {}, {}, {}, {}
    k = 0
    for n_ in names:
        a, b = sizes.get(n_, (13, 12))
        pos = [f"w{k + i}" for i in range(a)]; k += a
        neg = [f"w{k + i}" for i in range(b)]; k += b
        lists[n_] = (pos, neg)
        shift = 0.0
        if kind == "A1" and n_ == "UP":
            shift = 0.5
        if kind == "A2":
            shift = {"UP": 0.5, "BALANCE": 1.2}.get(n_, 0.4)
        for sign, side in ((+1, pos), (-1, neg)):
            for w in side:
                zipf[w] = rng.normal(4.5, 1.0) + (0.75 * sign if (kind == "A3" and n_ == "UP") else 0.0)
                dist[w] = rng.normal()
                base = 0.5 * rng.normal() + sign * shift
                if kind == "A3":
                    base += 0.8 * zipf[w]
                value[w] = base
                vnext[w] = base + 0.2 * rng.normal()
    return value, vnext, dist, zipf, lists


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
                E = base + (1.0 if (intact and kind != "B4") else 0.0) + rng.normal(scale=0.3)
                S = (3.0 if intact else 7.0) + rng.normal(scale=0.6)
                row = {"item": item, "triple": t, "valence": v, "order": order,
                       "E_reach": E, "E_next": E, "surp": S, "UP_spatial": {}, "VAL": {}}
                for L in LAYERS:
                    up = {"B1": 1.0 * E, "B4": 1.0 * E, "B2": 1.0 * code[v]}.get(kind, -0.5 * S)
                    row["UP_spatial"][str(L)] = up + rng.normal(scale=0.3)
                    row["VAL"][str(L)] = code[v] + rng.normal(scale=0.3)
                rows.append(row)
            item += 1
    return rows


def self_test():
    print("=" * 78)
    print("SYNTHETIC SELF-TEST (real verdict paths)")
    print("=" * 78)
    ok = True
    expect_A = {"A1": {"HOLDS_SPECIFIC"}, "A2": {"HOLDS_GENERIC"},
                "A3": {"NULL", "WEAK"}, "A4": {"NULL"}}
    for kind in ("A1", "A2", "A3", "A4"):
        out = analyse_A(*synthetic_A(kind))
        v = verdict_A(out); c = out["contrasts"]["UP"]
        good = v in expect_A[kind]
        print(f"  Part A world {kind}: d_raw {c['d_raw']:+.2f} (p {c['p_raw']:.3f})  d_ctrl {c['d_ctrl']:+.2f}"
              f" (p {c['p_ctrl']:.3f})  UP rank {out['up_rank']} -> {v}"
              f"  (want {'/'.join(sorted(expect_A[kind]))}) {'ok' if good else 'FAIL'}")
        ok &= good
    expect_B = {"B1": "STRONG_DISSOCIATION", "B2": "UP_FOLLOWS_VALENCE_ONLY",
                "B3": "PERPLEXITY_NOT_EFFECT", "B4": "UNINFORMATIVE"}
    for kind in ("B1", "B2", "B3", "B4"):
        out = analyse_B(synthetic_B(kind), LAYERS)
        v = verdict_B(out)
        good = v.startswith(expect_B[kind])
        print(f"  Part B world {kind}: G1 d_z {out['G1_dz']:+.2f}  T1 {out['T1']}  T2 {out['T2']}"
              f"  T3 {out['T3']} -> {v}  (want {expect_B[kind]}) {'ok' if good else 'FAIL'}")
        ok &= good
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return ok


# ---- model machinery --------------------------------------------------------
class Runner:
    def __init__(self):
        import torch
        from transformer_lens import HookedTransformer
        self.torch = torch
        print("\nLoading Pythia 410M...")
        self.model = HookedTransformer.from_pretrained("pythia-410m", device="mps")
        self.model.eval()
        assert self.model.cfg.n_layers == N_LAYERS
        self.dev = self.model.cfg.device
        self.hooks_all = [f"blocks.{L}.hook_resid_post" for L in range(N_LAYERS)]
        self.tk = self.model.tokenizer
        # POST-FREEZE FIX (pre-data, 2026-10-02): this TransformerLens build has
        # default_prepend_bos = False for Pythia, so model.to_tokens("a")[0, 0]
        # is the token "a", NOT a start token. exp174 used that call and so
        # prefixed every text with "a". The prereg says "every text is BOS +
        # tokens"; take the real start token from the tokenizer.
        self.bos = int(self.tk.bos_token_id)
        assert self.bos == 0 and self.tk.decode([self.bos]) == "<|endoftext|>"
        assert int(self.model.to_tokens("a")[0, 0]) != self.bos, "to_tokens now prepends BOS: re-check"

    def groups(self, words):
        return [self.tk.encode(" " + w, add_special_tokens=False) for w in words]

    def clean(self, ids):
        torch = self.torch
        toks = torch.tensor([[self.bos] + list(ids)], device=self.dev)
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(toks, names_filter=self.hooks_all)
        resid = torch.stack([cache[h][0] for h in self.hooks_all])          # [24, T1, d]
        logp = torch.log_softmax(logits[0].float(), dim=-1)                  # [T1, V]
        return toks, resid, logp

    def drop_kl(self, toks, jobs, source, clean_logp):
        """jobs: list of (position, layer). source[L] is either a [d] vector
        (the layer average) or a [T1, d] tensor (per-position replacement).
        Returns KL(clean || dropped) at every position: [len(jobs), T1]."""
        torch = self.torch
        T1 = toks.shape[1]
        out = np.zeros((len(jobs), T1), np.float64)
        clean_p = clean_logp.exp()
        for s in range(0, len(jobs), CHUNK):
            part = jobs[s: s + CHUNK]
            batch = toks.repeat(len(part), 1)
            byL = {}
            for i, (p, L) in enumerate(part):
                byL.setdefault(L, []).append((i, p))
            fwd = []
            for L, lst in byL.items():
                bi = torch.tensor([i for i, _ in lst], device=self.dev)
                pi = torch.tensor([p for _, p in lst], device=self.dev)

                def hook(resid, hook, bi=bi, pi=pi, L=L):
                    src = source[L]
                    resid[bi, pi, :] = (src if src.dim() == 1 else src[pi]).to(resid.dtype)
                    return resid
                fwd.append((f"blocks.{L}.hook_resid_post", hook))
            with torch.no_grad():
                logits = self.model.run_with_hooks(batch, fwd_hooks=fwd)
            logp = torch.log_softmax(logits.float(), dim=-1)
            kl = (clean_p[None] * (clean_logp[None] - logp)).sum(dim=-1)
            out[s: s + len(part)] = kl.float().cpu().numpy()
            del logits, logp, kl
        return out

    def holes(self, toks, clean_logp, mu_t, positions):
        """E_next and E_reach (natural units, summed over layers) for each
        requested position. E_reach is NaN for the last position."""
        T1 = toks.shape[1]
        jobs = [(p, L) for p in positions for L in range(N_LAYERS)]
        kl = self.drop_kl(toks, jobs, mu_t, clean_logp).reshape(len(positions), N_LAYERS, T1)
        e_next = np.array([kl[i, :, p].sum() for i, p in enumerate(positions)])
        e_reach = np.array([kl[i, :, p + 1:].mean(axis=1).sum() if p + 1 < T1 else np.nan
                            for i, p in enumerate(positions)])
        return e_next, e_reach


def mechanics_checks(run, frame_texts, mu_t):
    """Checks on frames holding common words only. Nothing about UP is seen."""
    print("MECHANICS CHECKS (common-word frames only)")
    ok = True
    for ids, p in frame_texts:
        toks, resid, logp = run.clean(ids)
        T1 = toks.shape[1]
        jobs = [(p, L) for L in range(N_LAYERS)]
        own = run.drop_kl(toks, jobs, [resid[L] for L in range(N_LAYERS)], logp)
        c1 = float(np.abs(own).max())
        kl = run.drop_kl(toks, jobs, mu_t, logp)
        c2 = float(np.abs(kl[N_LAYERS - 1, p + 1:]).max())
        c3 = float(np.abs(kl[:, :p]).max())
        real = float(kl[:, p].sum())
        good = c1 < IDENT_TOL and c2 < IDENT_TOL and c3 < IDENT_TOL and real > IDENT_TOL
        ok &= good
        print(f"  own-state drop max KL {c1:.2e} | last-layer drop, later positions {c2:.2e} | "
              f"earlier positions {c3:.2e} | real drop E_next {real:.3f} -> {'ok' if good else 'FAIL'}")
    print("MECHANICS", "PASS" if ok else "FAIL")
    return ok


def build_axes(R, words, lists):
    """R: [n, 24, d] word states (mean over frames). Returns {L: {axis: vec}}."""
    idx = {w: i for i, w in enumerate(words)}
    axes = {}
    for L in LAYERS:
        arr = R[:, L, :].astype(np.float64)
        aniso = arr.mean(axis=0); aniso /= np.linalg.norm(aniso)
        ma = lambda ws: arr[[idx[w] for w in ws]].mean(axis=0)
        freq = ma(lists["COMMON"]) - ma(lists["RARE"]); freq /= np.linalg.norm(freq)
        fo = freq - (freq @ aniso) * aniso; fo /= np.linalg.norm(fo)

        def strip(d):
            d = d - (d @ aniso) * aniso
            d = d - (d @ fo) * fo
            return d / np.linalg.norm(d)

        def axis(pos, neg):
            raw = ma(pos) - ma(neg)
            return strip(raw / np.linalg.norm(raw))
        axes[L] = {name: axis(*pn) for name, pn in lists["axes"].items()}
    return axes


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


if __name__ == "__main__":
    print("exp175 — is UP grounded in holding against gravity / effecting?  (prereg frozen at 1cbdb60)")
    print(f"stimuli sha256 {STIM_SHA} matches prereg; rule constants match prereg")
    if not self_test():
        print("Self-test failed: NOT running the model.")
        sys.exit(1)
    if "--selftest-only" in sys.argv:
        sys.exit(0)

    from wordfreq import zipf_frequency
    run = Runner()
    torch = run.torch
    W = STIM["words"]
    contrast_lists = {"UP": (W["UP"], W["DOWN"])}
    for n_ in SCHEMA_NAMES:
        contrast_lists[n_] = (W["schemas"][n_]["pos"], W["schemas"][n_]["neg"])
    contrast_lists["VAL"] = (W["VALPOS"], W["VALNEG"])
    analysis_words = sorted({w for pos, neg in contrast_lists.values() for w in pos + neg})
    all_words = sorted(set(analysis_words) | set(W["COMMON"]) | set(W["RARE"]))
    frames = STIM["frames"]

    def frame_text(w, fr):
        words = fr.split(" ")
        slot = words.index("{w}")
        words[slot] = w
        g = run.groups(words)
        ids = [t for x in g for t in x]
        p = sum(len(x) for x in g[: slot + 1])        # last token of the word; BOS is position 0
        return ids, p

    # ---- Part A, pass 1: clean states at the word position + layer averages
    print("\n" + "=" * 78)
    print("PART A — words in frames")
    print("=" * 78)
    cacheA = f"{ROOT}/exp175_cache_wordstates.npz"
    d_model = run.model.cfg.d_model
    if os.path.exists(cacheA) and str(np.load(cacheA, allow_pickle=True)["sha"]) == STIM_SHA:
        z = np.load(cacheA, allow_pickle=True)
        RA, muA = z["R"], z["mu"]
        print("  word states and layer averages loaded from cache")
    else:
        RA = np.zeros((len(all_words), len(frames), N_LAYERS, d_model), np.float32)
        tot = np.zeros((N_LAYERS, d_model)); cnt = 0
        for wi, w in enumerate(all_words):
            for fi, fr in enumerate(frames):
                ids, p = frame_text(w, fr)
                toks, resid, _ = run.clean(ids)
                RA[wi, fi] = resid[:, p, :].float().cpu().numpy()
                tot += resid[:, 1:, :].float().sum(dim=1).cpu().numpy().astype(np.float64); cnt += toks.shape[1] - 1
            if (wi + 1) % 100 == 0:
                print(f"  clean pass {wi + 1}/{len(all_words)} words")
        muA = tot / cnt
        np.savez(cacheA, R=RA, mu=muA, sha=STIM_SHA, words=np.array(all_words, dtype=object))
        print(f"  clean pass done: {len(all_words)} words x {len(frames)} frames; layer averages over {cnt} tokens (saved)")
    muA_t = torch.tensor(muA, dtype=torch.float32, device=run.dev)

    print()
    checks = [frame_text(w, frames[i % len(frames)]) for i, w in enumerate(["the", "and", "with", "that"])]
    if not mechanics_checks(run, checks, [muA_t[L] for L in range(N_LAYERS)]):
        print("STOP: the drop mechanics are wrong. Nothing below was computed.")
        sys.exit(2)

    # ---- Part A, pass 2: the holes
    rowsA_path = f"{ROOT}/exp175_rows_words.jsonl"
    rowsA = {}
    if os.path.exists(rowsA_path):
        for line in open(rowsA_path):
            r = json.loads(line); rowsA[(r["word"], r["frame"])] = r
        print(f"\n  resuming: {len(rowsA)} word-frame rows on disk")
    aidx = {w: i for i, w in enumerate(all_words)}
    with open(rowsA_path, "a") as fh:
        for wi, w in enumerate(analysis_words):
            for fi, fr in enumerate(frames):
                if (w, fi) in rowsA:
                    continue
                ids, p = frame_text(w, fr)
                toks, resid, logp = run.clean(ids)
                e_next, e_reach = run.holes(toks, logp, [muA_t[L] for L in range(N_LAYERS)], [p])
                dist = float(np.mean([np.log(np.linalg.norm(RA[aidx[w], fi, L].astype(np.float64) - muA[L]))
                                      for L in range(N_LAYERS)]))
                r = {"word": w, "frame": fi, "pos": p, "E_next": float(e_next[0]),
                     "E_reach": float(e_reach[0]), "dist": dist}
                fh.write(json.dumps(r) + "\n"); fh.flush(); rowsA[(w, fi)] = r
            if (wi + 1) % 60 == 0:
                print(f"  holes {wi + 1}/{len(analysis_words)} words (saved as they finish)")
    value = {w: float(np.mean([np.log(rowsA[(w, fi)]["E_reach"]) for fi in range(len(frames))])) for w in analysis_words}
    vnext = {w: float(np.mean([np.log(rowsA[(w, fi)]["E_next"]) for fi in range(len(frames))])) for w in analysis_words}
    dist = {w: float(np.mean([rowsA[(w, fi)]["dist"] for fi in range(len(frames))])) for w in analysis_words}
    zipf = {w: zipf_frequency(w, "en") for w in analysis_words}
    outA = analyse_A(value, vnext, dist, zipf, contrast_lists)
    print()
    report_A(outA)
    vA = verdict_A(outA)
    print(f"\nPART A VERDICT (frozen rule): {vA}")
    results = {"partA": _jsonable(outA), "partA_verdict": vA, "stimuli_sha": STIM_SHA,
               "word_values": {w: {"ln_E_reach": value[w], "ln_E_next": vnext[w], "dist": dist[w],
                                   "zipf": zipf[w]} for w in analysis_words}}
    json.dump(results, open(f"{ROOT}/exp175_results.json", "w"), indent=1)
    print("  (Part A results saved to exp175_results.json)")

    # ---- axes for Part B
    Rmean = RA.mean(axis=1)                                   # [n, 24, d]
    axes = build_axes(Rmean, all_words, {"COMMON": W["COMMON"], "RARE": W["RARE"],
                                         "axes": {"UP_spatial": (W["UP"], W["DOWN"]),
                                                  "VAL": (W["VALPOS"], W["VALNEG"])}})
    old = np.load(f"{ROOT}/exp174_cache_words.npz", allow_pickle=True)
    old_axes = build_axes(old["R"].astype(np.float64), list(old["words"]),
                          {"COMMON": W["COMMON"], "RARE": W["RARE"],
                           "axes": {"UP_literal_bare": (LAKOFF_UP, LAKOFF_DOWN)}})
    for L in LAYERS:
        axes[L]["UP_literal_bare"] = old_axes[L]["UP_literal_bare"]
    print("  axis check: cos(UP_spatial, UP_literal_bare) by layer: "
          + "  ".join(f"L{L} {float(axes[L]['UP_spatial'] @ axes[L]['UP_literal_bare']):+.3f}" for L in LAYERS)
          + " | cos(UP_spatial, VAL): "
          + "  ".join(f"L{L} {float(axes[L]['UP_spatial'] @ axes[L]['VAL']):+.3f}" for L in LAYERS))

    # ---- Part B
    print("\n" + "=" * 78)
    print("PART B — sentences: does the UP readout follow the hole, or follow happy?")
    print("=" * 78)
    texts = []
    for it, item in enumerate(STIM["items"]):
        g = run.groups(item["sentence"].split(" "))
        intact = [t for x in g for t in x]
        assert len(intact) == item["n_tokens"], "tokenisation drifted from the frozen stimuli"
        texts.append((f"{it}|intact", it, item["triple"], item["valence"], "intact", intact))
        for k, perm in enumerate(item["shuffles"]):
            ids = [t for j in perm for t in g[j]]
            assert sorted(ids) == sorted(intact) and ids != intact
            texts.append((f"{it}|shuf{k}", it, item["triple"], item["valence"], f"shuf{k}", ids))
    print(f"  {len(texts)} texts ({len(STIM['items'])} sentences x (1 intact + {N_SHUFFLES} scrambles)); token-identity assertions passed")
    mu_path = f"{ROOT}/exp175_cache_muB.npz"
    if os.path.exists(mu_path) and str(np.load(mu_path)["sha"]) == STIM_SHA:
        muB = np.load(mu_path)["mu"]; print("  layer averages loaded from cache")
    else:
        tot = np.zeros((N_LAYERS, d_model)); cnt = 0
        for (_, _, _, _, _, ids) in texts:
            toks, resid, _ = run.clean(ids)
            tot += resid[:, 1:, :].float().sum(dim=1).cpu().numpy().astype(np.float64); cnt += toks.shape[1] - 1
        muB = tot / cnt
        np.savez(mu_path, mu=muB, sha=STIM_SHA, n_tokens=cnt)
        print(f"  layer averages over {cnt} sentence tokens (saved)")
    muB_t = torch.tensor(muB, dtype=torch.float32, device=run.dev)
    muB_list = [muB_t[L] for L in range(N_LAYERS)]

    rowsB_path = f"{ROOT}/exp175_rows_texts.jsonl"
    rowsB, have = [], set()
    if os.path.exists(rowsB_path):
        for line in open(rowsB_path):
            r = json.loads(line); rowsB.append(r); have.add(r["key"])
        print(f"  resuming: {len(rowsB)} text rows on disk")
    with open(rowsB_path, "a") as fh:
        for n_, (key, it, triple, valence, order, ids) in enumerate(texts):
            if key in have:
                continue
            toks, resid, logp = run.clean(ids)
            T = len(ids)
            positions = list(range(1, T + 1))
            e_next, e_reach = run.holes(toks, logp, muB_list, positions)
            tk_np = toks[0].cpu().numpy()
            lp = logp.cpu().numpy()
            surp = np.array([-lp[p - 1, tk_np[p]] for p in positions])
            ln_reach = np.log(e_reach[:-1]); ln_next = np.log(e_next)
            row = {"key": key, "item": it, "triple": triple, "valence": valence, "order": order,
                   "n_tokens": T, "E_reach": float(ln_reach.mean()), "E_next": float(ln_next.mean()),
                   "surp": float(surp.mean()),
                   "tok_ln_reach": [float(x) for x in ln_reach], "tok_ln_next": [float(x) for x in ln_next],
                   "tok_surp": [float(x) for x in surp], "tok_ids": [int(x) for x in ids]}
            res_np = resid[:, 1:, :].float().cpu().numpy().astype(np.float64)   # [24, T, d]
            for name in ("UP_spatial", "UP_literal_bare", "VAL"):
                row[name] = {}; row["tok_" + name] = {}
                for L in LAYERS:
                    X = res_np[L]
                    U = X / np.linalg.norm(X, axis=1, keepdims=True)
                    proj = U @ axes[L][name]
                    row[name][str(L)] = float(proj.mean())
                    row["tok_" + name][str(L)] = [float(x) for x in proj]
            fh.write(json.dumps(row) + "\n"); fh.flush()
            rowsB.append(row)
            if (n_ + 1) % 40 == 0:
                print(f"  texts {n_ + 1}/{len(texts)} (rows saved as they finish)", flush=True)

    outB = analyse_B(rowsB, LAYERS, axis_key="UP_spatial")
    report_B(outB)
    vB = verdict_B(outB)
    print(f"\nPART B VERDICT (frozen rule, UP_spatial, reach): {vB}")
    # D1: within intact sentences, token-level correlation after removing position
    d1 = {}
    for L in LAYERS:
        rs = []
        for r in rowsB:
            if r["order"] != "intact":
                continue
            e = np.array(r["tok_ln_reach"]); u = np.array(r["tok_UP_spatial"][str(L)][:-1])
            pos = np.arange(len(e), dtype=float)
            rs.append(corrf(residualise(e, [pos]), residualise(u, [pos])))
        rs = np.array(rs)
        d1[L] = (float(rs.mean()), float(rs.mean() / (rs.std(ddof=1) / np.sqrt(len(rs)))))
    print("  D1 (descriptive) within intact sentences, token-level corr(ln E_reach, UP readout), position removed: "
          + "  ".join(f"L{L} mean r {d1[L][0]:+.3f} (t {d1[L][1]:+.1f})" for L in LAYERS))
    outB2 = analyse_B(rowsB, LAYERS, axis_key="UP_literal_bare")
    print("\nReported beside it, never substituting:")
    report_B(outB2)
    print(f"  (UP_literal_bare would read: {verdict_B(outB2)})")
    results.update({"partB": _jsonable(outB), "partB_verdict": vB, "partB_D1": _jsonable(d1),
                    "partB_bare": _jsonable(outB2), "partB_bare_verdict": verdict_B(outB2)})
    json.dump(results, open(f"{ROOT}/exp175_results.json", "w"), indent=1)
    print("\nAll results saved to exp175_results.json; rows in exp175_rows_words.jsonl and exp175_rows_texts.jsonl")
