"""
exp187_clean_steering.py — does a cleanly built UP direction still steer the
bundle? Frozen rules: PREREG_exp187.md (commit 8bff0a1, 2026-10-02 20:22
IST). Written AFTER the freeze.

exp116 redone: same model, layer, strengths, outcome measures and scoring.
exp116 cannot be imported (it loads its model and runs at import), so its
word lists, prompts and value grids are read out of its source file, and its
scoring functions are COPIED below (flagged in the prereg). The replication
gate checks the copy against exp116's saved numbers.

Every finished cell is written to disk; the run resumes.
"""
import json
import os
import re
import sys
from contextlib import nullcontext

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)

N_NULL = 30
SEED_NULL = 187
GATE_EV = 0.05
GATE_MC = 0.01
LAYER = 12
HOOK = f"blocks.{LAYER}.hook_resid_post"
STRENGTHS = [-12, -8, -4, -2, 0, 2, 4, 8, 12, 16]
PREREG = open(f"{ROOT}/PREREG_exp187.md").read()
for _n, _v in [("N_NULL", N_NULL), ("SEED_NULL", SEED_NULL), ("GATE_EV", GATE_EV), ("GATE_MC", GATE_MC)]:
    _m = re.search(rf"\b{_n} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_v), f"{_n} drifted from prereg"

SRC = open(f"{ROOT}/exp116_cluster_discrimination.py").read()


def grab(name, pat=r"(\[.*?\])"):
    m = re.search(rf"^{name} = {pat}", SRC, re.S | re.M)
    assert m, name
    return eval(m.group(1))


EXP111_UP, EXP111_DOWN = grab("EXP111_UP"), grab("EXP111_DOWN")
LAKOFF_UP, LAKOFF_DOWN = grab("LAKOFF_UP"), grab("LAKOFF_DOWN")
COMMON, RARE = grab("COMMON"), grab("RARE")
RNG = r"(list\(range\(.*?\)\))"
DVS_SPEC = [
    ("patient_height_cm", "ev", grab("PATIENT_HEIGHT_PROMPTS"), grab("HEIGHT_VALUES", RNG)),
    ("pool_depth_cm", "ev", grab("POOL_DEPTH_PROMPTS"), grab("DEPTH_VALUES", RNG)),
    ("door_width_cm", "ev", grab("DOOR_WIDTH_PROMPTS"), grab("WIDTH_VALUES", RNG)),
    ("widget_quantity", "ev", grab("WIDGET_QUANTITY_PROMPTS"), grab("WIDGET_VALUES", RNG)),
    ("status", "mc", grab("STATUS_PROMPTS"), (grab("HIGH_STATUS_CANDIDATES"), grab("LOW_STATUS_CANDIDATES"))),
    ("affect", "mc", grab("AFFECT_PROMPTS"), (grab("POS_CANDIDATES"), grab("NEG_CANDIDATES"))),
]
assert grab("STRENGTHS") == STRENGTHS
GATE = {-12: (165.900, -0.645), 0: (168.398, -0.181), 16: (171.567, 0.356)}       # exp116, u111_clean: height, affect

import exp180_paired_retry as e180            # exp180's frozen pairs and frames
import exp185_bundle_vs_default as e185       # exp185's frozen sortings (Claude's)


def main():
    from transformer_lens import HookedTransformer
    device = "mps"
    print("exp187 — clean steering (prereg frozen at 8bff0a1)")
    model = HookedTransformer.from_pretrained("pythia-1.4b", device=device)
    model.eval()
    tk = model.tokenizer
    BOS = int(tk.bos_token_id)
    assert model.to_tokens("the").shape[1] == 1, "to_tokens now prepends a start token: exp116's format would change"

    # ---- exp116's directions, exactly as exp116 built them (bare words, no start marker)
    def res(w):
        toks = model.to_tokens(w)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=HOOK)
        return cache[HOOK][0, -1, :].clone()

    def dir_from(up, down):
        u = torch.stack([res(w) for w in up]).mean(0)
        d = torch.stack([res(w) for w in down]).mean(0)
        raw = u - d
        return raw / raw.norm()

    freq_axis = dir_from(COMMON, RARE)
    strip_o = lambda v: (v - (v @ freq_axis) * freq_axis) / (v - (v @ freq_axis) * freq_axis).norm()
    dirs = {"ORIG_u111": strip_o(dir_from(EXP111_UP, EXP111_DOWN)), "ORIG_ulak": strip_o(dir_from(LAKOFF_UP, LAKOFF_DOWN))}

    # ---- clean directions: single-token words inside exp180's sentences, with a start marker
    frames = e180.STIM["frames"]; P = e180.STIM["pairs"]
    bundle = e185.BUNDLE_OLD + e185.BUNDLE_NEW; nolink = e185.NOLINK_OLD + e185.NOLINK_NEW
    pool = sorted({w for ps in list(P.values()) + [bundle, nolink] for p in ps for w in p})

    def state(w):
        acc = 0
        for fr in frames:
            ws = fr.split(" "); slot = ws.index("{w}"); ws[slot] = w
            g = [tk.encode(" " + x, add_special_tokens=False) for x in ws]
            ids = [BOS] + [t for x in g for t in x]
            p = sum(len(x) for x in g[: slot + 1])
            with torch.no_grad():
                _, cache = model.run_with_cache(torch.tensor([ids], device=device), names_filter=HOOK)
            acc = acc + cache[HOOK][0, p, :].clone()
        return acc / len(frames)

    S = {w: state(w) for w in pool + COMMON + RARE}
    print(f"  clean word states: {len(S)} words x {len(frames)} sentences")
    mean_of = lambda ws: torch.stack([S[w] for w in ws]).mean(0)
    fq = mean_of(COMMON) - mean_of(RARE); fq = fq / fq.norm()

    def clean_dir(a, b):
        raw = mean_of(a) - mean_of(b); raw = raw / raw.norm()
        v = raw - (raw @ fq) * fq
        return v / v.norm()

    first = lambda ps: [p[0] for p in ps]; second = lambda ps: [p[1] for p in ps]
    dirs["CLEAN_UP"] = clean_dir(first(P["vertical"]), second(P["vertical"]))
    dirs["CLEAN_VALENCE"] = clean_dir(first(P["valence"]), second(P["valence"]))
    dirs["CLEAN_BUNDLE"] = clean_dir(first(bundle), second(bundle))
    dirs["CLEAN_NOLINK"] = clean_dir(first(nolink), second(nolink))
    rng = np.random.default_rng(SEED_NULL)
    for k in range(N_NULL):
        ws = list(rng.choice(pool, 62, replace=False))
        dirs[f"NULL_{k:02d}"] = clean_dir(ws[:31], ws[31:])
    cos = lambda a, b: float(dirs[a] @ dirs[b])
    print(f"  cos(CLEAN_UP, ORIG_ulak) {cos('CLEAN_UP', 'ORIG_ulak'):+.3f}  cos(CLEAN_UP, ORIG_u111) {cos('CLEAN_UP', 'ORIG_u111'):+.3f}"
          f"  cos(CLEAN_UP, CLEAN_VALENCE) {cos('CLEAN_UP', 'CLEAN_VALENCE'):+.3f}  cos(CLEAN_UP, CLEAN_BUNDLE) {cos('CLEAN_UP', 'CLEAN_BUNDLE'):+.3f}")

    # ---- exp116's scoring, COPIED, with an optional start marker in front of the prompt
    def single_token_ids(cands):
        return [tk.encode(w, add_special_tokens=False)[0] for w in cands if len(tk.encode(w, add_special_tokens=False)) == 1]

    def steered(d, s):
        if s == 0.0:
            return nullcontext()
        return model.hooks(fwd_hooks=[(HOOK, lambda resid, hook: resid + s * d)])

    def expected_value(d, s, prompts, values, bos):
        suf = {v: tk.encode(f" {v}", add_special_tokens=False) for v in values}
        means = []
        with steered(d, s):
            for prompt in prompts:
                pids = ([BOS] if bos else []) + tk.encode(prompt, add_special_tokens=False)
                lp = {}
                for v in values:
                    sfx = suf[v]
                    tokens = torch.tensor([pids + sfx], device=device)
                    with torch.no_grad():
                        logits = model(tokens)
                    logp = torch.log_softmax(logits[0], dim=-1)
                    start = len(pids) - 1
                    lp[v] = sum(logp[start + i, tid].item() for i, tid in enumerate(sfx))
                lps = np.array([lp[v] for v in values]); lps -= lps.max()
                p = np.exp(lps); p /= p.sum()
                means.append(float(np.sum(np.array(values) * p)))
        return float(np.mean(means))

    def mass_contrast(d, s, prompts, pos_ids, neg_ids, bos):
        diffs = []
        with steered(d, s):
            for prompt in prompts:
                ids = ([BOS] if bos else []) + tk.encode(prompt, add_special_tokens=False)
                with torch.no_grad():
                    logits = model(torch.tensor([ids], device=device))
                lp = torch.log_softmax(logits[0, -1, :], dim=-1)
                diffs.append(torch.logsumexp(lp[pos_ids], 0).item() - torch.logsumexp(lp[neg_ids], 0).item())
        return float(np.mean(diffs))

    DVS = [(n, k, pr, (single_token_ids(pl[0]), single_token_ids(pl[1])) if k == "mc" else pl) for n, k, pr, pl in DVS_SPEC]

    res_path = f"{ROOT}/exp187_cells.json"
    cells = json.load(open(res_path)) if os.path.exists(res_path) else {}

    def cell(dname, bos, s):
        key = f"{dname}|{'bos' if bos else 'nobos'}|{s}"
        if key not in cells:
            d = dirs[dname]
            row = {}
            for n, k, pr, pl in DVS:
                row[n] = expected_value(d, float(s), pr, pl, bos) if k == "ev" else mass_contrast(d, float(s), pr, pl[0], pl[1], bos)
            cells[key] = row
            json.dump(cells, open(res_path, "w"))
        return cells[key]

    # ---- gate
    print("\nREPLICATION GATE — exp116's u111_clean in exp116's own format")
    ok = True
    for s, (h, a) in GATE.items():
        r = cell("ORIG_u111", False, s)
        good = abs(r["patient_height_cm"] - h) <= GATE_EV and abs(r["affect"] - a) <= GATE_MC
        ok &= good
        print(f"  s {s:+3d}: patient height {r['patient_height_cm']:.3f} (exp116 {h:.3f})  affect {r['affect']:+.3f} (exp116 {a:+.3f})  {'ok' if good else 'OUT'}")
    print("REPLICATION GATE", "PASS" if ok else "FAIL")
    if not ok:
        print("STOP: the copied scoring does not reproduce exp116."); sys.exit(2)

    names = [n for n, *_ in DVS]

    def table(dname, bos):
        print(f"\n--- {dname} ({'start marker' if bos else 'exp116 format, no start marker'}) ---")
        print(f"  {'s':>4}  " + "  ".join(f"{n[:12]:>12}" for n in names))
        for s in STRENGTHS:
            r = cell(dname, bos, s)
            print(f"  {s:>+4}  " + "  ".join(f"{r[n]:>12.3f}" for n in names), flush=True)

    for dname, bos in [("CLEAN_UP", True), ("CLEAN_UP", False), ("ORIG_u111", False), ("ORIG_ulak", False),
                       ("ORIG_u111", True), ("ORIG_ulak", True), ("CLEAN_VALENCE", True), ("CLEAN_BUNDLE", True),
                       ("CLEAN_NOLINK", True)]:
        table(dname, bos)

    effect = lambda dname, bos: {n: cell(dname, bos, 8)[n] - cell(dname, bos, -8)[n] for n in names}
    null = []
    for k in range(N_NULL):
        null.append(effect(f"NULL_{k:02d}", True))
        if (k + 1) % 10 == 0:
            print(f"  random-word null {k + 1}/{N_NULL}", flush=True)
    print("\n" + "=" * 76 + "\nEFFECT = value at +8 minus value at −8 (start-marker prompts)\n" + "=" * 76)
    p95 = {n: float(np.percentile([x[n] for x in null], 95)) for n in names}
    p5 = {n: float(np.percentile([x[n] for x in null], 5)) for n in names}
    med = {n: float(np.median([x[n] for x in null])) for n in names}
    print(f"  {'direction':>16}  " + "  ".join(f"{n[:12]:>12}" for n in names))
    print(f"  {'null 5%':>16}  " + "  ".join(f"{p5[n]:>+12.3f}" for n in names))
    print(f"  {'null median':>16}  " + "  ".join(f"{med[n]:>+12.3f}" for n in names))
    print(f"  {'null 95%':>16}  " + "  ".join(f"{p95[n]:>+12.3f}" for n in names))
    eff = {}
    for dname in ("CLEAN_UP", "ORIG_u111", "ORIG_ulak", "CLEAN_VALENCE", "CLEAN_BUNDLE", "CLEAN_NOLINK"):
        eff[dname] = effect(dname, True)
        print(f"  {dname:>16}  " + "  ".join(f"{eff[dname][n]:>+12.3f}{'*' if eff[dname][n] > p95[n] and eff[dname][n] > 0 else ' '}" for n in names))
    print("  (* = positive and above the null's 95th percentile)")

    def label(e):
        moves = lambda n: e[n] > 0 and e[n] > p95[n]
        if not moves("affect"):
            return "NOT_BEYOND_RANDOM"
        return "BUNDLE_STEERS" if (moves("status") or moves("widget_quantity")) else "AFFECT_ONLY"

    v = label(eff["CLEAN_UP"])
    print(f"\n  >>> CLEAN UP: {v} <<<")
    print("  same rule, other directions: " + ", ".join(f"{d} {label(eff[d])}" for d in ("ORIG_u111", "ORIG_ulak", "CLEAN_VALENCE", "CLEAN_BUNDLE", "CLEAN_NOLINK")))
    e = eff["CLEAN_UP"]
    print(f"  literal verticality under CLEAN UP: patient height {e['patient_height_cm']:+.3f} cm, pool depth {e['pool_depth_cm']:+.3f} cm")
    json.dump({"verdict": v, "effects": eff, "null_p5": p5, "null_median": med, "null_p95": p95,
               "labels": {d: label(eff[d]) for d in eff}}, open(f"{ROOT}/exp187_results.json", "w"), indent=1)
    print("\nresults saved to exp187_results.json; every cell in exp187_cells.json")


if __name__ == "__main__":
    main()
