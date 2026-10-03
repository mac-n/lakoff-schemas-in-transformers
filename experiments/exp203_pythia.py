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
    ("affect", "mc", grab("AFFECT_PROMPTS"), ([w for w in grab("POS_CANDIDATES") if w.strip() != "uplifted"], [w for w in grab("NEG_CANDIDATES") if w.strip() != "low"])),
]
assert grab("STRENGTHS") == STRENGTHS
GATE = {-12: (165.900, -0.645), 0: (168.398, -0.181), 16: (171.567, 0.356)}       # exp116, u111_clean: height, affect

import exp180_paired_retry as e180            # exp180's frozen pairs and frames
import exp185_bundle_vs_default as e185       # exp185's frozen sortings (Claude's)


def main():
    from transformer_lens import HookedTransformer
    device = "mps"
    print("exp203 (Pythia 1.4B L12) — affect without uplifted/low; directions and scoring from exp187")
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

    DVS_SPEC_ = [x for x in DVS_SPEC if x[0] == 'affect']
    DVS = [(n, k, pr, (single_token_ids(pl[0]), single_token_ids(pl[1])) if k == "mc" else pl) for n, k, pr, pl in DVS_SPEC_]

    res_path = f"{ROOT}/exp203_cells_pythia.json"
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

    print("  affect candidates:", DVS[0][3])
    eff = lambda d: cell(d, True, 8)["affect"] - cell(d, True, -8)["affect"]
    null = sorted(eff(f"NULL_{k:02d}") for k in range(N_NULL)); p95 = float(np.percentile(null, 95))
    up, val = eff("CLEAN_UP"), eff("CLEAN_VALENCE")
    print(f"  CLEAN_UP {up:+.3f} | CLEAN_VALENCE {val:+.3f} | null median {np.median(null):+.3f}, 95th {p95:+.3f}, max {max(null):+.3f}")
    print(f"  >>> Pythia L12: {'SURVIVES' if up > p95 else 'FALLS'} <<<")
    json.dump(dict(up=up, valence=val, null=null, p95=p95, verdict='SURVIVES' if up > p95 else 'FALLS'), open(f"{ROOT}/exp203_results_pythia.json", "w"), indent=1)


if __name__ == "__main__":
    main()
