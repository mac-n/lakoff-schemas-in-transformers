"""
exp194_gpt2_steering.py — clean UP steering in GPT-2-medium at three layers.
Frozen rules: PREREG_exp194.md (commit c70dac8, 3 Oct 2026 00:53 IST).
Written AFTER the freeze. Scoring is exp116's as copied in exp187 (that copy
passed exp187's replication gate). Cells saved as they finish; resumes.
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
import exp187_clean_steering as e187        # DVS_SPEC, COMMON, RARE, STRENGTHS (read from exp116's source)
import exp180_paired_retry as e180
import exp185_bundle_vs_default as e185

N_NULL = 20
SEED_NULL = 194
LAYERS = [12, 6, 18]
PRIMARY = 12
PREREG = open(f"{ROOT}/PREREG_exp194.md").read()
for _n, _v in [("N_NULL", N_NULL), ("SEED_NULL", SEED_NULL)]:
    _m = re.search(rf"\b{_n} = ([0-9]+)", PREREG)
    assert _m and int(_m.group(1)) == _v, f"{_n} drifted from prereg"
STRENGTHS = e187.STRENGTHS
COMMON, RARE, DVS_SPEC = e187.COMMON, e187.RARE, e187.DVS_SPEC


def main():
    from transformer_lens import HookedTransformer
    device = "mps"
    print("exp194 — clean UP steering in GPT-2-medium (prereg frozen at c70dac8)")
    model = HookedTransformer.from_pretrained("gpt2-medium", device=device); model.eval()
    tk = model.tokenizer; BOS = int(tk.bos_token_id)
    frames = e180.STIM["frames"]; P = e180.STIM["pairs"]
    bundle = e185.BUNDLE_OLD + e185.BUNDLE_NEW; nolink = e185.NOLINK_OLD + e185.NOLINK_NEW
    pool = sorted({w for ps in list(P.values()) + [bundle, nolink] for p in ps for w in p})
    first = lambda ps: [p[0] for p in ps]; second = lambda ps: [p[1] for p in ps]
    hooks = {L: f"blocks.{L}.hook_resid_post" for L in LAYERS}

    def states(w):
        acc = {L: 0 for L in LAYERS}
        for fr in frames:
            ws = fr.split(" "); slot = ws.index("{w}"); ws[slot] = w
            g = [tk.encode(" " + x, add_special_tokens=False) for x in ws]
            ids = [BOS] + [t for x in g for t in x]; p = sum(len(x) for x in g[: slot + 1])
            with torch.no_grad():
                _, cache = model.run_with_cache(torch.tensor([ids], device=device), names_filter=list(hooks.values()))
            for L in LAYERS:
                acc[L] = acc[L] + cache[hooks[L]][0, p, :].clone()
        return {L: acc[L] / len(frames) for L in LAYERS}

    S = {w: states(w) for w in pool + COMMON + RARE}
    print(f"  clean word states: {len(S)} words x {len(frames)} sentences", flush=True)
    rng = np.random.default_rng(SEED_NULL)
    null_draws = [list(rng.choice(pool, 62, replace=False)) for _ in range(N_NULL)]
    dirs = {}
    for L in LAYERS:
        mean_of = lambda ws, L=L: torch.stack([S[w][L] for w in ws]).mean(0)
        fq = mean_of(COMMON) - mean_of(RARE); fq = fq / fq.norm()

        def clean(a, b, fq=fq, mean_of=mean_of):
            raw = mean_of(a) - mean_of(b); raw = raw / raw.norm()
            v = raw - (raw @ fq) * fq
            return v / v.norm()
        dirs[L] = {"CLEAN_UP": clean(first(P["vertical"]), second(P["vertical"])),
                   "CLEAN_VALENCE": clean(first(P["valence"]), second(P["valence"])),
                   "CLEAN_BUNDLE": clean(first(bundle), second(bundle)), "CLEAN_NOLINK": clean(first(nolink), second(nolink))}
        for k, ws in enumerate(null_draws):
            dirs[L][f"NULL_{k:02d}"] = clean(ws[:31], ws[31:])

    def single_token_ids(cands):
        return [tk.encode(w, add_special_tokens=False)[0] for w in cands if len(tk.encode(w, add_special_tokens=False)) == 1]

    def steered(L, d, s):
        if s == 0.0:
            return nullcontext()
        return model.hooks(fwd_hooks=[(hooks[L], lambda resid, hook: resid + s * d)])

    def expected_value(L, d, s, prompts, values):
        suf = {v: tk.encode(f" {v}", add_special_tokens=False) for v in values}
        means = []
        with steered(L, d, s):
            for prompt in prompts:
                pids = [BOS] + tk.encode(prompt, add_special_tokens=False)
                lp = {}
                for v in values:
                    sfx = suf[v]
                    with torch.no_grad():
                        logits = model(torch.tensor([pids + sfx], device=device))
                    logp = torch.log_softmax(logits[0], dim=-1)
                    start = len(pids) - 1
                    lp[v] = sum(logp[start + i, tid].item() for i, tid in enumerate(sfx))
                lps = np.array([lp[v] for v in values]); lps -= lps.max()
                p = np.exp(lps); p /= p.sum()
                means.append(float(np.sum(np.array(values) * p)))
        return float(np.mean(means))

    def mass_contrast(L, d, s, prompts, pos_ids, neg_ids):
        diffs = []
        with steered(L, d, s):
            for prompt in prompts:
                ids = [BOS] + tk.encode(prompt, add_special_tokens=False)
                with torch.no_grad():
                    logits = model(torch.tensor([ids], device=device))
                lp = torch.log_softmax(logits[0, -1, :], dim=-1)
                diffs.append(torch.logsumexp(lp[pos_ids], 0).item() - torch.logsumexp(lp[neg_ids], 0).item())
        return float(np.mean(diffs))

    DVS = [(n, k, pr, (single_token_ids(pl[0]), single_token_ids(pl[1])) if k == "mc" else pl) for n, k, pr, pl in DVS_SPEC]
    names = [n for n, *_ in DVS]
    res_path = f"{ROOT}/exp194_cells.json"
    cells = json.load(open(res_path)) if os.path.exists(res_path) else {}

    def cell(L, dname, s):
        key = f"L{L}|{dname}|{s}"
        if key not in cells:
            d = dirs[L][dname]; row = {}
            for n, k, pr, pl in DVS:
                row[n] = expected_value(L, d, float(s), pr, pl) if k == "ev" else mass_contrast(L, d, float(s), pr, pl[0], pl[1])
            cells[key] = row; json.dump(cells, open(res_path, "w"))
        return cells[key]

    results = {}
    for L in LAYERS:
        print(f"\n--- layer {L}{' (PRIMARY)' if L == PRIMARY else ''} ---", flush=True)
        if L == PRIMARY:
            print(f"  CLEAN_UP dose curve:  {'s':>4}  " + "  ".join(f"{n[:12]:>12}" for n in names))
            for s in STRENGTHS:
                r = cell(L, "CLEAN_UP", s)
                print(f"                        {s:>+4}  " + "  ".join(f"{r[n]:>12.3f}" for n in names), flush=True)
        eff = lambda dname: {n: cell(L, dname, 8)[n] - cell(L, dname, -8)[n] for n in names}
        null = [eff(f"NULL_{k:02d}") for k in range(N_NULL)]
        p95 = {n: float(np.percentile([x[n] for x in null], 95)) for n in names}
        p5 = {n: float(np.percentile([x[n] for x in null], 5)) for n in names}
        E = {d: eff(d) for d in ("CLEAN_UP", "CLEAN_VALENCE", "CLEAN_BUNDLE", "CLEAN_NOLINK")}
        print(f"  {'effect (+8 minus -8)':>22}  " + "  ".join(f"{n[:12]:>12}" for n in names))
        print(f"  {'null 5%':>22}  " + "  ".join(f"{p5[n]:>+12.3f}" for n in names))
        print(f"  {'null 95%':>22}  " + "  ".join(f"{p95[n]:>+12.3f}" for n in names))
        for d in E:
            print(f"  {d:>22}  " + "  ".join(f"{E[d][n]:>+12.3f}{'*' if E[d][n] > 0 and E[d][n] > p95[n] else ' '}" for n in names))
        moves = lambda d, n: E[d][n] > 0 and E[d][n] > p95[n]
        gate = moves("CLEAN_VALENCE", "affect")
        if not gate:
            v = "INVALID"
        else:
            v = "REPLICATES" if moves("CLEAN_UP", "affect") else "DOES_NOT_REPLICATE"
        print(f"  sanity gate (valence moves affect): {'PASS' if gate else 'FAIL'}  ->  layer {L}: {v}"
              f"   [status moves: {moves('CLEAN_UP', 'status')}, widgets move: {moves('CLEAN_UP', 'widget_quantity')}]", flush=True)
        results[str(L)] = dict(verdict=v, gate=gate, effects=E, null_p5=p5, null_p95=p95)
    print(f"\n  >>> GPT-2-medium, layer {PRIMARY}: {results[str(PRIMARY)]['verdict']} <<<")
    print("  other layers: " + ", ".join(f"L{L} {results[str(L)]['verdict']}" for L in LAYERS if L != PRIMARY))
    json.dump(results, open(f"{ROOT}/exp194_results.json", "w"), indent=1)
    print("results saved to exp194_results.json; every cell in exp194_cells.json")


if __name__ == "__main__":
    main()
