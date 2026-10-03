"""
exp202_progress_or_distance.py — is the goal spent by progress, or does it
fade with distance? Frozen rules: PREREG_exp202.md (commit a5856b9,
3 Oct 2026 09:50 IST). Written AFTER the freeze.
Llama-3.2-1B-Instruct, HF transformers, float32, MPS. Caches as it goes.
"""
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)
from exp201_goal_dynamics import TOPICS, READER, POST

MAX_NEW = 600
LEN_RATIO = 15
N_BOOT = 2000
SEED = 202
PREREG = open(f"{ROOT}/PREREG_exp202.md").read()
for _n, _v in [("MAX_NEW", MAX_NEW), ("LEN_RATIO", LEN_RATIO), ("N_BOOT", N_BOOT), ("SEED", SEED)]:
    _m = re.search(rf"\b{_n} = ([0-9]+)", PREREG)
    assert _m and int(_m.group(1)) == _v, f"{_n} drifted from prereg"
FORMS = {
    "haiku": ("Write one haiku about {t}.", "Write three haikus about {t}."),
    "limerick": ("Write one limerick about {t}.", "Write three limericks about {t}."),
    "facts": ("Write a numbered list of three facts about {t}.", "Write a numbered list of eight facts about {t}."),
    "story": ("Write a two-sentence story about {t}.", "Write a six-sentence story about {t}."),
}
FN = list(FORMS)
LAYERS = [2, 8]
PRIMARY = 2
CACHE = f"{ROOT}/exp202_cache.npz"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    name = "unsloth/Llama-3.2-1B-Instruct"
    tk = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev).eval()
    eos = [tk.convert_tokens_to_ids(x) for x in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>")]
    EOT = eos[0]
    post_ids = [EOT] + tk.encode(POST, add_special_tokens=False)
    print(f"exp202 — progress or distance? (prereg frozen at a5856b9); device {dev}", flush=True)

    def prompt_ids(text):
        ids = tk.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
        return list(ids["input_ids"] if not isinstance(ids, list) else ids)

    items = [(f, li_, ti) for ti in range(len(TOPICS)) for f in FN for li_ in (0, 1)]   # li_ 0 short, 1 long
    cache = dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}
    meta = json.loads(str(cache["meta"])) if "meta" in cache else {}

    # ---- pass 1: generate; prompt-end states at LAYERS
    for k, (f, l, ti) in enumerate(items):
        key = f"{f}|{l}|{ti}"
        if f"X0|{key}" in cache:
            continue
        p = prompt_ids(FORMS[f][l].format(t=TOPICS[ti]))
        with torch.no_grad():
            g = model.generate(torch.tensor([p], device=dev), attention_mask=torch.ones(1, len(p), dtype=torch.long, device=dev),
                               max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=eos, pad_token_id=EOT)
            out = model(torch.tensor([p], device=dev), output_hidden_states=True)
        resp = g[0, len(p):].tolist()
        finished = bool(resp and resp[-1] in eos)
        while resp and resp[-1] in eos:
            resp = resp[:-1]
        cache[f"X0|{key}"] = np.stack([out.hidden_states[L + 1][0, -1].float().cpu().numpy() for L in LAYERS])
        cache[f"resp|{key}"] = np.array(resp); cache[f"pid|{key}"] = np.array(p)
        meta[key] = dict(n=len(resp), finished=finished, text=tk.decode(resp))
        cache["meta"] = np.array(json.dumps(meta)); np.savez(CACHE, **cache)
        if (k + 1) % 16 == 0:
            print(f"  generated {k + 1}/{len(items)}", flush=True)

    yf = np.array([FN.index(f) for f, _, _ in items]); yt = np.array([ti for _, _, ti in items])
    X0 = {L: np.stack([cache[f"X0|{f}|{l}|{ti}"][j].astype(np.float64) for f, l, ti in items]) for j, L in enumerate(LAYERS)}

    def dirs_without(L, topic):
        m = yt != topic; gm = X0[L][m].mean(0)
        U = np.stack([X0[L][m & (yf == j)].mean(0) - gm for j in range(len(FN))])
        return U / np.linalg.norm(U, axis=1, keepdims=True)

    U = {L: {t: dirs_without(L, t) for t in range(len(TOPICS))} for L in LAYERS}
    gm0 = {L: X0[L].mean(0) for L in LAYERS}

    def sig_vec(P, f):          # P: [..., n_forms] projections
        return P[..., f] - (P.sum(-1) - P[..., f]) / (P.shape[-1] - 1)

    # ---- pass 2: per-token projections, instructed and reader (cached)
    p_path = f"{ROOT}/exp202_proj.npz"
    pc = dict(np.load(p_path)) if os.path.exists(p_path) else {}
    done = [(f, l, ti) for f, l, ti in items if meta[f"{f}|{l}|{ti}"]["finished"] and meta[f"{f}|{l}|{ti}"]["n"] >= 10]
    for f, l, ti in done:
        key = f"{f}|{l}|{ti}"
        if f"ins|{key}" in pc:
            continue
        resp = cache[f"resp|{key}"].tolist(); p = cache[f"pid|{key}"].tolist(); r = prompt_ids(READER.format(t=TOPICS[ti]))
        for tag, pre in (("ins", p), ("rd", r)):
            with torch.no_grad():
                out = model(torch.tensor([pre + resp + post_ids], device=dev), output_hidden_states=True)
            arr = []
            for L in LAYERS:
                H = out.hidden_states[L + 1][0, len(pre):].float().cpu().numpy().astype(np.float64)   # answer + post tokens
                arr.append(H @ U[L][ti].T)                                                          # [tokens, forms]
            pc[f"{tag}|{key}"] = np.stack(arr)
        np.savez(p_path, **pc)
    print(f"  finished answers: {len(done)} of {len(items)}", flush=True)

    rng = np.random.default_rng(SEED)
    results = {}
    for j, L in enumerate(LAYERS):
        S0 = np.mean([sig_vec((X0[L][items.index(it)] - gm0[L]) @ U[L][it[2]].T, FN.index(it[0])) for it in done])
        D, Post = {}, {}
        for f, l, ti in done:
            key = f"{f}|{l}|{ti}"; n = meta[key]["n"]; fi = FN.index(f)
            d = (sig_vec(pc[f"ins|{key}"][j], fi) - sig_vec(pc[f"rd|{key}"][j], fi)) / S0
            D[(f, l, ti)] = d[:n]; Post[(f, l, ti)] = float(d[n:].mean())
        # pooled fraction-done curve (all finished answers)
        def frac_bins(d):
            n = len(d); b = np.floor(10 * np.arange(n) / n).astype(int)
            return np.array([d[b == q].mean() for q in range(10)])
        FB = {k_: frac_bins(v) for k_, v in D.items()}
        # pairs
        pairs = []
        for f in FN:
            for ti in range(len(TOPICS)):
                s, lg = (f, 0, ti), (f, 1, ti)
                if s in D and lg in D and len(D[lg]) >= LEN_RATIO / 10 * len(D[s]):
                    pairs.append((f, ti))
        Kf = {f: int(np.median([len(D[(f, 0, ti)]) for ff, ti in pairs if ff == f])) for f in FN if any(ff == f for ff, _ in pairs)}

        def tok_bins(d, K):
            out = np.full(10, np.nan); w = K / 10
            for q in range(10):
                lo, hi = int(round(q * w)), int(round((q + 1) * w))
                if hi > lo and lo < len(d):
                    out[q] = d[lo:min(hi, len(d))].mean()
            return out

        def deltas(topics_sample):
            per_form_p, per_form_k = [], []
            for f in Kf:
                ps = [(ff, ti) for ff, ti in pairs if ff == f]
                sel = [pt for t in topics_sample for pt in ps if pt[1] == t]
                if not sel:
                    continue
                dp = np.mean([FB[(f, 1, ti)][1:] - FB[(f, 0, ti)][1:] for _, ti in sel])
                dk_rows = [tok_bins(D[(f, 1, ti)], Kf[f]) - tok_bins(D[(f, 0, ti)], Kf[f]) for _, ti in sel]
                per_form_p.append(dp); per_form_k.append(np.nanmean(np.stack(dk_rows)))
            return np.mean(per_form_p), np.mean(per_form_k)

        all_t = np.arange(len(TOPICS))
        dp, dk = deltas(all_t)
        B = np.array([deltas(rng.choice(all_t, len(all_t))) for _ in range(N_BOOT)])
        ci_p, ci_k = np.percentile(B[:, 0], [2.5, 97.5]), np.percentile(B[:, 1], [2.5, 97.5])
        allfb = np.stack(list(FB.values()))
        slope_b = [np.polyfit(np.arange(10), allfb[rng.integers(0, len(allfb), len(allfb))].mean(0), 1)[0] for _ in range(N_BOOT)]
        ci_s = np.percentile(slope_b, [2.5, 97.5])
        if ci_s[0] <= 0 <= ci_s[1] or ci_s[0] > 0:
            v = "NO_DECLINE"
        elif ci_k[0] > 0 and ci_p[0] <= 0 <= ci_p[1]:
            v = "PROGRESS"
        elif ci_p[1] < 0 and ci_k[0] <= 0 <= ci_k[1]:
            v = "DISTANCE"
        elif ci_k[0] > 0 and ci_p[1] < 0:
            v = "BOTH"
        else:
            v = "UNRESOLVED"
        print(f"\n=== layer {L}{' (PRIMARY)' if L == PRIMARY else ''}: S0 {S0:+.3f}; {len(pairs)} pairs kept ===")
        print(f"  pooled fraction-done curve: " + " ".join(f"{x:+.2f}" for x in allfb.mean(0)) +
              f" | POST {np.mean(list(Post.values())):+.2f}; slope CI [{ci_s[0]:+.3f}, {ci_s[1]:+.3f}]")
        for f in Kf:
            ps = [ti for ff, ti in pairs if ff == f]
            ratio = np.median([len(D[(f, 1, ti)]) / len(D[(f, 0, ti)]) for ti in ps])
            print(f"  {f:9s} pairs {len(ps):2d}, length ratio {ratio:.1f} (short ~{Kf[f]} tok)")
            print(f"     by fraction done  short: " + " ".join(f"{x:+.2f}" for x in np.mean([FB[(f, 0, ti)] for ti in ps], 0)))
            print(f"                       long:  " + " ".join(f"{x:+.2f}" for x in np.mean([FB[(f, 1, ti)] for ti in ps], 0)))
            ts = np.nanmean([tok_bins(D[(f, 0, ti)], Kf[f]) for ti in ps], 0); tl = np.nanmean([tok_bins(D[(f, 1, ti)], Kf[f]) for ti in ps], 0)
            print(f"     by tokens         short: " + " ".join(f"{x:+.2f}" for x in ts))
            print(f"                       long:  " + " ".join(f"{x:+.2f}" for x in tl))
        print(f"  Delta_p (long − short at same fraction done) {dp:+.3f} CI [{ci_p[0]:+.3f}, {ci_p[1]:+.3f}]")
        print(f"  Delta_k (long − short at same token distance) {dk:+.3f} CI [{ci_k[0]:+.3f}, {ci_k[1]:+.3f}]")
        print(f"  >>> layer {L}: {v} <<<", flush=True)
        results[str(L)] = dict(verdict=v, pairs=len(pairs), dp=float(dp), dk=float(dk), ci_p=ci_p.tolist(), ci_k=ci_k.tolist(),
                               slope_ci=ci_s.tolist(), curve=allfb.mean(0).tolist(), post=float(np.mean(list(Post.values()))))
    print(f"\n  >>> exp202 (layer {PRIMARY}): {results[str(PRIMARY)]['verdict']} <<<")
    json.dump(results, open(f"{ROOT}/exp202_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
