"""
exp201_goal_dynamics.py — what happens to the goal of an instruction while
the model carries it out? Frozen rules: PREREG_exp201.md (commit 85fa970,
3 Oct 2026 09:31 IST). Written AFTER the freeze.
Llama-3.2-1B-Instruct, HF transformers, float32, MPS. Per-prompt results
are cached to disk as they finish (resumes).
"""
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
MAX_NEW = 100
N_PERM = 100
SEED = 201
N_BOOT = 2000
PREREG = open(f"{ROOT}/PREREG_exp201.md").read()
for _n, _v in [("MAX_NEW", MAX_NEW), ("N_PERM", N_PERM), ("SEED", SEED), ("N_BOOT", N_BOOT)]:
    _m = re.search(rf"\b{_n} = ([0-9]+)", PREREG)
    assert _m and int(_m.group(1)) == _v, f"{_n} drifted from prereg"

FORMS = {
    "haiku": "Write a haiku about {t}.",
    "limerick": "Write a limerick about {t}.",
    "list": "Write a numbered list of five facts about {t}.",
    "email": "Write a short email to a friend about {t}.",
    "joke": "Tell a short joke about {t}.",
    "story": "Write a three-sentence story about {t}.",
}
READER = "Write something about {t}."
TOPICS = ["rain", "the sea", "a cat", "coffee", "winter", "a train", "friendship", "the moon", "gardening", "a lost key",
          "mountains", "bread", "a library", "thunder", "a bicycle", "autumn leaves", "a lighthouse", "homework", "a birthday",
          "the city at night"]
FNAMES = list(FORMS)
LAYERS = list(range(2, 15))
POST = "<|start_header_id|>user<|end_header_id|>\n\nThank you!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
CACHE = f"{ROOT}/exp201_cache.npz"


def produced(form, text):
    """Rough report-only check that the asked-for form was produced."""
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if form == "haiku":
        return len(lines) == 3 or (len(lines) == 4 and len(lines[0].split()) <= 6)
    if form == "limerick":
        return 5 <= len(lines) <= 7
    if form == "list":
        return sum(bool(re.match(r"\s*\d+[.)]", l)) for l in lines) >= 4
    if form == "email":
        return bool(re.search(r"^(hi|hey|dear|hello)\b", text.strip(), re.I | re.M)) or "subject" in text.lower()
    if form == "joke":
        return len(text.split()) <= 60
    if form == "story":
        return 2 <= len(re.findall(r"[.!?](\s|$)", text)) <= 5
    return False


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    name = "unsloth/Llama-3.2-1B-Instruct"
    tk = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(dev).eval()
    eos = [tk.convert_tokens_to_ids(x) for x in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>")]
    EOT = tk.convert_tokens_to_ids("<|eot_id|>")
    post_ids = tk.encode(POST, add_special_tokens=False)
    assert tk.convert_tokens_to_ids("<|start_header_id|>") in post_ids, "special tokens not parsed in POST"
    post_ids = [EOT] + post_ids
    print(f"exp201 — goal dynamics (prereg frozen at 85fa970); device {dev}; transformers layers {len(model.model.layers)}", flush=True)

    def prompt_ids(text):
        ids = tk.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return list(ids)

    def hidden(ids):
        with torch.no_grad():
            out = model(torch.tensor([ids], device=dev), output_hidden_states=True)
        # hidden_states[L+1] = output of block L (index 0 = embeddings)
        return out, torch.stack([out.hidden_states[L + 1][0] for L in LAYERS]).float().cpu().numpy()

    items = [(f, ti) for ti in range(len(TOPICS)) for f in FNAMES]
    cache = dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}
    meta = json.loads(str(cache["meta"])) if "meta" in cache else {}

    # ---------- pass 1: generation + prompt-end states at all layers (small; cached as it goes)
    for k, (f, ti) in enumerate(items):
        key = f"{f}|{ti}"
        if f"X0|{key}" in cache:
            continue
        t = TOPICS[ti]
        p = prompt_ids(FORMS[f].format(t=t))
        with torch.no_grad():
            g = model.generate(torch.tensor([p], device=dev), max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=eos,
                               pad_token_id=EOT)
        resp = g[0, len(p):].tolist()
        while resp and resp[-1] in eos:
            resp = resp[:-1]
        text = tk.decode(resp)
        _, H = hidden(p)
        cache[f"X0|{key}"] = H[:, -1, :]                                   # [layers, d] at the last prompt token
        cache[f"resp|{key}"] = np.array(resp)
        cache[f"pid|{key}"] = np.array(p)
        meta[key] = dict(n=len(resp), text=text, produced=bool(produced(f, text)))
        cache["meta"] = np.array(json.dumps(meta))
        np.savez(CACHE, **cache)
        if (k + 1) % 12 == 0:
            print(f"  generated {k + 1}/{len(items)}", flush=True)

    X0 = {L: np.stack([cache[f"X0|{f}|{ti}"][li].astype(np.float64) for f, ti in items]) for li, L in enumerate(LAYERS)}
    yf = np.array([FNAMES.index(f) for f, _ in items]); yt = np.array([ti for _, ti in items])

    # ---------- step 1: topic-held-out ridge one-vs-rest
    def ridge_cv(X, y, groups):
        correct = 0
        for gte in np.unique(groups):
            tr, te = groups != gte, groups == gte
            mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6; Z = (X[tr] - mu) / sd
            Y = -np.ones((tr.sum(), len(FNAMES))); Y[np.arange(tr.sum()), y[tr]] = 1
            W = np.linalg.solve(Z.T @ Z + 10.0 * np.eye(Z.shape[1]), Z.T @ Y)
            correct += int(((((X[te] - mu) / sd) @ W).argmax(1) == y[te]).sum())
        return correct / len(y)

    acc = {L: ridge_cv(X0[L], yf, yt) for L in LAYERS}
    LAYER = max(LAYERS, key=lambda L: (acc[L], -L))
    rng = np.random.default_rng(SEED)
    perm = [ridge_cv(X0[LAYER], rng.permutation(yf), yt) for _ in range(N_PERM)]
    p95 = float(np.percentile(perm, 95))
    gate1 = acc[LAYER] > p95
    print("\nSTEP 1 — which form was asked for, read at the last prompt token (topic held out; chance 0.167)")
    print("  " + "  ".join(f"L{L}:{acc[L]:.2f}" for L in LAYERS))
    print(f"  LAYER = {LAYER} (accuracy {acc[LAYER]:.2f}); permutation 95th {p95:.2f} -> gate {'PASS' if gate1 else 'FAIL'}", flush=True)
    li = LAYERS.index(LAYER)

    # ---------- step 2: patch the prompt-end state
    blk = model.model.layers[LAYER]

    def logp_resp(pids, resp, patch_vec=None):
        handle = None
        if patch_vec is not None:
            pos = len(pids) - 1
            vec = torch.tensor(patch_vec, dtype=torch.float32, device=dev)

            def hook(mod, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                hs[:, pos, :] = vec
                return out
            handle = blk.register_forward_hook(hook)
        try:
            ids = pids + resp
            with torch.no_grad():
                lg = model(torch.tensor([ids], device=dev)).logits[0].float()
            lp = torch.log_softmax(lg, -1)
            s = len(pids) - 1
            return float(np.mean([lp[s + i, tok].item() for i, tok in enumerate(resp)]))
        finally:
            if handle:
                handle.remove()

    s2_path = f"{ROOT}/exp201_step2.json"
    s2 = json.load(open(s2_path)) if os.path.exists(s2_path) else {}
    if str(LAYER) not in s2:
        rows = []
        for f, ti in items:
            fj = FNAMES[(FNAMES.index(f) + 1) % len(FNAMES)]; tk_ = (ti + 1) % len(TOPICS)
            pid = cache[f"pid|{f}|{ti}"].tolist()
            yj = cache[f"resp|{fj}|{ti}"].tolist()[:40]
            src = cache[f"X0|{fj}|{ti}"][li].astype(np.float32)
            ctrl = cache[f"X0|{f}|{tk_}"][li].astype(np.float32)
            base = logp_resp(pid, yj)
            rows.append(dict(real=logp_resp(pid, yj, src) - base, ctrl=logp_resp(pid, yj, ctrl) - base))
        s2[str(LAYER)] = rows; json.dump(s2, open(s2_path, "w"))
    rows = s2[str(LAYER)]
    d = np.array([r["real"] - r["ctrl"] for r in rows])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); gate2 = lo > 0
    print(f"\nSTEP 2 — patch the prompt-end state at layer {LAYER} from another form's prompt")
    print(f"  change in mean log-prob of the SOURCE form's own response: real {np.mean([r['real'] for r in rows]):+.4f}, "
          f"control (same-form source) {np.mean([r['ctrl'] for r in rows]):+.4f}; difference {d.mean():+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}] -> gate {'PASS' if gate2 else 'FAIL'}", flush=True)

    # ---------- step 3: goal signal through and after the response
    keep = [(f, ti) for f, ti in items if meta[f"{f}|{ti}"]["n"] >= 10]
    print(f"\nSTEP 3 — {len(keep)} of {len(items)} responses have at least 10 tokens", flush=True)
    NB = 11  # 10 response bins + POST

    def binned(H, n):
        """H: [tokens, d] from prompt-end onward. returns [NB, d] means and prompt-end vector"""
        resp = H[1:1 + n]; post = H[1 + n:]
        b = np.floor(10 * np.arange(n) / n).astype(int)
        out = np.stack([resp[b == j].mean(0) for j in range(10)] + [post.mean(0)])
        return out, H[0]

    # pass 2: full sequences at LAYER only, instructed and reader, binned per prompt (cached as it goes)
    b_path = f"{ROOT}/exp201_bins_L{LAYER}.npz"
    bc = dict(np.load(b_path)) if os.path.exists(b_path) else {}
    for f, ti in keep:
        key = f"{f}|{ti}"
        if f"ins|{key}" in bc:
            continue
        resp = cache[f"resp|{key}"].tolist(); n = len(resp); p = cache[f"pid|{key}"].tolist()
        r = prompt_ids(READER.format(t=TOPICS[ti]))
        for tag_, pre in (("ins", p), ("rd", r)):
            with torch.no_grad():
                out = model(torch.tensor([pre + resp + post_ids], device=dev), output_hidden_states=True)
            Hs = out.hidden_states[LAYER + 1][0, len(pre) - 1:].float().cpu().numpy().astype(np.float64)
            bc[f"{tag_}|{key}"] = binned(Hs, n)[0]
        np.savez(b_path, **bc)
    B_ins, B_rd, P0 = [], [], []
    for f, ti in keep:
        B_ins.append(bc[f"ins|{f}|{ti}"]); B_rd.append(bc[f"rd|{f}|{ti}"]); P0.append(X0[LAYER][items.index((f, ti))])
    B_ins, B_rd, P0 = np.stack(B_ins), np.stack(B_rd), np.stack(P0)
    kf = np.array([FNAMES.index(f) for f, _ in keep]); kt = np.array([ti for _, ti in keep])
    B_ins_c = B_ins - B_ins.mean(0, keepdims=True); B_rd_c = B_rd - B_rd.mean(0, keepdims=True)
    P0c = P0 - P0.mean(0, keepdims=True)
    X0all = X0[LAYER]

    def dirs_without(topic):
        m = yt != topic; gm = X0all[m].mean(0)
        U = np.stack([X0all[m & (yf == j)].mean(0) - gm for j in range(len(FNAMES))])
        return U / np.linalg.norm(U, axis=1, keepdims=True)

    U_by_topic = {t: dirs_without(t) for t in range(len(TOPICS))}

    def sig(h, f, U):
        pr = U @ h
        return pr[f] - np.mean(np.delete(pr, f))

    S_ins = np.array([[sig(B_ins_c[i, b], kf[i], U_by_topic[kt[i]]) for b in range(NB)] for i in range(len(keep))])
    S_rd = np.array([[sig(B_rd_c[i, b], kf[i], U_by_topic[kt[i]]) for b in range(NB)] for i in range(len(keep))])
    S0 = np.array([sig(P0c[i], kf[i], U_by_topic[kt[i]]) for i in range(len(keep))])
    D = S_ins - S_rd                                    # per prompt, per bin

    def stats(idx):
        G = D[idx].mean(0) / S0[idx].mean()
        x = np.arange(10); slope = np.polyfit(x, G[:10], 1)[0]
        return G, slope

    G, slope = stats(np.arange(len(keep)))
    boots = [stats(rng.integers(0, len(keep), len(keep))) for _ in range(N_BOOT)]
    Gb = np.stack([b[0] for b in boots]); Sb = np.array([b[1] for b in boots])
    ci = lambda a: np.percentile(a, [2.5, 97.5])
    g1 = ci(Gb[:, 0]); mid = Gb[:, 3:7].mean(1)
    d_mid_1 = ci(Gb[:, 0] - mid); d_mid_10 = ci(Gb[:, 9] - mid); sl = ci(Sb)
    held_d = ci(Gb[:, :10].mean(1) - Gb[:, 10])
    print(f"  prompt-end goal signal S0 = {S0.mean():+.3f} (instructed); G below in units of S0, instructed minus reader")
    print("  bin:   " + "  ".join(f"{b + 1:>6}" for b in range(10)) + "    POST")
    print("  G:     " + "  ".join(f"{G[b]:>+6.2f}" for b in range(10)) + f"  {G[10]:>+6.2f}")
    print("  ins:   " + "  ".join(f"{S_ins[:, b].mean() / S0.mean():>+6.2f}" for b in range(10)) + f"  {S_ins[:, 10].mean() / S0.mean():>+6.2f}")
    print("  reader:" + "  ".join(f"{S_rd[:, b].mean() / S0.mean():>+6.2f}" for b in range(10)) + f"  {S_rd[:, 10].mean() / S0.mean():>+6.2f}")
    print(f"  G(bin 1) CI [{g1[0]:+.2f}, {g1[1]:+.2f}]; slope over bins {slope:+.3f} CI [{sl[0]:+.3f}, {sl[1]:+.3f}]")
    print(f"  bin1 − mid(4–7) CI [{d_mid_1[0]:+.2f}, {d_mid_1[1]:+.2f}]; bin10 − mid CI [{d_mid_10[0]:+.2f}, {d_mid_10[1]:+.2f}]; "
          f"mean(bins) − POST CI [{held_d[0]:+.2f}, {held_d[1]:+.2f}]")
    mean_resp = G[:10].mean()
    if g1[0] <= 0:
        v = "NO_GOAL_STATE"
    elif d_mid_1[0] > 0 and d_mid_10[0] > 0:
        v = "U_SHAPED"
    elif sl[1] < 0 and G[9] < 0.5 * G[0]:
        v = "SPENT"
    elif sl[1] >= 0 and held_d[0] > 0 and G[10] < 0.5 * mean_resp:
        v = "HELD"
    else:
        v = "OTHER"
    tag = "" if (gate1 and gate2) else " (UNGATED: " + ", ".join(x for x, ok in (("step 1", gate1), ("step 2", gate2)) if not ok) + " failed)"
    print("\n  G by form (bins 1, 5, 10, POST):")
    for j, f in enumerate(FNAMES):
        m = kf == j
        if m.sum():
            Gf = D[m].mean(0) / S0[m].mean()
            print(f"    {f:9s} n={m.sum():2d} produced {np.mean([meta[f'{f}|{ti}']['produced'] for ti in kt[m]]):.2f}   "
                  f"{Gf[0]:+.2f} {Gf[4]:+.2f} {Gf[9]:+.2f} {Gf[10]:+.2f}")
    print(f"\n  >>> STEP 1 {'PASS' if gate1 else 'FAIL'} | STEP 2 {'PASS' if gate2 else 'FAIL'} | STEP 3: {v}{tag} <<<")
    json.dump(dict(layer=LAYER, step1_acc={str(L): a for L, a in acc.items()}, step1_p95=p95, gate1=bool(gate1),
                   step2_diff=float(d.mean()), step2_ci=[float(lo), float(hi)], gate2=bool(gate2),
                   G=G.tolist(), slope=float(slope), slope_ci=sl.tolist(), verdict=v, n_kept=len(keep),
                   S_ins=(S_ins.mean(0) / S0.mean()).tolist(), S_rd=(S_rd.mean(0) / S0.mean()).tolist()),
              open(f"{ROOT}/exp201_results.json", "w"), indent=1)
    print("results saved to exp201_results.json")


if __name__ == "__main__":
    main()
