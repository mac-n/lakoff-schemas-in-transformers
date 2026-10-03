"""
exp181_entropy_on_request.py — can a model change its attention entropy on
request? Frozen rules: PREREG_exp181.md (commit 300a2e2, 2026-10-02 18:59
IST). Written AFTER the freeze.

Hugging Face transformers with attention outputs (eager attention).
Results are written per sentence as they finish; the run resumes.

Usage:  ./lakoff/bin/python3 exp181_entropy_on_request.py --selftest-only
        ./lakoff/bin/python3 exp181_entropy_on_request.py instruct
        ./lakoff/bin/python3 exp181_entropy_on_request.py base
        ./lakoff/bin/python3 exp181_entropy_on_request.py analyse
"""
import json
import os
import re
import sys

import numpy as np

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)

ALPHA = 0.05
N_PERM = 10000
PREREG = open(f"{ROOT}/PREREG_exp181.md").read()
for _name, _val in [("ALPHA", ALPHA), ("N_PERM", N_PERM)]:
    _m = re.search(rf"\b{_name} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_val), f"{_name} drifted from prereg"

MODELS = {"instruct": "unsloth/Llama-3.2-1B-Instruct", "base": "meta-llama/Llama-3.2-1B"}
TEMPLATES = [
    dict(you=("Keep your attention sharply focused while reading the text below.",
              "Keep your attention loosely relaxed while reading the text below."),
         other=("She kept her attention sharply focused while reading the text below.",
                "She kept her attention loosely relaxed while reading the text below.")),
    dict(you=("Read the next passage with narrow, intense attention.",
              "Read the next passage with broad, gentle attention."),
         other=("They read the next passage with narrow, intense attention.",
                "They read the next passage with broad, gentle attention.")),
    dict(you=("Pay close attention to every word of the text that follows.",
              "Pay loose attention to every word of the text that follows."),
         other=("He paid close attention to every word of the text that follows.",
                "He paid loose attention to every word of the text that follows.")),
    dict(you=("Let your attention narrow to a single point as you read.",
              "Let your attention widen to the whole scene as you read."),
         other=("She let her attention narrow to a single point as she read.",
                "She let her attention widen to the whole scene as she read.")),
]
NEUTRAL = "Read the text below."
CONDS = [("neutral", NEUTRAL)]
for _i, _t in enumerate(TEMPLATES):
    CONDS += [(f"t{_i}_you_focus", _t["you"][0]), (f"t{_i}_you_relax", _t["you"][1]),
              (f"t{_i}_other_focus", _t["other"][0]), (f"t{_i}_other_relax", _t["other"][1])]


def texts():
    from attn_entropy_lib import PROMPTS
    from exp161_balance_entropy_prereg import FRESH_PROMPTS as P161
    from exp166_prompt_verify import CANDIDATE as P166
    out = list(PROMPTS) + list(P161) + list(P166)
    assert len(out) == 120 and len(set(out)) == 120
    return out


def signflip(d, rng):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    return float((np.sum(np.abs((signs * d).mean(axis=1)) >= obs - 1e-15) + 1) / (N_PERM + 1))


def contrast(rows, key, layer=None):
    """rows: {text index: {cond: {key: [per-layer values]}}}. Returns D_you, D_other, I per sentence."""
    idx = sorted(rows)
    def val(i, cond):
        v = rows[i][cond][key]
        return float(np.mean(v)) if layer is None else float(v[layer])
    Dy = np.array([np.mean([val(i, f"t{t}_you_focus") - val(i, f"t{t}_you_relax") for t in range(len(TEMPLATES))]) for i in idx])
    Do = np.array([np.mean([val(i, f"t{t}_other_focus") - val(i, f"t{t}_other_relax") for t in range(len(TEMPLATES))]) for i in idx])
    return Dy, Do, Dy - Do


def stats(d, rng):
    d = np.asarray(d, float)
    return dict(mean=float(d.mean()), d_z=float(d.mean() / d.std(ddof=1)), p=signflip(d, rng))


def verdict(sy, si):
    if sy["mean"] < 0 and sy["p"] < ALPHA:
        return "CONTROL_ON_REQUEST" if (si["mean"] < 0 and si["p"] < ALPHA) else "WORD_EFFECT_ONLY"
    if sy["mean"] > 0 and sy["p"] < ALPHA:
        return "REVERSED"
    return "NULL"


def self_test():
    print("SYNTHETIC SELF-TEST (real verdict path)")
    ok = True
    want = {"control": "CONTROL_ON_REQUEST", "words": "WORD_EFFECT_ONLY", "nothing": "NULL", "reversed": "REVERSED"}
    for kind, expect in want.items():
        rng = np.random.default_rng(3)
        rows = {}
        for i in range(120):
            base = rng.normal(0.5, 0.05)
            r = {}
            for t in range(len(TEMPLATES)):
                for who in ("you", "other"):
                    eff = {"control": -0.02 if who == "you" else 0.0, "words": -0.02, "nothing": 0.0,
                           "reversed": 0.02}[kind]
                    r[f"t{t}_{who}_focus"] = {"within": list(base + eff + rng.normal(0, 0.01, 16))}
                    r[f"t{t}_{who}_relax"] = {"within": list(base + rng.normal(0, 0.01, 16))}
            rows[i] = r
        Dy, Do, I = contrast(rows, "within")
        rng2 = np.random.default_rng(181)
        got = verdict(stats(Dy, rng2), stats(I, rng2))
        good = got == expect
        ok &= good
        print(f"  world {kind}: {got} (want {expect}) {'ok' if good else 'FAIL'}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return ok


def run_model(tag):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[tag]
    tk = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", torch_dtype=torch.float32)
    model.to("mps"); model.eval()
    formats = ["plain", "chat"] if tag == "instruct" else ["plain"]
    T = texts()
    for cond, prefix in CONDS:
        pass
    for t in TEMPLATES:           # equal token length within each focus / relax pair
        for who in ("you", "other"):
            a, b = t[who]
            assert len(tk.encode(a, add_special_tokens=False)) == len(tk.encode(b, add_special_tokens=False))
    for fmt in formats:
        path = f"{ROOT}/exp181_rows_{tag}_{fmt}.jsonl"
        done = set()
        if os.path.exists(path):
            done = {json.loads(line)["text"] for line in open(path)}
            print(f"[{tag} {fmt}] resuming: {len(done)} sentences on disk")
        with open(path, "a") as fh:
            for i, text in enumerate(T):
                if i in done:
                    continue
                row = {"text": i, "conds": {}}
                ref_ids = None
                for cond, prefix in CONDS:
                    content = prefix + "\n\n" + text
                    if fmt == "plain":
                        full = content
                        enc = tk(full, return_offsets_mapping=True, add_special_tokens=True)
                    else:
                        full = tk.apply_chat_template([{"role": "user", "content": content}],
                                                      tokenize=False, add_generation_prompt=True)
                        enc = tk(full, return_offsets_mapping=True, add_special_tokens=False)
                    start_char = full.index(text); end_char = start_char + len(text)
                    offs = enc["offset_mapping"]
                    span = [k for k, (a, b) in enumerate(offs) if a >= start_char and b <= end_char and b > a]
                    s, e = span[0], span[-1] + 1
                    assert span == list(range(s, e))
                    ids = enc["input_ids"]
                    if ref_ids is None:
                        ref_ids = ids[s:e]
                    assert ids[s:e] == ref_ids, "sentence tokens differ between conditions"
                    instr = [k for k, (a, b) in enumerate(offs)
                             if b > a and a >= full.index(prefix) and b <= full.index(prefix) + len(prefix)]
                    x = torch.tensor([ids], device="mps")
                    with torch.no_grad():
                        out = model(x, output_attentions=True)
                    within, full_ent, share = [], [], []
                    for att in out.attentions:                    # [1, heads, T, T]
                        A = att[0].float().cpu().numpy().astype(np.float64)
                        w_q, f_q, s_q = [], [], []
                        for q in range(s + 1, e):
                            p = A[:, q, s:q + 1]
                            p = p / p.sum(axis=1, keepdims=True)
                            p = np.clip(p, 1e-12, 1.0)
                            w_q.append(float((-(p * np.log(p)).sum(axis=1)).mean() / np.log(q - s + 1)))
                            pf = np.clip(A[:, q, :q + 1], 1e-12, 1.0)
                            f_q.append(float((-(pf * np.log(pf)).sum(axis=1)).mean() / np.log(q + 1)))
                            s_q.append(float(A[:, q, instr].sum(axis=1).mean()))
                        within.append(float(np.mean(w_q))); full_ent.append(float(np.mean(f_q)))
                        share.append(float(np.mean(s_q)))
                    row["conds"][cond] = {"within": within, "full": full_ent, "instr_share": share,
                                          "n_text_tokens": e - s, "n_tokens": len(ids)}
                    del out
                fh.write(json.dumps(row) + "\n"); fh.flush()
                if (i + 1) % 20 == 0:
                    print(f"[{tag} {fmt}] sentences {i + 1}/{len(T)} (saved as they finish)", flush=True)
    del model


def load(tag, fmt):
    rows = {}
    for line in open(f"{ROOT}/exp181_rows_{tag}_{fmt}.jsonl"):
        r = json.loads(line); rows[r["text"]] = r["conds"]
    assert len(rows) == 120
    return rows


def analyse():
    rng = np.random.default_rng(181)
    results = {}
    cells = [("instruct", "chat"), ("instruct", "plain"), ("base", "plain")]
    I_store = {}
    for tag, fmt in cells:
        rows = load(tag, fmt)
        print(f"\n{'=' * 72}\n{tag} model, {fmt} format{'   (PRIMARY)' if (tag, fmt) == cells[0] else ''}\n{'=' * 72}")
        cell = {}
        for key, label in (("within", "within-text entropy (PRIMARY measure)"), ("full", "entropy over everything visible"),
                           ("instr_share", "share of attention on the instruction")):
            Dy, Do, I = contrast(rows, key)
            sy, so, si = stats(Dy, rng), stats(Do, rng), stats(I, rng)
            cell[key] = dict(D_you=sy, D_other=so, I=si)
            print(f"  {label}: focus minus relax")
            print(f"     addressed to the model  {sy['mean']:+.5f}  d_z {sy['d_z']:+.2f}  p {sy['p']:.4f}")
            print(f"     about someone else      {so['mean']:+.5f}  d_z {so['d_z']:+.2f}  p {so['p']:.4f}")
            print(f"     difference (I)          {si['mean']:+.5f}  d_z {si['d_z']:+.2f}  p {si['p']:.4f}")
            if key == "within":
                I_store[(tag, fmt)] = I
                base_level = float(np.mean([np.mean(rows[i]["neutral"]["within"]) for i in rows]))
                v = verdict(sy, si)
                cell["verdict"] = v; cell["neutral_level"] = base_level
                print(f"     (neutral-instruction level of the measure: {base_level:.4f})")
                print(f"  >>> {tag} {fmt}: {v} <<<")
                prof = []
                for L in range(16):
                    dy, _, ii = contrast(rows, "within", layer=L)
                    prof.append((float(dy.mean()), float(ii.mean())))
                cell["per_layer"] = prof
                print("     per layer, D_you: " + " ".join(f"{a:+.4f}" for a, _ in prof))
                print("     per layer, I:     " + " ".join(f"{b:+.4f}" for _, b in prof))
        results[f"{tag}_{fmt}"] = cell
    tun = I_store[("instruct", "plain")] - I_store[("base", "plain")]
    st = stats(tun, rng)
    print(f"\nTUNING = I(instruct, plain) − I(base, plain): {st['mean']:+.5f}  d_z {st['d_z']:+.2f}  p {st['p']:.4f}")
    results["tuning"] = st
    json.dump(results, open(f"{ROOT}/exp181_results.json", "w"), indent=1)
    print("\nresults saved to exp181_results.json")


if __name__ == "__main__":
    print("exp181 — attention entropy on request (prereg frozen at 300a2e2)")
    if not self_test():
        sys.exit(1)
    arg = sys.argv[1] if len(sys.argv) > 1 else "--selftest-only"
    if arg == "--selftest-only":
        sys.exit(0)
    if arg == "analyse":
        analyse()
    else:
        run_model(arg)
