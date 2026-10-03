"""exp192_structure_clean.py — Finding 3 (exp123's relational structure) on
clean axes. Frozen rules: PREREG_exp192.md. Written AFTER the freeze. Uses
exp175's cached word states; no model run."""
import json, re, sys
import numpy as np
ROOT = "/Users/macn/Documents/embeddingexp"
K_NULL, SEED_NULL = 100, 192
PREREG = open(f"{ROOT}/PREREG_exp192.md").read()
for n, v in [("K_NULL", K_NULL), ("SEED_NULL", SEED_NULL)]:
    m = re.search(rf"\b{n} = ([0-9]+)", PREREG); assert m and int(m.group(1)) == v
PREDICTED = [("UP-DOWN", "LIGHT-DARK"), ("UP-DOWN", "BALANCE"), ("LIGHT-DARK", "BALANCE"),
             ("FORCE", "DIFFICULTY-BURDEN"), ("UP-DOWN", "FORCE"), ("FORWARD-BACK", "PATH-MOTION")]
z = np.load(f"{ROOT}/exp175_cache_wordstates.npz", allow_pickle=True)
words = list(z["words"]); R = z["R"].astype(np.float64).mean(axis=1)       # [words, 24, d]
ix = {w: i for i, w in enumerate(words)}
W = json.load(open(f"{ROOT}/exp175_stimuli.json"))["words"]
lists = {"UP-DOWN": (W["UP"], W["DOWN"])}
for n, d in W["schemas"].items():
    lists[n] = (d["pos"], d["neg"])
NAMES = sorted(lists)
assert len(NAMES) == 8
NL = R.shape[1]
freq = {}
for L in range(NL):
    f = R[[ix[w] for w in W["COMMON"]], L].mean(0) - R[[ix[w] for w in W["RARE"]], L].mean(0)
    freq[L] = f / np.linalg.norm(f)

def axes_from(lsts, L):
    out = {}
    for n, (a, b) in lsts.items():
        d = R[[ix[w] for w in a], L].mean(0) - R[[ix[w] for w in b], L].mean(0)
        d = d / np.linalg.norm(d); d = d - (d @ freq[L]) * freq[L]
        out[n] = d / np.linalg.norm(d)
    return out

def metrics(lsts):
    mats = []
    for L in range(NL):
        ax = axes_from(lsts, L)
        M = np.array([[ax[a] @ ax[b] for b in NAMES] for a in NAMES]); mats.append(M)
    mats = np.array(mats); iu = np.triu_indices(8, 1)
    sig = mats[:, iu[0], iu[1]]; sig = sig / np.linalg.norm(sig, axis=1, keepdims=True)
    S = sig @ sig.T; m2 = float(S[~np.eye(NL, dtype=bool)].mean())
    pi = {(NAMES.index(a), NAMES.index(b)) for a, b in PREDICTED} | {(NAMES.index(b), NAMES.index(a)) for a, b in PREDICTED}
    pred = [mats[:, i, j].mean() for i, j in zip(*iu) if (i, j) in pi]
    unp = [mats[:, i, j].mean() for i, j in zip(*iu) if (i, j) not in pi]
    return m2, float(np.mean(pred) - np.mean(unp)), mats, float(np.mean(pred)), float(np.mean(unp))

m2, m3, mats, pm, um = metrics(lists)
pool = sorted({w for a, b in lists.values() for w in a + b}); sizes = {n: (len(a), len(b)) for n, (a, b) in lists.items()}
rng = np.random.default_rng(SEED_NULL)
null = []
for k in range(K_NULL):
    perm = list(rng.permutation(pool)); c = 0; fake = {}
    for n in NAMES:
        a, b = sizes[n]; fake[n] = (perm[c:c + a], perm[c + a:c + a + b]); c += a + b
    null.append(metrics(fake)[:2])
null = np.array(null)
p95 = np.percentile(null, 95, axis=0)
print("exp192 — Finding 3 on clean axes (prereg frozen)")
print(f"  M2 configuration similarity across layers: real {m2:.3f} | null median {np.median(null[:,0]):.3f}, 95th {p95[0]:.3f} -> {'stands out' if m2 > p95[0] else 'does not stand out'}   (exp123: 0.91 vs 0.79)")
print(f"  M3 predicted minus unpredicted couplings: real {m3:+.3f} (predicted {pm:+.3f}, unpredicted {um:+.3f}) | null median {np.median(null[:,1]):+.3f}, 95th {p95[1]:+.3f} -> {'stands out' if m3 > p95[1] else 'does not stand out'}   (exp123: +0.214 vs +0.004)")
v = {2: "STRUCTURE_HOLDS", 1: "STRUCTURE_PARTIAL", 0: "STRUCTURE_GONE"}[int(m2 > p95[0]) + int(m3 > p95[1])]
mean_mat = mats.mean(0)
print("  mean cosine matrix across layers:")
print("    " + " ".join(f"{n[:7]:>8}" for n in NAMES))
for i, n in enumerate(NAMES):
    print(f"    {n[:7]:>8} " + " ".join(f"{mean_mat[i,j]:+8.3f}" for j in range(8)))
conn = {n: float(np.abs(mean_mat[i]).sum() - 1) for i, n in enumerate(NAMES)}
print("  predicted pairs, mean cosine and layers positive: " + "; ".join(f"{a}-{b} {mats[:, NAMES.index(a), NAMES.index(b)].mean():+.3f} ({(mats[:, NAMES.index(a), NAMES.index(b)] > 0).sum()}/24)" for a, b in PREDICTED))
print(f"  most-connected axis (sum of |cos| to the others): {max(conn, key=conn.get)}  ({', '.join(f'{n} {c:.2f}' for n, c in sorted(conn.items(), key=lambda x: -x[1]))})")
print(f"\n  >>> {v} <<<")
json.dump({"verdict": v, "M2": m2, "M3": m3, "null_p95": p95.tolist(), "null_median": np.median(null, axis=0).tolist(),
           "predicted_mean": pm, "unpredicted_mean": um, "connectivity": conn}, open(f"{ROOT}/exp192_results.json", "w"), indent=1)
