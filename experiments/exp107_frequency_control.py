"""
exp107_frequency_control.py

Niamh: "is there any way to control for the frequency axis [in the decomposition]?"

exp106 used PC1-stripping, but PC1 only ~0.89-correlates with frequency and freq is
also smeared across PC3/PC4 -- a blunt control. Cleaner: build an EXPLICIT frequency
axis by regressing log(rank) on the embeddings (the best linear predictor of
frequency, wherever it lives), project it out, refit & re-decompose the plural
operation. Then quantify how much of the operation was frequency.

Compares the plural operator's decomposition under three controls:
  raw (mean-centered) | PC1-stripped (exp106) | frequency-axis regressed out
and reports, for the UNCONTROLLED operator, how much of its action couples to the
frequency axis (input side ||E f|| and output side ||f^T E||).
"""
import numpy as np
import gensim.downloader as api
from scipy.stats import spearmanr

rng = np.random.default_rng(0)
def unit(v): return v / (np.linalg.norm(v) + 1e-12)
def unit_rows(X): return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)

print("Loading GloVe...")
kv = api.load("glove-wiki-gigaword-300")
vocab = set(kv.index_to_key); mu = kv.vectors.mean(axis=0); d = 300

# ---- frequency axis: regress log(rank) on centered embeddings (top 100k) ----
M = 100000
Xc = kv.vectors[:M] - mu
y = np.log1p(np.arange(M)).astype(float); y -= y.mean()
w_freq, *_ = np.linalg.lstsq(Xc, y, rcond=None)
f = unit(w_freq)
rho_before, _ = spearmanr(Xc @ f, np.arange(M))
# R^2 of rank explained by the single frequency direction
pred = Xc @ w_freq
r2 = 1 - np.sum((y - pred)**2) / np.sum(y**2)
print(f"\n  frequency axis f = unit(regression of log-rank on embeddings)")
print(f"    spearman(X@f, rank) = {rho_before:+.3f}   (linear R^2 of rank ~ X: {r2:.3f})")

# PC1 (exp106's blunt control) for comparison
ev, evec = np.linalg.eigh((Xc.T @ Xc) / M)
pc1 = unit(evec[:, -1])
print(f"    cos(frequency axis, PC1) = {f @ pc1:+.3f}")

# ---- plural pairs ----
pairs = [(wd, wd + "s") for wd in kv.index_to_key[:80000]
         if wd.isalpha() and len(wd) >= 3 and not wd.endswith("s") and (wd + "s") in vocab]
rng.shuffle(pairs); cut = int(0.85 * len(pairs)); tr, te = pairs[:cut], pairs[cut:]
print(f"  plural pairs: {len(pairs)}")

def make_space(kind):
    if kind == "raw":  proj = lambda v: v
    if kind == "pc1":  proj = lambda v: v - (v @ pc1) * pc1
    if kind == "freq": proj = lambda v: v - (v @ f) * f
    return lambda wd: proj(kv[wd] - mu)

def fit_action(vec, lam_frac=0.01):
    A = np.stack([vec(a) for a, b in tr]); B = np.stack([vec(b) for a, b in tr])
    AtA = A.T @ A; lam = lam_frac * np.linalg.eigvalsh(AtA)[-1]
    E = np.linalg.solve(AtA + lam * np.eye(d), A.T @ (B - A))
    Ate = np.stack([vec(a) for a, b in te]); Bte = unit_rows(np.stack([vec(b) for a, b in te]))
    cos = float(np.mean(np.sum(unit_rows(Ate @ (np.eye(d) + E)) * Bte, axis=1)))
    cos_id = float(np.mean(np.sum(unit_rows(Ate) * Bte, axis=1)))
    return E, cos_id, cos

print("\n" + "=" * 74)
print("PLURAL OPERATOR DECOMPOSITION under three frequency controls")
print("=" * 74)
print(f"  {'control':<8}{'cos id':>8}{'cos op':>8}{'||E||':>8}{'sigma1':>8}{'rank@50%':>10}"
      f"{'cos(v1,f)':>11}{'cos(u1,f)':>11}")
for kind in ["raw", "pc1", "freq"]:
    E, cos_id, cos_op = fit_action(make_space(kind))
    U, s, Vt = np.linalg.svd(E)
    energy = np.cumsum(s**2) / (s**2).sum()
    rank50 = int(np.searchsorted(energy, 0.5) + 1)
    cv1 = abs(Vt[0] @ f); cu1 = abs(U[:, 0] @ f)   # does top component align with frequency?
    print(f"  {kind:<8}{cos_id:>8.3f}{cos_op:>8.3f}{np.linalg.norm(E):>8.2f}{s[0]:>8.3f}"
          f"{rank50:>10}{cv1:>11.3f}{cu1:>11.3f}")

# ---- how much of the UNCONTROLLED operator couples to frequency? ----
Eraw, _, _ = fit_action(make_space("raw"))
in_couple = np.linalg.norm(Eraw @ f) / np.linalg.norm(Eraw)      # reads from freq axis
out_couple = np.linalg.norm(f @ Eraw) / np.linalg.norm(Eraw)     # writes to freq axis
E_clean = (np.eye(d) - np.outer(f, f)) @ Eraw @ (np.eye(d) - np.outer(f, f))
removed = 1 - np.linalg.norm(E_clean) / np.linalg.norm(Eraw)
print("\n  Frequency coupling of the UNCONTROLLED (raw) plural operator:")
print(f"    ||E f|| / ||E||  (input couples to frequency):  {in_couple:.3f}")
print(f"    ||f^T E|| / ||E|| (output couples to frequency): {out_couple:.3f}")
print(f"    fraction of ||E|| removed by deflating f both sides: {removed:.3f}")

print("""
READING
  - cos(v1,f)/cos(u1,f) near 1 => the dominant component IS the frequency axis;
    near 0 => the dominant component is something else (e.g. grammatical number).
  - If 'cos op' and the dominant-direction words survive in the 'freq' row, the
    plural operation is real beyond frequency. If they collapse, it was frequency.
  - input/output coupling and the deflation 'removed' fraction say how much of the
    raw operator is just a frequency shift.
""")
