"""
exp39: LOSS axis for completeness, with comparison to all other candidate
predictive-processing primitives (COHERENCE, SUCCESS_FAILURE, EXIST).

The deeper methodological question raised by Niamh: the UD-FB residual has
goodbad-flavored vocabulary in it (confident, hope, achieve, win) despite
having been V+A-residualized. This may be because A_VALENCE is built from
only one slice of valence (hedonic-pleasant words); other valence dimensions
(achievement-valence, possession-valence, security-valence) live in different
sub-directions that our construction doesn't capture.

LOSS is the canonical loss-aversion / deprivation primitive — testing it
gives us a fourth candidate axis to evaluate alongside COHERENCE, SUCCESS,
EXIST. The key question: are all four of these the same axis with different
vocabularies, or distinct primitives that all just correlate with VALENCE?
"""
import numpy as np
import gensim.downloader as api
from lakoff_canonical_vocabulary import UP_DOWN_MML, FORWARD_BACK_MML, EXISTENCE_MML


print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")


VALENCE_PAIRS = [
    ("pleasant", "unpleasant"), ("desirable", "undesirable"),
    ("agreeable", "disagreeable"), ("enjoyable", "distasteful"),
    ("delightful", "awful"), ("beneficial", "harmful"),
    ("wonderful", "terrible"), ("excellent", "dreadful"),
    ("favorable", "unfavorable"), ("satisfying", "frustrating"),
    ("nice", "nasty"), ("kind", "cruel"),
]
AROUSAL_PAIRS = [
    ("intense", "mild"), ("intense", "gentle"),
    ("alert", "drowsy"), ("urgent", "leisurely"),
    ("frantic", "tranquil"), ("energetic", "lethargic"),
    ("aroused", "relaxed"), ("sharp", "dull"),
    ("acute", "subtle"), ("vivid", "faint"),
    ("electric", "placid"), ("turbulent", "still"),
]
COHERENCE_PAIRS = [
    ("coherent", "incoherent"), ("consistent", "inconsistent"),
    ("aligned", "misaligned"), ("ordered", "disordered"),
    ("organized", "disorganized"), ("harmonious", "discordant"),
    ("predictable", "surprising"), ("predictable", "unpredictable"),
    ("expected", "unexpected"), ("ordinary", "anomalous"),
    ("regular", "irregular"), ("normal", "aberrant"),
    ("orderly", "chaotic"), ("structured", "unstructured"),
    ("uniform", "erratic"),
]
SUCCESS_FAILURE_PAIRS = [
    ("win", "lose"), ("won", "lost"), ("winning", "losing"),
    ("succeed", "fail"), ("succeeded", "failed"), ("success", "failure"),
    ("successful", "unsuccessful"), ("score", "miss"),
    ("scored", "missed"), ("correct", "incorrect"),
    ("accomplish", "fail"), ("achievement", "failure"),
    ("triumph", "defeat"), ("victory", "defeat"),
    ("pass", "fail"), ("passed", "failed"),
    ("hit", "miss"), ("reward", "punishment"),
]
# LOSS: gain-vs-deprivation vocabulary — possession-valence axis
LOSS_PAIRS = [
    ("gain", "loss"),
    ("gained", "lost"),
    ("gaining", "losing"),
    ("profit", "loss"),
    ("abundance", "scarcity"),
    ("fortune", "misfortune"),
    ("security", "threat"),
    ("safety", "danger"),
    ("wealth", "poverty"),
    ("plenty", "lack"),
    ("prosperity", "ruin"),
    ("surplus", "deficit"),
    ("having", "lacking"),
    ("acquired", "deprived"),
    ("rich", "poor"),
    ("affluent", "destitute"),
    ("secure", "vulnerable"),
    ("protected", "exposed"),
    ("safe", "endangered"),
]


def build_axis(pairs, label=""):
    offs, missing = [], []
    for a, c in pairs:
        if a in wv.key_to_index and c in wv.key_to_index:
            offs.append(wv[a] - wv[c])
        else:
            missing.append((a, c))
    if missing and label:
        print(f"  [{label}] missing: {missing}")
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw)


def residualize(v, axes):
    r = v.copy()
    for a in axes:
        r = r - float(r @ a) * a
    nrm = np.linalg.norm(r)
    return r / nrm if nrm > 1e-12 else r


A_val = build_axis(VALENCE_PAIRS)
A_aro = build_axis(AROUSAL_PAIRS)
A_coh = build_axis(COHERENCE_PAIRS)
A_suc = build_axis(SUCCESS_FAILURE_PAIRS)
A_loss = build_axis(LOSS_PAIRS, "LOSS")
A_exist = build_axis(EXISTENCE_MML)
A_ud = build_axis(UP_DOWN_MML)
A_fb = build_axis(FORWARD_BACK_MML)


print("\n=== TEST 1: V/A loading of LOSS (and other candidates for reference) ===")
print(f"  {'axis':>15}  {'cos(V)':>8}  {'cos(A)':>8}  {'%V':>5}  {'%A':>5}")
for name, axis in [("VALENCE", A_val), ("COHERENCE", A_coh), ("SUCCESS", A_suc),
                   ("LOSS", A_loss), ("EXIST", A_exist), ("UD", A_ud)]:
    cv = float(axis @ A_val)
    ca = float(axis @ A_aro)
    print(f"  {name:>15}  {cv:>+8.3f}  {ca:>+8.3f}  {cv**2*100:>4.1f}%  {ca**2*100:>4.1f}%")


print("\n=== TEST 2: Pairwise cosines among candidate primitives (raw) ===")
candidates = {"VALENCE": A_val, "COHERENCE": A_coh, "SUCCESS": A_suc,
              "LOSS": A_loss, "EXIST": A_exist, "UD": A_ud, "FB": A_fb}
names = list(candidates.keys())
print(f"  {'':>10}  " + "  ".join(f"{n:>8}" for n in names))
for n1 in names:
    print(f"  {n1:>10}  " + "  ".join(f"{float(candidates[n1] @ candidates[n2]):>+8.3f}" for n2 in names))


print("\n=== TEST 3: Pairwise cosines after V residualization (NOT V+A — only V) ===")
print("  Removing only valence to see if candidates remain related beyond pure goodbad.")
cands_v = {n: residualize(v, [A_val]) for n, v in candidates.items() if n != "VALENCE"}
v_names = list(cands_v.keys())
print(f"  {'':>10}  " + "  ".join(f"{n:>8}" for n in v_names))
for n1 in v_names:
    print(f"  {n1:>10}  " + "  ".join(f"{float(cands_v[n1] @ cands_v[n2]):>+8.3f}" for n2 in v_names))


print("\n=== TEST 4: Pairwise cosines after V+A residualization ===")
cands_va = {n: residualize(v, [A_val, A_aro]) for n, v in candidates.items() if n != "VALENCE"}
va_names = list(cands_va.keys())
print(f"  {'':>10}  " + "  ".join(f"{n:>8}" for n in va_names))
for n1 in va_names:
    print(f"  {n1:>10}  " + "  ".join(f"{float(cands_va[n1] @ cands_va[n2]):>+8.3f}" for n2 in va_names))


print("\n=== TEST 5: LOSS vs UD-FB residual (after V+A+EXIST) ===")
ud_vae = residualize(A_ud, [A_val, A_aro, A_exist])
fb_vae = residualize(A_fb, [A_val, A_aro, A_exist])
shared_resid = (ud_vae + fb_vae) / 2
shared_resid = shared_resid / np.linalg.norm(shared_resid)
print(f"  cos(residual, LOSS)     = {float(shared_resid @ A_loss):+.4f}")
print(f"  cos(residual, LOSS_VA)  = {float(shared_resid @ residualize(A_loss, [A_val, A_aro])):+.4f}")


print("\n=== TEST 6: LOSS nearest neighbors ===")
print(f"  Positive pole (gain/security/abundance):")
for w, s in wv.similar_by_vector(A_loss, topn=15):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  Negative pole (loss/threat/scarcity):")
for w, s in wv.similar_by_vector(-A_loss, topn=15):
    print(f"    {s:>+.4f}  {w}")

print("\n=== After V+A residualization (LOSS_VA) ===")
loss_va = residualize(A_loss, [A_val, A_aro])
print(f"  Positive pole:")
for w, s in wv.similar_by_vector(loss_va, topn=15):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  Negative pole:")
for w, s in wv.similar_by_vector(-loss_va, topn=15):
    print(f"    {s:>+.4f}  {w}")

np.savez(
    "/Users/macn/Documents/embeddingexp/exp39_results.npz",
    loss=A_loss,
    loss_va=loss_va,
)
print("\nSaved: exp39_results.npz")
