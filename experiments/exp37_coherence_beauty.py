"""
exp37: Test COHERENCE and BEAUTIFUL/UGLY as candidate underlying primitives
for the salience-cluster asymmetry and the UD-FB residual.

Theoretical hypothesis: the asymmetric loading we keep finding on the salience
cluster (DOWN-dominant, DARK-dominant, BACK-dominant, etc.) might be the
COHERENCE-vs-ANTI-COHERENCE primitive expressed in different metaphorical
vocabularies. In predictive-processing terms: states matching prediction
(coherent) are baseline; states violating prediction (incoherent, surprising,
disruptive) are salience-loaded.

Two candidate constructions:
  1. COHERENCE: predicted-vs-disrupted vocabulary (coherent/incoherent,
     consistent/inconsistent, ordered/disordered, predictable/surprising, etc.)
  2. BEAUTIFUL/UGLY: aesthetic vocabulary, framed in the Reber/Winkielman
     processing-fluency tradition (beauty as cognitive ease)

Tests:
  - cos(each candidate, V) and cos(each, A) — how much affect/arousal loading?
  - cos(COHERENCE, BEAUTIFUL/UGLY) — same axis or different?
  - cos(each, DIFF) — does it absorb DIFF? (DIFF is anti-cluster in exp34)
  - cos(each, EXISTENCE) — does coherence-realization map onto becoming?
  - cos(each, UD-FB-residual_VAE) — does it explain the fourth axis?
  - After residualizing UD and FB with V+A+COHERENCE: does cos(UD,FB) collapse?
  - Nearest neighbors of each candidate to verify semantic shape
"""
import numpy as np
import gensim.downloader as api
from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML, LIGHT_DARK_MML,
    EXISTENCE_MML, DIFFICULTY_BURDEN_MML, PATH_MOTION_MML,
)


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

# COHERENCE: predicted-state vs disrupted-state vocabulary.
# Avoids Lakoff schema words. Tries to capture predictive-processing structural
# axis: states matching expectation vs prediction-error states.
COHERENCE_PAIRS = [
    ("coherent", "incoherent"),
    ("consistent", "inconsistent"),
    ("aligned", "misaligned"),
    ("ordered", "disordered"),
    ("organized", "disorganized"),
    ("harmonious", "discordant"),
    ("predictable", "surprising"),
    ("predictable", "unpredictable"),
    ("expected", "unexpected"),
    ("ordinary", "anomalous"),
    ("regular", "irregular"),
    ("normal", "aberrant"),
    ("orderly", "chaotic"),
    ("structured", "unstructured"),
    ("uniform", "erratic"),
]

# BEAUTIFUL/UGLY: aesthetic vocabulary. Will have substantial valence loading
# that we'll residualize away. The question is what's left semantically.
BEAUTIFUL_UGLY_PAIRS = [
    ("beautiful", "ugly"),
    ("pretty", "hideous"),
    ("lovely", "repulsive"),
    ("gorgeous", "grotesque"),
    ("attractive", "unattractive"),
    ("elegant", "awkward"),
    ("graceful", "clumsy"),
    ("exquisite", "ghastly"),
    ("handsome", "ugly"),
    ("charming", "repugnant"),
    ("aesthetic", "unaesthetic"),
    ("refined", "crude"),
    ("polished", "rough"),
    ("delicate", "coarse"),
]


def build_axis(pairs, label=""):
    offs = []
    missing = []
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


print("\n=== Building axes ===")
A_val = build_axis(VALENCE_PAIRS)
A_aro = build_axis(AROUSAL_PAIRS)
A_coh = build_axis(COHERENCE_PAIRS, "COHERENCE")
A_bea = build_axis(BEAUTIFUL_UGLY_PAIRS, "BEAUTIFUL_UGLY")

A_ud = build_axis(UP_DOWN_MML)
A_io_clean = build_axis(IN_OUT_MML_CLEAN)
A_fb = build_axis(FORWARD_BACK_MML)
A_ld = build_axis(LIGHT_DARK_MML)
A_exist = build_axis(EXISTENCE_MML)
A_diff = build_axis(DIFFICULTY_BURDEN_MML)
A_path = build_axis(PATH_MOTION_MML)


# =================================================================
# TEST 1: How much valence/arousal in each candidate?
# =================================================================
print("\n=== TEST 1: Affect-loading of candidate primitives ===")
print(f"  {'axis':>15}  {'cos(V)':>8}  {'cos(A)':>8}  {'%V':>5}  {'%A':>5}  {'%resid':>7}")
for name, axis in [("COHERENCE", A_coh), ("BEAUTIFUL_UGLY", A_bea)]:
    cv = float(axis @ A_val)
    ca = float(axis @ A_aro)
    r = axis - cv*A_val - ca*A_aro
    rn = float(np.linalg.norm(r))
    print(f"  {name:>15}  {cv:>+8.3f}  {ca:>+8.3f}  {cv**2*100:>4.1f}%  {ca**2*100:>4.1f}%  {rn**2*100:>6.1f}%")


# =================================================================
# TEST 2: Are COHERENCE and BEAUTIFUL/UGLY the same axis?
# =================================================================
print("\n=== TEST 2: COHERENCE vs BEAUTIFUL_UGLY ===")
print(f"  cos(COH, BEAUTY)         = {float(A_coh @ A_bea):+.4f}")
print(f"  cos(COH_VA, BEAUTY_VA)   = {float(residualize(A_coh, [A_val, A_aro]) @ residualize(A_bea, [A_val, A_aro])):+.4f}")
print(f"  High both ways → same underlying axis. Higher VA-residualized = robust shared content.")


# =================================================================
# TEST 3: Do candidates absorb the salience cluster + DIFF + UD-FB residual?
# =================================================================
print("\n=== TEST 3: Candidate alignments with cluster axes ===")
schemas = {
    "UD": A_ud, "IO_CLEAN": A_io_clean, "FB": A_fb, "LD": A_ld,
    "EXIST": A_exist, "DIFF": A_diff, "PATH": A_path,
}
print(f"  {'':>15}  " + "  ".join(f"{n:>8}" for n in schemas))
for cand_name, cand in [("COHERENCE", A_coh), ("BEAUTIFUL_UGLY", A_bea)]:
    print(f"  {cand_name:>15}  " + "  ".join(f"{float(cand @ axis):>+8.3f}" for axis in schemas.values()))

# Same after V+A residualization
print("\n=== TEST 3b: After V+A residualization of everything ===")
schemas_va = {name: residualize(axis, [A_val, A_aro]) for name, axis in schemas.items()}
coh_va = residualize(A_coh, [A_val, A_aro])
bea_va = residualize(A_bea, [A_val, A_aro])
print(f"  {'':>15}  " + "  ".join(f"{n:>8}" for n in schemas))
for cand_name, cand in [("COHERENCE_VA", coh_va), ("BEAUTIFUL_UGLY_VA", bea_va)]:
    print(f"  {cand_name:>15}  " + "  ".join(f"{float(cand @ axis):>+8.3f}" for axis in schemas_va.values()))


# =================================================================
# TEST 4: Does COHERENCE absorb the UD-FB residual?
# =================================================================
print("\n=== TEST 4: Candidates vs UD-FB shared residual (after V+A+EXIST) ===")
ud_vae = residualize(A_ud, [A_val, A_aro, A_exist])
fb_vae = residualize(A_fb, [A_val, A_aro, A_exist])
shared_resid = (ud_vae + fb_vae) / 2
shared_resid = shared_resid / np.linalg.norm(shared_resid)

print(f"  cos(shared_residual, COHERENCE)      = {float(shared_resid @ A_coh):+.4f}")
print(f"  cos(shared_residual, COHERENCE_VA)   = {float(shared_resid @ coh_va):+.4f}")
print(f"  cos(shared_residual, BEAUTIFUL_UGLY) = {float(shared_resid @ A_bea):+.4f}")
print(f"  cos(shared_residual, BEAUTY_VA)      = {float(shared_resid @ bea_va):+.4f}")


# =================================================================
# TEST 5: Triple residualization with V+A+COHERENCE — does UD-FB collapse?
# =================================================================
print("\n=== TEST 5: UD-FB cosine through residualization stages ===")
ud_v = residualize(A_ud, [A_val])
fb_v = residualize(A_fb, [A_val])
ud_va = residualize(A_ud, [A_val, A_aro])
fb_va = residualize(A_fb, [A_val, A_aro])
ud_vac = residualize(A_ud, [A_val, A_aro, A_coh])
fb_vac = residualize(A_fb, [A_val, A_aro, A_coh])
ud_vae_full = residualize(A_ud, [A_val, A_aro, A_exist])
fb_vae_full = residualize(A_fb, [A_val, A_aro, A_exist])
ud_full = residualize(A_ud, [A_val, A_aro, A_exist, A_coh])
fb_full = residualize(A_fb, [A_val, A_aro, A_exist, A_coh])

print(f"  Raw:                              cos(UD, FB) = {float(A_ud @ A_fb):+.4f}")
print(f"  V removed:                        cos(UD_V, FB_V) = {float(ud_v @ fb_v):+.4f}")
print(f"  V+A removed:                      cos(UD_VA, FB_VA) = {float(ud_va @ fb_va):+.4f}")
print(f"  V+A+EXIST removed:                cos = {float(ud_vae_full @ fb_vae_full):+.4f}")
print(f"  V+A+COHERENCE removed:            cos = {float(ud_vac @ fb_vac):+.4f}")
print(f"  V+A+EXIST+COHERENCE removed:      cos = {float(ud_full @ fb_full):+.4f}")
print(f"  ^^^ If COHERENCE explains residual: V+A+COH gets cos(UD,FB) to near zero.")


# =================================================================
# TEST 6: Same for UD-LD and FB-LD
# =================================================================
print("\n=== TEST 6: UD-LD and FB-LD through V+A+COHERENCE ===")
ld_vac = residualize(A_ld, [A_val, A_aro, A_coh])
print(f"  Raw:           cos(UD, LD) = {float(A_ud @ A_ld):+.4f}    cos(FB, LD) = {float(A_fb @ A_ld):+.4f}")
print(f"  V+A removed:   cos = {float(ud_va @ residualize(A_ld, [A_val, A_aro])):+.4f}    cos = {float(fb_va @ residualize(A_ld, [A_val, A_aro])):+.4f}")
print(f"  V+A+COH:       cos = {float(ud_vac @ ld_vac):+.4f}    cos = {float(fb_vac @ ld_vac):+.4f}")


# =================================================================
# TEST 7: Nearest neighbors of COHERENCE and BEAUTIFUL_UGLY
# =================================================================
print("\n=== TEST 7: Nearest neighbors of COHERENCE ===")
print(f"  Positive pole (coherence):")
for w, s in wv.similar_by_vector(A_coh, topn=15):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  Negative pole (incoherence / disruption):")
for w, s in wv.similar_by_vector(-A_coh, topn=15):
    print(f"    {s:>+.4f}  {w}")

print("\n=== After V+A residualization (COHERENCE_VA) ===")
print(f"  Positive pole:")
for w, s in wv.similar_by_vector(coh_va, topn=15):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  Negative pole:")
for w, s in wv.similar_by_vector(-coh_va, topn=15):
    print(f"    {s:>+.4f}  {w}")


print("\n=== TEST 8: Nearest neighbors of BEAUTIFUL_UGLY ===")
print(f"  Positive pole (beautiful):")
for w, s in wv.similar_by_vector(A_bea, topn=15):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  Negative pole (ugly):")
for w, s in wv.similar_by_vector(-A_bea, topn=15):
    print(f"    {s:>+.4f}  {w}")

print("\n=== After V+A residualization (BEAUTY_VA) ===")
print(f"  Positive pole:")
for w, s in wv.similar_by_vector(bea_va, topn=15):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  Negative pole:")
for w, s in wv.similar_by_vector(-bea_va, topn=15):
    print(f"    {s:>+.4f}  {w}")


# =================================================================
# Save
# =================================================================
np.savez(
    "/Users/macn/Documents/embeddingexp/exp37_results.npz",
    coherence=A_coh,
    beautiful_ugly=A_bea,
    coherence_va=coh_va,
    beauty_va=bea_va,
    shared_resid=shared_resid,
)
print("\nSaved: exp37_results.npz")
