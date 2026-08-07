"""
exp30: Nearest-neighbor probe of the UD-FB shared residual direction in word2vec.

We've shown the UD-FB residual (after V+A+PATH removal) is +0.40 but doesn't
align with any named axis we've tried. Instead of constructing more candidate
axes, take the residual direction itself and ask: what words live near this
direction in word2vec? What's semantically there?

Three directions to probe:
  1. The shared UD-FB residual (the +0.40 alignment after V+A+PATH)
  2. UD-only residual (UD after subtracting V+A+PATH+FB-shared)
  3. FB-only residual (FB after subtracting V+A+PATH+UD-shared)

For each direction, report the top-K most-aligned words on each pole.
"""
import numpy as np
import gensim.downloader as api

print("Loading glove-wiki-gigaword-300 (cached)...")
wv = api.load("glove-wiki-gigaword-300")

# Reuse all the same anchor pairs and schema pairs
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
PATH_PAIRS = [
    ("moving", "stationary"), ("moved", "stayed"),
    ("traveled", "remained"), ("travels", "remains"),
    ("traveling", "staying"), ("journey", "rest"),
    ("going", "staying"), ("departed", "stayed"),
    ("shifted", "fixed"), ("migrated", "settled"),
    ("wandered", "paused"), ("flowing", "static"),
    ("flowing", "still"), ("dynamic", "static"),
    ("motion", "rest"), ("mobile", "immobile"),
    ("changing", "constant"), ("drifted", "anchored"),
    ("roamed", "stopped"), ("transit", "stationary"),
]
UD_PAIRS = [
    ("rose", "fell"), ("rising", "falling"), ("climbing", "descending"),
    ("ascended", "descended"), ("soaring", "plunging"), ("lifted", "sunk"),
    ("higher", "lower"), ("upward", "downward"), ("up", "down"),
    ("happy", "sad"), ("cheerful", "gloomy"), ("elated", "dejected"),
    ("joyful", "sorrowful"), ("uplifted", "depressed"),
    ("more", "less"), ("increase", "decrease"), ("grew", "shrank"),
    ("expanded", "contracted"), ("multiplied", "dwindled"),
    ("promoted", "demoted"), ("prestigious", "disgraced"),
    ("esteemed", "despised"), ("eminent", "lowly"),
    ("healthy", "sick"), ("thriving", "ailing"), ("strong", "weak"),
    ("vigorous", "feeble"), ("robust", "frail"),
]
FB_PAIRS = [
    ("forward", "backward"), ("advance", "retreat"), ("ahead", "behind"),
    ("forwards", "backwards"), ("progress", "regress"), ("improved", "declined"),
    ("gained", "lost"), ("advanced", "retreated"),
    ("future", "past"), ("next", "previous"), ("later", "earlier"),
    ("upcoming", "former"), ("evolved", "devolved"), ("developed", "regressed"),
    ("onward", "return"),
]


def build_axis(pairs):
    offsets = []
    for a, c in pairs:
        if a in wv.key_to_index and c in wv.key_to_index:
            offsets.append(wv[a] - wv[c])
    raw = np.stack(offsets).mean(axis=0)
    return raw / np.linalg.norm(raw)


A_val = build_axis(VALENCE_PAIRS)
A_aro = build_axis(AROUSAL_PAIRS)
A_path = build_axis(PATH_PAIRS)
A_ud = build_axis(UD_PAIRS)
A_fb = build_axis(FB_PAIRS)


def residualize(v, axes_to_remove):
    r = v.copy()
    for a in axes_to_remove:
        r = r - float(r @ a) * a
    nrm = np.linalg.norm(r)
    return r / nrm if nrm > 1e-12 else r


# UD after V+A+PATH; FB after V+A+PATH
ud_vap = residualize(A_ud, [A_val, A_aro, A_path])
fb_vap = residualize(A_fb, [A_val, A_aro, A_path])

# Shared UD-FB direction (the +0.40 residual). Average of the two normalized residuals.
shared = (ud_vap + fb_vap) / 2
shared = shared / np.linalg.norm(shared)

# UD-only: UD residual after removing the shared component too
ud_specific = ud_vap - float(ud_vap @ shared) * shared
ud_specific = ud_specific / np.linalg.norm(ud_specific)

# FB-only: FB residual after removing the shared component too
fb_specific = fb_vap - float(fb_vap @ shared) * shared
fb_specific = fb_specific / np.linalg.norm(fb_specific)


def top_words(direction, k=15):
    """Most-aligned words on the positive pole of direction."""
    return wv.similar_by_vector(direction, topn=k)


def show(name, direction, k=15):
    print(f"\n=== {name} ===")
    print(f"  positive pole (most aligned with direction):")
    for w, s in top_words(direction, k):
        print(f"    {s:>+.4f}  {w}")
    print(f"  negative pole (most aligned with -direction):")
    for w, s in top_words(-direction, k):
        print(f"    {s:>+.4f}  {w}")


show("Raw UD axis (for comparison)", A_ud)
show("Raw FB axis (for comparison)", A_fb)
show("Shared UD-FB residual (after V+A+PATH)", shared)
show("UD-specific residual (UD minus shared)", ud_specific)
show("FB-specific residual (FB minus shared)", fb_specific)

# Also report cosines
print(f"\n=== Cosine sanity checks ===")
print(f"  cos(shared, V)    = {float(shared @ A_val):+.4f}")
print(f"  cos(shared, A)    = {float(shared @ A_aro):+.4f}")
print(f"  cos(shared, PATH) = {float(shared @ A_path):+.4f}")
print(f"  cos(shared, UD)   = {float(shared @ A_ud):+.4f}")
print(f"  cos(shared, FB)   = {float(shared @ A_fb):+.4f}")
print(f"  cos(ud_specific, shared) = {float(ud_specific @ shared):+.4f}  (should be ~0)")
print(f"  cos(fb_specific, shared) = {float(fb_specific @ shared):+.4f}  (should be ~0)")

np.savez(
    "/Users/macn/Documents/embeddingexp/exp30_results.npz",
    shared=shared,
    ud_specific=ud_specific,
    fb_specific=fb_specific,
    ud_vap=ud_vap,
    fb_vap=fb_vap,
)
print(f"\nSaved: exp30_results.npz")
