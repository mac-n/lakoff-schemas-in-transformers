"""
exp31: yang/yin test — is LD_residual (after V+A removal) the same axis as the
UD-FB residual?

If yes, yang/yin is the underlying primitive across UD, LD, FB at the
linguistic-distributional level. The Lakoff spatial schemas are then
vocabulary-specific expressions of the same deeper trajectory-shape.

Tests:
  1. Build LD axis, residualize with V + A (matching exp28 methodology)
  2. Compare LD_VA with UD_VA and FB_VA — cosines should be high if shared
  3. Compare LD_VA with shared UD-FB residual — should be high if same axis
  4. Build a "yang/yin candidate" as the mean of all three V+A-residualized
     axes, look up its nearest words. If it reads cleanly as yang/yin,
     we have a single direction shared across all three schemas.
  5. Also show LD_VA's own nearest neighbors to verify yang/yin shape.
"""
import numpy as np
import gensim.downloader as api

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
LD_PAIRS = [
    ("bright", "dark"), ("light", "dark"), ("illuminated", "shadowed"),
    ("radiant", "dim"), ("sunny", "gloomy"), ("shining", "dim"),
    ("luminous", "shadowy"), ("glowing", "darkened"),
    ("clear", "obscure"), ("lucid", "murky"), ("transparent", "opaque"),
    ("obvious", "hidden"), ("hopeful", "hopeless"), ("optimistic", "pessimistic"),
    ("good", "evil"), ("pure", "tainted"), ("noble", "wicked"),
    ("virtuous", "sinful"), ("righteous", "corrupt"),
    ("enlightened", "ignorant"), ("informed", "uninformed"),
    ("known", "unknown"), ("revealed", "concealed"),
]
IO_PAIRS = [
    ("inside", "outside"), ("contained", "released"), ("enclosed", "freed"),
    ("in", "out"), ("entered", "exited"), ("internal", "external"),
    ("remembered", "forgotten"), ("retained", "dismissed"),
    ("married", "divorced"), ("engaged", "estranged"), ("together", "apart"),
    ("trapped", "escaped"), ("stranded", "rescued"), ("captured", "freed"),
    ("included", "excluded"), ("admitted", "expelled"),
]


def build_axis(pairs):
    offs = [wv[a] - wv[c] for a, c in pairs if a in wv.key_to_index and c in wv.key_to_index]
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
A_ud = build_axis(UD_PAIRS)
A_fb = build_axis(FB_PAIRS)
A_ld = build_axis(LD_PAIRS)
A_io = build_axis(IO_PAIRS)

ud_va = residualize(A_ud, [A_val, A_aro])
fb_va = residualize(A_fb, [A_val, A_aro])
ld_va = residualize(A_ld, [A_val, A_aro])
io_va = residualize(A_io, [A_val, A_aro])

shared_udfb = (ud_va + fb_va) / 2
shared_udfb = shared_udfb / np.linalg.norm(shared_udfb)


print("\n=== Decomposition check: how much V/A is in each schema? ===")
print(f"  {'schema':>14}  {'cos(V)':>8}  {'cos(A)':>8}")
print(f"  {'UP-DOWN':>14}  {float(A_ud @ A_val):>+8.4f}  {float(A_ud @ A_aro):>+8.4f}")
print(f"  {'FORWARD-BACK':>14}  {float(A_fb @ A_val):>+8.4f}  {float(A_fb @ A_aro):>+8.4f}")
print(f"  {'LIGHT-DARK':>14}  {float(A_ld @ A_val):>+8.4f}  {float(A_ld @ A_aro):>+8.4f}")
print(f"  {'IN-OUT':>14}  {float(A_io @ A_val):>+8.4f}  {float(A_io @ A_aro):>+8.4f}")


print("\n=== Pairwise cosines between V+A-residualized schema axes ===")
axes_va = {"UD": ud_va, "FB": fb_va, "LD": ld_va, "IO": io_va}
for n1, v1 in axes_va.items():
    for n2, v2 in axes_va.items():
        if n1 < n2:
            print(f"  cos({n1}_VA, {n2}_VA) = {float(v1 @ v2):+.4f}")


print("\n=== LD residual vs shared UD-FB direction ===")
print(f"  cos(LD_VA, shared_UD_FB) = {float(ld_va @ shared_udfb):+.4f}")
print(f"  (high cosine = yang/yin is a single shared axis across UD/FB/LD)")


print("\n=== IO residual vs shared UD-FB direction (sanity — IO should NOT be yang/yin) ===")
print(f"  cos(IO_VA, shared_UD_FB) = {float(io_va @ shared_udfb):+.4f}")


# Build yang/yin candidate as the mean of UD, FB, LD V+A residuals
yangyin = (ud_va + fb_va + ld_va) / 3
yangyin = yangyin / np.linalg.norm(yangyin)


print("\n=== Yang/yin candidate axis (mean of UD_VA + FB_VA + LD_VA) ===")
print(f"  positive pole (yang):")
for w, s in wv.similar_by_vector(yangyin, topn=20):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  negative pole (yin):")
for w, s in wv.similar_by_vector(-yangyin, topn=20):
    print(f"    {s:>+.4f}  {w}")


print("\n=== LD_VA nearest neighbors (for comparison) ===")
print(f"  positive pole:")
for w, s in wv.similar_by_vector(ld_va, topn=15):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  negative pole:")
for w, s in wv.similar_by_vector(-ld_va, topn=15):
    print(f"    {s:>+.4f}  {w}")


print("\n=== Cosines: yang/yin candidate vs each schema residual ===")
for n, v in axes_va.items():
    print(f"  cos(yangyin, {n}_VA) = {float(yangyin @ v):+.4f}")
print(f"  cos(yangyin, V)    = {float(yangyin @ A_val):+.4f}")
print(f"  cos(yangyin, A)    = {float(yangyin @ A_aro):+.4f}")


print("\n=== Reading guide ===")
print(f"  If cos(LD_VA, shared_UDFB) > +0.4 → yang/yin is the shared underlying axis")
print(f"  If yangyin pole nearest neighbors are clearly emergence vs collapse → confirmed")
print(f"  If IO_VA cosine with yangyin is small (~0) → IO is genuinely separate (good)")

np.savez(
    "/Users/macn/Documents/embeddingexp/exp31_results.npz",
    yangyin=yangyin,
    ud_va=ud_va, fb_va=fb_va, ld_va=ld_va, io_va=io_va,
    shared_udfb=shared_udfb,
)
print(f"\nSaved: exp31_results.npz")
