"""
exp32: Direct construction of yangyinspace in word2vec.

Build a yang/yin axis from explicit yang/yin anchor pairs (change-of-existence
focus: emerging vs vanishing, born vs died, creation vs destruction, built vs
demolished, etc.) that contain NO UD/FB/LD/IO vocabulary.

Then test three things:
  1. Does the directly-constructed yangyinspace match the inferred one from
     averaging V+A-residualized UD/FB/LD axes? (should: cos ~ +0.6+)
  2. Is yangyinspace empirically distinct from valence and arousal? (should:
     cos with V, A near zero)
  3. THE HEADLINE TEST: cos(A_inout, yangyinspace_direct).
     If traditional Taoism is right (IN = yin), cosine should be NEGATIVE.
     If our previous +0.22 was from valence contamination, cosine should drop
       toward zero with clean direct construction.
     If positive (>+0.2), modern English has reorganized IN as yang (would be
       a surprising cross-cultural finding).

Also: build a "clean IO" from direction-only pairs (no inclusion/exclusion
valence) and compare to the membership-loaded original IO. The cleaner the
IO anchors, the more decisive the IN-as-yang vs IN-as-yin reading.
"""
import numpy as np
import gensim.downloader as api

print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")


# ============================== YANG/YIN ANCHORS ==============================
# Change-of-existence: things coming into being vs going out of being.
# Avoiding all UD/FB/LD/IO vocabulary AND avoiding obvious valence words.
YANGYIN_PAIRS = [
    # Birth/death
    ("born", "died"),
    ("birth", "death"),
    ("birth", "demise"),
    # Creation/destruction
    ("creating", "destroying"),
    ("creation", "destruction"),
    ("create", "destroy"),
    ("created", "destroyed"),
    # Building/demolition
    ("built", "demolished"),
    ("building", "demolishing"),
    ("construct", "demolish"),
    ("constructed", "dismantled"),
    # Emerge/vanish
    ("emerging", "vanishing"),
    ("emerged", "vanished"),
    ("emerge", "vanish"),
    ("appearing", "disappearing"),
    ("appeared", "disappeared"),
    # Arising/dissipating
    ("arising", "dissipating"),
    ("arise", "dissipate"),
    # Assembling/disintegrating
    ("assembled", "disassembled"),
    ("formed", "disintegrated"),
    ("generated", "dissolved"),
    ("materialized", "vanished"),
    # Founding/abandoning
    ("founded", "abandoned"),
    ("establish", "abandon"),
    ("established", "abandoned"),
    # Starting/ending
    ("starting", "ending"),
    ("began", "ended"),
    ("begin", "end"),
    ("started", "ended"),
    ("originated", "terminated"),
    ("commenced", "concluded"),
    # Activation/dormancy
    ("active", "dormant"),
    ("activated", "deactivated"),
]


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
# Original IO (mixed membership + difficulty + direction-only)
IO_PAIRS_MIXED = [
    ("inside", "outside"), ("contained", "released"), ("enclosed", "freed"),
    ("in", "out"), ("entered", "exited"), ("internal", "external"),
    ("remembered", "forgotten"), ("retained", "dismissed"),
    ("married", "divorced"), ("engaged", "estranged"), ("together", "apart"),
    ("trapped", "escaped"), ("stranded", "rescued"), ("captured", "freed"),
    ("included", "excluded"), ("admitted", "expelled"),
]
# Clean IO (direction-only, no inclusion/exclusion or difficulty valence)
IO_PAIRS_CLEAN = [
    ("inside", "outside"),
    ("contained", "released"),
    ("enclosed", "freed"),
    ("in", "out"),
    ("entered", "exited"),
    ("internal", "external"),
    ("interior", "exterior"),
    ("inward", "outward"),
    ("indoors", "outdoors"),
]


def build_axis(pairs, label=""):
    offs = []
    missing = []
    for a, c in pairs:
        if a in wv.key_to_index and c in wv.key_to_index:
            offs.append(wv[a] - wv[c])
        else:
            missing.append((a, c))
    if missing:
        print(f"  [{label}] missing: {missing}")
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw), len(offs)


def residualize(v, axes):
    r = v.copy()
    for a in axes:
        r = r - float(r @ a) * a
    nrm = np.linalg.norm(r)
    return r / nrm if nrm > 1e-12 else r


# Build all axes
print("\n=== Building axes ===")
A_val, _ = build_axis(VALENCE_PAIRS, "VAL")
A_aro, _ = build_axis(AROUSAL_PAIRS, "ARO")
A_yyn, n_yyn = build_axis(YANGYIN_PAIRS, "YANGYIN_DIRECT")
A_ud, _ = build_axis(UD_PAIRS, "UD")
A_fb, _ = build_axis(FB_PAIRS, "FB")
A_ld, _ = build_axis(LD_PAIRS, "LD")
A_io_mixed, _ = build_axis(IO_PAIRS_MIXED, "IO_MIXED")
A_io_clean, _ = build_axis(IO_PAIRS_CLEAN, "IO_CLEAN")
print(f"  YANGYIN_DIRECT built from {n_yyn} pairs")


# Build the inferred yangyin (mean of V+A-residualized UD, FB, LD)
ud_va = residualize(A_ud, [A_val, A_aro])
fb_va = residualize(A_fb, [A_val, A_aro])
ld_va = residualize(A_ld, [A_val, A_aro])
A_yyn_inferred = (ud_va + fb_va + ld_va) / 3
A_yyn_inferred = A_yyn_inferred / np.linalg.norm(A_yyn_inferred)


print("\n=== Sanity checks on YANGYIN_DIRECT ===")
print(f"  cos(YANGYIN_DIRECT, VALENCE)  = {float(A_yyn @ A_val):+.4f}  (should be small if yang/yin is not valence)")
print(f"  cos(YANGYIN_DIRECT, AROUSAL)  = {float(A_yyn @ A_aro):+.4f}  (should be small)")
print(f"  cos(YANGYIN_DIRECT, YANGYIN_INFERRED) = {float(A_yyn @ A_yyn_inferred):+.4f}  (high cosine = methods agree)")


# Residualize YANGYIN_DIRECT with V and A to make it as comparable as possible
A_yyn_va = residualize(A_yyn, [A_val, A_aro])
print(f"  cos(YANGYIN_DIRECT after V+A removed, YANGYIN_INFERRED) = {float(A_yyn_va @ A_yyn_inferred):+.4f}")


print("\n=== YANGYIN_DIRECT alignments with schemas ===")
print(f"  cos(YANGYIN_DIRECT, UD)    = {float(A_yyn @ A_ud):+.4f}")
print(f"  cos(YANGYIN_DIRECT, FB)    = {float(A_yyn @ A_fb):+.4f}")
print(f"  cos(YANGYIN_DIRECT, LD)    = {float(A_yyn @ A_ld):+.4f}")
print(f"  cos(YANGYIN_DIRECT, IO_MIXED) = {float(A_yyn @ A_io_mixed):+.4f}")
print(f"  cos(YANGYIN_DIRECT, IO_CLEAN) = {float(A_yyn @ A_io_clean):+.4f}")


print("\n=== After V+A residualization on YANGYIN_DIRECT ===")
print(f"  cos(YANGYIN_DIRECT_VA, UD_VA) = {float(A_yyn_va @ ud_va):+.4f}")
print(f"  cos(YANGYIN_DIRECT_VA, FB_VA) = {float(A_yyn_va @ fb_va):+.4f}")
print(f"  cos(YANGYIN_DIRECT_VA, LD_VA) = {float(A_yyn_va @ ld_va):+.4f}")

io_mixed_va = residualize(A_io_mixed, [A_val, A_aro])
io_clean_va = residualize(A_io_clean, [A_val, A_aro])
print(f"  cos(YANGYIN_DIRECT_VA, IO_MIXED_VA) = {float(A_yyn_va @ io_mixed_va):+.4f}")
print(f"  cos(YANGYIN_DIRECT_VA, IO_CLEAN_VA) = {float(A_yyn_va @ io_clean_va):+.4f}")


print("\n=== THE HEADLINE TEST: IN/OUT vs yang/yin ===")
print(f"  Traditional Taoism predicts: IN = YIN (cosine should be NEGATIVE)")
print(f"  Using A_io_mixed (with membership/difficulty valence):")
print(f"    cos(IO_MIXED, YANGYIN_DIRECT)       = {float(A_io_mixed @ A_yyn):+.4f}")
print(f"    cos(IO_MIXED, YANGYIN_DIRECT_VA)    = {float(A_io_mixed @ A_yyn_va):+.4f}")
print(f"    cos(IO_MIXED_VA, YANGYIN_DIRECT_VA) = {float(io_mixed_va @ A_yyn_va):+.4f}")
print(f"  Using A_io_clean (direction-only, no membership valence):")
print(f"    cos(IO_CLEAN, YANGYIN_DIRECT)       = {float(A_io_clean @ A_yyn):+.4f}")
print(f"    cos(IO_CLEAN, YANGYIN_DIRECT_VA)    = {float(A_io_clean @ A_yyn_va):+.4f}")
print(f"    cos(IO_CLEAN_VA, YANGYIN_DIRECT_VA) = {float(io_clean_va @ A_yyn_va):+.4f}")


print("\n=== Nearest neighbors of YANGYIN_DIRECT (the directly-constructed axis) ===")
print(f"  Yang pole:")
for w, s in wv.similar_by_vector(A_yyn, topn=20):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  Yin pole:")
for w, s in wv.similar_by_vector(-A_yyn, topn=20):
    print(f"    {s:>+.4f}  {w}")


print("\n=== Nearest neighbors of YANGYIN_DIRECT after V+A removal ===")
print(f"  Yang pole (after V+A removal):")
for w, s in wv.similar_by_vector(A_yyn_va, topn=15):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  Yin pole (after V+A removal):")
for w, s in wv.similar_by_vector(-A_yyn_va, topn=15):
    print(f"    {s:>+.4f}  {w}")


print("\n=== Reading guide ===")
print(f"  Cross-method agreement: high cos(YANGYIN_DIRECT_VA, YANGYIN_INFERRED) means yang/yin recovers from both methods")
print(f"  IN-as-YIN prediction:")
print(f"    cos(IO_CLEAN_VA, YANGYIN_DIRECT_VA) NEGATIVE → traditional Taoism holds, IN is yin")
print(f"    cos near 0 → IO has no yang/yin content")
print(f"    cos positive → IN is yang in modern English (or valence still contaminating)")

np.savez(
    "/Users/macn/Documents/embeddingexp/exp32_results.npz",
    yangyin_direct=A_yyn,
    yangyin_direct_va=A_yyn_va,
    yangyin_inferred=A_yyn_inferred,
    io_mixed=A_io_mixed,
    io_clean=A_io_clean,
)
print(f"\nSaved: exp32_results.npz")
