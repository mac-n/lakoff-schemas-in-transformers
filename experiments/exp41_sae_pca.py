"""
exp41: Port the exp40 PCA analysis to SAE substrate.

In word2vec (exp40) the cluster decomposed into 3 cognitive primitives (PC1-3:
salience, motion, equilibrium-runaway) + 3 register axes (PC4-6). Does the same
basis recover in Pythia 70m SAE space, or does computation reorganize it?

Approach for this first pass: use the SAE-encoded schema offsets we already
have from exp26 (UD, IO, FB, LD, LR at all 6 res-sm layers). For each layer:
  1. Build A_schema = mean(pole_a_offset - pole_c_offset) across triples
  2. Stack the 5 schema axes
  3. PCA
  4. Report PC structure per layer

Note: this is a smaller PCA than word2vec (only 5 axes vs 11), so we expect
fewer interpretable PCs. We'd need VALENCE / AROUSAL / EXISTENCE / COHERENCE
SAE-encoded sentence triples to do the full replication. For now, this tests
whether the SAE schemas alone have orthogonal structure that maps onto the
word2vec PCs.

Substrate: Pythia 70m residual-stream SAEs via SAE Lens (same as exp22-26).
"""
import numpy as np
import torch
from sklearn.decomposition import PCA


# Load exp26 cached results — schema offset data at all 6 layers
print("Loading exp26 results...")
data = torch.load("/Users/macn/Documents/embeddingexp/exp26_results.pt", weights_only=False)
per_layer = data["per_layer"]
n_layers = len(per_layer)
print(f"Loaded {n_layers} layers from exp26")


# We'll need the raw offset arrays — exp26 stored within_pair_cos and projections,
# but we need to reconstruct the schema axes from the offset arrays.
# Let me check what we have.
print(f"\nKeys per layer:", list(per_layer[0].keys()))


# We don't have the raw offsets cached. We need to rebuild axes by re-running
# encoding (slow) OR we can use a workaround: the projections themselves tell
# us the relative geometry of the offsets, but not enough to do PCA.
#
# Pragmatic move: re-encode the schemas to get raw offset arrays, then PCA per layer.
print("\nNo raw offsets cached — re-encoding schemas for PCA. ~5-10 min total.")

import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
# Import the schema definitions from exp26
from exp26_phase_a_measurement_gaps import (
    SCHEMAS, model, device, collect
)
from sae_lens import SAE


def enc_via_sae(text, hook, sae):
    with torch.no_grad():
        return sae.encode(collect(text, hook)).max(0).values.numpy().astype(np.float64)


per_layer_pca = {}

for layer in range(n_layers):
    print(f"\n=== Layer {layer} ===")
    hook = f"blocks.{layer}.hook_resid_post"
    sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=hook, device="cpu")
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res

    # Build the 5 schema axes from triples
    schema_axes = {}
    for schema, (triples, name_a, name_c) in SCHEMAS.items():
        offsets_diff = []
        for (dom, b, a, c) in triples:
            b_v = enc_via_sae(b, hook, sae)
            a_v = enc_via_sae(a, hook, sae)
            c_v = enc_via_sae(c, hook, sae)
            off_a = a_v - b_v
            off_c = c_v - b_v
            offsets_diff.append(off_a - off_c)
        axis_raw = np.stack(offsets_diff).mean(axis=0)
        nrm = np.linalg.norm(axis_raw)
        if nrm > 1e-12:
            schema_axes[schema] = axis_raw / nrm
        else:
            schema_axes[schema] = axis_raw
        print(f"  {schema}: built")

    # Stack and PCA
    names = list(schema_axes.keys())
    M = np.stack([schema_axes[n] for n in names])

    pca = PCA(n_components=min(len(names), 5))
    pca.fit(M)

    print(f"  Explained variance per PC:")
    for i, ev in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {ev*100:>5.1f}%")

    print(f"  PC loadings (input axis loadings):")
    for pc_idx in range(min(3, len(pca.components_))):
        pc = pca.components_[pc_idx]
        pc_unit = pc / np.linalg.norm(pc)
        loadings = [(name, float(schema_axes[name] @ pc_unit)) for name in names]
        loadings.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"    PC{pc_idx+1}: " + ", ".join(f"{name}{load:+.2f}" for name, load in loadings))

    per_layer_pca[layer] = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_,
        "axis_names": names,
        "schema_axes": schema_axes,
    }
    del sae


print("\n\n" + "="*72)
print("SUMMARY: PC1 loadings per layer (does the same structure emerge across layers?)")
print("="*72)
print(f"  {'layer':>5}  " + "  ".join(f"{n:>10}" for n in SCHEMAS))
for layer in range(n_layers):
    pca_result = per_layer_pca[layer]
    pc1 = pca_result["components"][0]
    pc1_unit = pc1 / np.linalg.norm(pc1)
    loadings = [float(pca_result["schema_axes"][n] @ pc1_unit) for n in SCHEMAS]
    print(f"  {layer:>5}  " + "  ".join(f"{l:>+10.3f}" for l in loadings))

print(f"\n  {'PC2 loadings:'}")
print(f"  {'layer':>5}  " + "  ".join(f"{n:>10}" for n in SCHEMAS))
for layer in range(n_layers):
    pca_result = per_layer_pca[layer]
    pc2 = pca_result["components"][1]
    pc2_unit = pc2 / np.linalg.norm(pc2)
    loadings = [float(pca_result["schema_axes"][n] @ pc2_unit) for n in SCHEMAS]
    print(f"  {layer:>5}  " + "  ".join(f"{l:>+10.3f}" for l in loadings))


# Save
torch.save({"per_layer_pca": per_layer_pca}, "/Users/macn/Documents/embeddingexp/exp41_results.pt")
print("\nSaved: exp41_results.pt")
