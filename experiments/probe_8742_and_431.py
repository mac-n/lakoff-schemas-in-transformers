"""
Two specific lookups:
1. What are the SOURCE features that point at feat 8742 ("duality/comparison")?
   If they're schematically diverse but related-in-shape (all comparison-makers,
   all contrast-words from different domains), that's schema-shaped.
2. What are cluster 431's diverse TARGET features (13957, 4071, 10981, 4530)?
   If they're all register features, hub-and-spoke wins. If they're relational/
   structural, we have schema candidates.
"""
import json
import re
import time

import numpy as np
import requests
from collections import Counter

NEURONPEDIA_MODEL = "pythia-70m-deduped"
NEURONPEDIA_SAE = "5-res-sm"
API_DELAY = 0.3

data = np.load("/Users/macn/Documents/embeddingexp/schema_operators_70m_results.npz")
diff_labels = data["diff_labels"]
pair_i = data["pair_i"]
pair_j = data["pair_j"]


def fetch_feature(feat_idx):
    url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL}/{NEURONPEDIA_SAE}/{feat_idx}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except json.JSONDecodeError:
            clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', r.text)
            return json.loads(clean, strict=False)
    except Exception as e:
        return {"_error": str(e)}


def feature_summary(d):
    if d is None:
        return "[no data]"
    if "_error" in d:
        return f"[error: {d['_error']}]"
    expls = d.get("explanations") or []
    if expls:
        return expls[0].get("description", "").strip()[:200]
    return "(no auto-interp description)"


def lookup_batch(feats, label):
    print(f"\n=== {label} ===")
    for f in feats:
        d = fetch_feature(int(f))
        desc = feature_summary(d)
        print(f"  feat {int(f):>6d}: {desc}")
        time.sleep(API_DELAY)


# --- Cluster 552: feat 8742 is top target ---
m_552 = np.where(diff_labels == 552)[0]
sources_552 = pair_i[m_552]
targets_552 = pair_j[m_552]

# Get the sources that specifically point at 8742
sources_to_8742 = sources_552[targets_552 == 8742]
print(f"Cluster 552 has {len(m_552)} edges; {len(sources_to_8742)} of them point at feat 8742.")
print(f"Looking up the first 12 distinct source features that point at 8742...")

distinct_sources = list(dict.fromkeys(sources_to_8742.tolist()))[:12]
lookup_batch(distinct_sources, f"Sources pointing at feat 8742 ('duality/comparison')")

# Also look at OTHER targets in cluster 552 (the small tail)
other_targets = [int(t) for t in targets_552 if t != 8742]
target_counts = Counter(other_targets)
print(f"\nCluster 552 has {len(set(other_targets))} other distinct targets beyond 8742.")
print(f"Top 5 other targets:")
for t, n in target_counts.most_common(5):
    print(f"  feat {t}: {n} edges")


# --- Cluster 431: diverse targets ---
m_431 = np.where(diff_labels == 431)[0]
sources_431 = pair_i[m_431]
targets_431 = pair_j[m_431]
target_counts_431 = Counter(targets_431.tolist())

print(f"\n\nCluster 431 has {len(m_431)} edges across {len(set(targets_431.tolist()))} distinct targets.")
print(f"All targets and their edge-counts:")
for t, n in target_counts_431.most_common():
    print(f"  feat {t}: {n} edges")

# Look up all of cluster 431's targets
all_431_targets = [t for t, _ in target_counts_431.most_common()]
lookup_batch(all_431_targets, f"All targets in cluster 431 (the most mixed cluster)")

# Also look at the sources in cluster 431 — are they themselves diverse?
print(f"\nCluster 431 has {len(set(sources_431.tolist()))} distinct sources.")
print(f"Sampling 8 sources to look up:")
sample_sources_431 = list(set(sources_431.tolist()))[:8]
lookup_batch(sample_sources_431, f"Sources in cluster 431")
