import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
sys.path.insert(0, '.')
from feature_extractor import extract_24_features_batch

print("Loading training data (sample 50K)...")
df = pd.read_parquet("task_A/train.parquet")
df = df.sample(n=50000, random_state=42).reset_index(drop=True)

print("Extracting 24 features...")
feat_df = extract_24_features_batch(df["code"], show_progress=True)
feat_df["label"] = df["label"].values

FEAT_COLS = [c for c in feat_df.columns if c != "label"]

human = feat_df[feat_df.label == 0]
ai    = feat_df[feat_df.label == 1]

results = []
for f in FEAT_COLS:
    h = human[f].dropna().values
    a = ai[f].dropna().values
    pooled_std = np.sqrt((h.var() + a.var()) / 2)
    if pooled_std < 1e-10:
        continue
    d = abs((a.mean() - h.mean()) / pooled_std)
    results.append((f, d))

results.sort(key=lambda x: x[1], reverse=True)
top10 = results[:10]

print("\nTop 10 features by Cohen's d:")
for name, val in top10:
    print(f"  {name:<35} d={val:.3f}")

names  = [r[0] for r in top10]
values = [r[1] for r in top10]

colors = []
for v in values:
    if v > 0.8:
        colors.append("#d62728")
    elif v > 0.5:
        colors.append("#ff7f0e")
    else:
        colors.append("#2ca02c")

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(names[::-1], values[::-1], color=colors[::-1])
ax.set_xlabel("|Cohen's d|")
ax.set_title("Feature Discriminative Power: Human vs AI Code")

red_p    = mpatches.Patch(color="#d62728", label="Large effect (d > 0.8)")
orange_p = mpatches.Patch(color="#ff7f0e", label="Medium effect (d > 0.5)")
green_p  = mpatches.Patch(color="#2ca02c", label="Moderate (d < 0.5)")
ax.legend(handles=[red_p, orange_p, green_p], loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig("cohens_d_features_new.png", dpi=150, bbox_inches="tight")
plt.savefig("cohens_d_features_new.pdf", bbox_inches="tight")
print("\nSaved cohens_d_features_new.png / .pdf")
