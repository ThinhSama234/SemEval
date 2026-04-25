import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
ppl = pd.read_parquet('train_perplexity.parquet')

human = ppl[ppl['label'] == 0]['overall_ppl'].clip(upper=5)
ai = ppl[ppl['label'] == 1]['overall_ppl'].clip(upper=5)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# ── Left: Overlapping density histogram ──
bins = np.linspace(0, 5, 120)
axes[0].hist(human, bins=bins, alpha=0.6, density=True, color='#2ca02c', label='Human', edgecolor='none')
axes[0].hist(ai, bins=bins, alpha=0.6, density=True, color='#d62728', label='AI-Generated', edgecolor='none')
axes[0].axvline(human.median(), color='#2ca02c', linestyle='--', linewidth=1.5, label=f'Human median = {human.median():.2f}')
axes[0].axvline(ai.median(), color='#d62728', linestyle='--', linewidth=1.5, label=f'AI median = {ai.median():.2f}')
axes[0].set_xlabel('Perplexity (clipped at 5)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('Perplexity Distribution: Human vs AI', fontsize=12)
axes[0].legend(fontsize=9)

# ── Right: Box plot per language ──
data_list = []
labels_list = []
colors_box = []
for lang in ['Python', 'Java', 'C++']:
    for lbl, name, c in [(0, 'Human', '#2ca02c'), (1, 'AI', '#d62728')]:
        subset = ppl[(ppl['label'] == lbl) & (ppl['language'] == lang)]['overall_ppl'].clip(upper=5)
        if len(subset) > 0:
            data_list.append(subset.values)
            labels_list.append(f'{lang}\n{name}')
            colors_box.append(c)

bp = axes[1].boxplot(data_list, tick_labels=labels_list, patch_artist=True, widths=0.6,
                     showfliers=False, medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_ylabel('Perplexity', fontsize=11)
axes[1].set_title('Perplexity by Language and Label', fontsize=12)

# Custom legend
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(facecolor='#2ca02c', alpha=0.7, label='Human'),
                         Patch(facecolor='#d62728', alpha=0.7, label='AI-Generated')],
               fontsize=9)

plt.tight_layout()
plt.savefig('perplexity_distribution.pdf', bbox_inches='tight')
plt.savefig('perplexity_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: perplexity_distribution.pdf and perplexity_distribution.png")
