import matplotlib.pyplot as plt
import numpy as np

features = [
    'indent_consistency', 'max_indent', 'space_ratio', 'empty_line_ratio',
    'avg_identifier_len', 'leading_tab_lines', 'unique_char_ratio',
    'compression_ratio', 'shannon_entropy', 'comment_ratio'
]
cohens_d = [1.095, 0.981, 0.925, 0.849, 0.799, 0.634, 0.617, 0.534, 0.434, 0.453]

fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#d62728' if d > 0.8 else '#ff7f0e' if d > 0.5 else '#2ca02c' for d in cohens_d]
ax.barh(features[::-1], cohens_d[::-1], color=colors[::-1], edgecolor='black')
ax.set_xlabel("|Cohen's d|", fontsize=12)
ax.set_title("Feature Discriminative Power: Human vs AI Code", fontsize=13)
ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect (d>0.8)')
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect (d>0.5)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('cohens_d_features.pdf', bbox_inches='tight')
plt.savefig('cohens_d_features.png', dpi=300, bbox_inches='tight')
print("Saved: cohens_d_features.pdf and cohens_d_features.png")
