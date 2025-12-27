"""
README Example - Basic recovery of corrupted low-rank data.

How to run:
    python -m examples.readme_example
"""
import os

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from r_pca import RobustPCA, np

# generate low rank synthetic data
N = 100
num_groups = 3
num_values_per_group = 40

Ds = []
for k in range(num_groups):
    d = np.ones((N, num_values_per_group)) * (k + 1) * 10
    Ds.append(d)

D = np.hstack(Ds)

# decimate 5% of data
n1, n2 = D.shape
S = np.random.rand(n1, n2)
D[S < 0.05] = 0

# use RobustPCA to estimate the degraded data as L + S, where L is low rank, and S is sparse
rpca = RobustPCA(D)
L, S = rpca.fit(max_iter=500, iter_print=50)

# Calculate reconstruction error
reconstruction = L + S
reconstruction_error = np.linalg.norm(D - reconstruction, 'fro')
relative_error = reconstruction_error / np.linalg.norm(D, 'fro')

print(f"\nReconstruction Error ||D - (L+S)||_F: {reconstruction_error:.6f}")
print(f"Relative Error ||D - (L+S)||_F / ||D||_F: {relative_error:.2e}")

# Plot as images: D = L + S (matching README cartoon)
fig = plt.figure(figsize=(14, 4))

# Create grid: 3 images + 2 equation symbols
gs = fig.add_gridspec(1, 5, width_ratios=[3, 0.5, 3, 0.5, 3], wspace=0.1)

# Use inferno colormap where black = 0
# D and L share scale [0, 30], S uses [-30, 0] so negative corrections appear dark
cmap = 'inferno'
data_vmin, data_vmax = 0, 30

# D (corrupted input) - black spots are the zeros (corruptions)
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(D, aspect='auto', cmap=cmap, vmin=data_vmin, vmax=data_vmax)
ax0.set_title('D (Corrupted)', fontsize=14, fontweight='bold')
ax0.set_xlabel('Columns')
ax0.set_ylabel('Rows')

# "=" symbol
ax_eq = fig.add_subplot(gs[0, 1])
ax_eq.text(0.5, 0.5, '=', fontsize=32, fontweight='bold', ha='center', va='center')
ax_eq.axis('off')

# L (learned low-rank) - clean recovery, no black
ax1 = fig.add_subplot(gs[0, 2])
ax1.imshow(L, aspect='auto', cmap=cmap, vmin=data_vmin, vmax=data_vmax)
ax1.set_title('L (Low-rank)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Columns')

# "+" symbol
ax_plus = fig.add_subplot(gs[0, 3])
ax_plus.text(0.5, 0.5, '+', fontsize=32, fontweight='bold', ha='center', va='center')
ax_plus.axis('off')

# S (learned sparse) - negative values where corruption was, flip scale so negatives are dark
ax2 = fig.add_subplot(gs[0, 4])
ax2.imshow(-S, aspect='auto', cmap=cmap, vmin=data_vmin, vmax=data_vmax)
ax2.set_title('S (Sparse)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Columns')

plt.suptitle('Robust PCA: D = L + S', fontsize=16, fontweight='bold', y=1.02)
plt.savefig(os.path.join(os.path.dirname(__file__), 'readme_example_result.png'), dpi=150, bbox_inches='tight')
print("\nSaved to examples/readme_example_result.png")
