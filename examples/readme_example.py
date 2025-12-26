"""
README Example - Direct copy from README to verify it works.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Store clean version
D_clean = D.copy()

# decimate 20% of data
n1, n2 = D.shape
S = np.random.rand(n1, n2)
D[S < 0.2] = 0

# use RobustPCA to estimate the degraded data as L + S, where L is low rank, and S is sparse
rpca = RobustPCA(D)
L, S = rpca.fit(max_iter=500, iter_print=50)

# Calculate MSE
mse_corrupted = np.mean((D - D_clean) ** 2)
mse_recovered = np.mean((L - D_clean) ** 2)
print(f"\nMSE (corrupted vs clean): {mse_corrupted:.4f}")
print(f"MSE (recovered vs clean): {mse_recovered:.4f}")
print(f"Improvement: {mse_corrupted/mse_recovered:.1f}x")

# Plot as images
fig, axes = plt.subplots(1, 4, figsize=(14, 4))

axes[0].imshow(D_clean, aspect='auto', cmap='viridis')
axes[0].set_title('Original (Clean)')
axes[0].set_xlabel('Columns')
axes[0].set_ylabel('Rows')

axes[1].imshow(D, aspect='auto', cmap='viridis')
axes[1].set_title('Corrupted (20% zeros)')

axes[2].imshow(L, aspect='auto', cmap='viridis')
axes[2].set_title('Recovered (L)')

axes[3].imshow(np.abs(S), aspect='auto', cmap='hot')
axes[3].set_title('Sparse (S)')

plt.suptitle('Robust PCA: README Example', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'readme_example_result.png'), dpi=150)
print("\nSaved to examples/readme_example_result.png")
