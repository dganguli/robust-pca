Robust-PCA
==========

A Python implementation of Robust PCA using Principal Component Pursuit by alternating directions. The theory and implementation of the algorithm is described here: https://arxiv.org/pdf/0912.3599.pdf (doi: 10.1145/1970392.1970395)

## Usage

```python
from r_pca import RobustPCA

# generate low rank synthetic data
N = 100
num_groups = 3
num_values_per_group = 40

Ds = []
for k in range(num_groups):
    d = np.ones((N, num_values_per_group)) * (k + 1) * 10
    Ds.append(d)

D = np.hstack(Ds)

# decimate 20% of data
n1, n2 = D.shape
S = np.random.rand(n1, n2)
D[S < 0.2] = 0

# use RobustPCA to estimate the degraded data as L + S, where L is low rank, and S is sparse
rpca = RobustPCA(D)
L, S = rpca.fit(max_iter=10000, iter_print=100)

# visually inspect results (requires matplotlib)
import matplotlib.pyplot as plt
rpca.plot_fit()
plt.show()
```

## Running Tests

```bash
pip install pytest
python -m pytest test_r_pca.py -v
```
