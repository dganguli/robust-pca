Robust-PCA
==========

A Python implementation of R-PCA using principle component pursuit by alternating directions. The theory and implementation of the algorithm is described here: http://statweb.stanford.edu/~candes/papers/RobustPCA.pdf

```python
# generate low rank synthetic data
N = 100
num_groups = 3
num_values_per_group = 40
p_missing = 0.2

Ds = []
for k in range(num_groups):
    d = np.ones((N, num_values_per_group)) * (k + 1) * 10
    Ds.append(d)

D = np.hstack(Ds)

# decimate 20% of data 
n1, n2 = D.shape
S = np.random.rand(n1, n2)
D[S < 0.2] = 0

# use R_pca to estimate the degraded data as L + S, where L is low rank, and S is sparse
rpca = R_pca(D)
L, S = rpca.fit(max_iter=10000, iter_print=100)
rpca.plot_fit()

# visually inspect results (requires matplotlib)
from pylab import plt

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(D)

plt.subplot(1, 3, 2)
plt.imshow(L)

plt.subplot(1, 3, 3)
plt.imshow(S)
plt.show()

```
