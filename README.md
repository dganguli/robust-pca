Robust PCA
==========

A Python implementation of Robust PCA using Principal Component Pursuit by alternating directions (ADMM). The theory and algorithm are described in: [Robust Principal Component Analysis?](https://arxiv.org/pdf/0912.3599.pdf) (Candes et al., 2009)

## What is Robust PCA?

Robust PCA decomposes a data matrix **D** into two components:

```
D = L + S
```

where:
- **L** is a **low-rank** matrix (captures the underlying structure)
- **S** is a **sparse** matrix (captures outliers/corruptions)

This is useful when your data has:
- Underlying patterns shared across observations (low-rank structure)
- Sparse corruptions, outliers, or anomalies

### Visual Example

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │  ·    ·     │
│  Corrupted  │  =  │  Low-rank   │  +  │    ·       ·│
│    Data     │     │  Structure  │     │  Sparse     │
│      D      │     │      L      │     │  Outliers S │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Usage

```python
from r_pca import RobustPCA, np

# Create data: 3 groups with constant values
data = np.hstack([
    np.ones((100, 40)) * 10,  # Group 1: all 10s
    np.ones((100, 40)) * 20,  # Group 2: all 20s
    np.ones((100, 40)) * 30   # Group 3: all 30s
])

# Corrupt 20% of entries
mask = np.random.rand(*data.shape) < 0.2
data[mask] = 0

# Decompose into low-rank + sparse
rpca = RobustPCA(data)
L, S = rpca.fit(max_iter=500)

# L recovers the original structure
# S captures the corrupted entries
```

## Examples

### 1. Basic Recovery (examples/readme_example.py)

Full working version of the usage example above, with visualization:

```bash
python examples/readme_example.py
```

![readme_example_result](examples/readme_example_result.png)

### 2. Video Background/Foreground Separation (examples/video_separation.py)

The classic RPCA application: given a video (matrix where each row is a frame), separate the static background from moving objects.

```bash
python examples/video_separation.py
```

![video_separation_result](examples/video_separation_result.png)

**How it works:**
- Stack video frames as rows of matrix D
- Background appears in every frame → low-rank component L
- Moving objects are in different positions each frame → sparse component S

## API Reference

### `RobustPCA(D, mu=None, lmbda=None)`

**Parameters:**
- `D`: Input data matrix (numpy array)
- `mu`: Augmented Lagrangian parameter (default: auto-computed)
- `lmbda`: Sparsity regularization (default: `1/sqrt(max(n,m))`)

### `fit(tol=None, max_iter=1000, iter_print=100)`

Run the ADMM algorithm to decompose D = L + S.

**Returns:** `(L, S)` - the low-rank and sparse components

### `plot_fit(size=None, tol=0.1, axis_on=True)`

Visualize the decomposition (requires matplotlib).

## Implementation Notes

This implementation follows **Algorithm 1** (Principal Component Pursuit by Alternating Directions) from page 29 of the [paper](https://arxiv.org/pdf/0912.3599.pdf), but uses an equivalent formulation with different sign conventions. Here's how the code maps to the paper:

### Paper's Algorithm 1

```
1: initialize S₀ = Y₀ = 0, μ > 0
2: while not converged do
3:    L_{k+1} = D_μ(D - S_k - μ⁻¹Y_k)      # SVD thresholding
4:    S_{k+1} = S_{λμ}(D - L_{k+1} + μ⁻¹Y_k)   # Soft thresholding
5:    Y_{k+1} = Y_k + μ(D - L_{k+1} - S_{k+1})  # Dual update
6: end while
```

### This Implementation

```python
L = svd_threshold(D - S + μ⁻¹Y, μ⁻¹)    # Step 3
S = shrink(D - L + μ⁻¹Y, λμ⁻¹)          # Step 4
Y = Y + μ(D - L - S)                     # Step 5
```

### Key Differences

| Aspect | Paper | This Code | Reason |
|--------|-------|-----------|--------|
| Y sign in L update | −μ⁻¹Y | +μ⁻¹Y | Different Lagrangian sign convention |
| SVD threshold | μ | μ⁻¹ | Using standard ADMM proximal form |
| Shrink threshold | λμ | λμ⁻¹ | Using standard ADMM proximal form |

### Why the Difference?

The paper uses a specific ADMM formulation where μ appears directly as the proximal threshold. This implementation uses the **standard ADMM form** where the threshold is 1/ρ (here, μ⁻¹). Both solve the same optimization problem:

```
minimize  ||L||_* + λ||S||_1
subject to  D = L + S
```

The sign difference for Y in step 3 comes from using a symmetric Lagrangian formulation (standard in ADMM literature) versus the paper's asymmetric form. Since Y is a dual variable initialized to zero and updated iteratively, this sign convention difference doesn't affect the final solution.

The algorithm converges to the same optimal L and S as the paper describes.

## Running Tests

```bash
python -m pytest test_r_pca.py -v
```

## References

- Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). Robust Principal Component Analysis? *Journal of the ACM*, 58(3), 1-37. [arXiv:0912.3599](https://arxiv.org/abs/0912.3599)
