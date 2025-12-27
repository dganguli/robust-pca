"""
COIL-20 Object Recovery using Robust PCA

Demonstrates RPCA on real photographs from the Columbia Object Image Library.
Multiple views of the same object (rotating on a turntable) create natural
low-rank structure.

    D = L + S

where:
    D = corrupted multi-view images (stacked as rows)
    L = recovered clean images (low-rank: same object, different angles)
    S = sparse corruption pattern

How to run:
    python -m examples.coil20_recovery
"""
import os
import io
import re
import urllib.request
import zipfile

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from r_pca import RobustPCA


COIL20_URL = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"


def fetch_coil20_object(obj_id=1, size=64):
    """Fetch COIL-20 images for a single object directly into memory.

    Args:
        obj_id: Object ID (1-20). Object 1 is a duck, 3 is a car, etc.
        size: Resize images to size x size (default 64 for speed)

    Returns:
        images: List of numpy arrays (grayscale images)
        angles: List of rotation angles
    """
    print(f"Fetching COIL-20 object {obj_id} from Columbia University...")

    with urllib.request.urlopen(COIL20_URL) as response:
        zip_data = io.BytesIO(response.read())

    images = []
    angles = []

    # Pattern: obj{id}__{angle}.png
    pattern = re.compile(rf"obj{obj_id}__(\d+)\.png$")

    with zipfile.ZipFile(zip_data) as zf:
        for name in sorted(zf.namelist()):
            match = pattern.search(name)
            if match:
                angle = int(match.group(1))
                img_data = zf.read(name)
                img = Image.open(io.BytesIO(img_data)).convert('L')  # Grayscale
                img = img.resize((size, size), Image.LANCZOS)  # Downsample
                images.append(np.array(img, dtype=np.float64) / 255.0)
                angles.append(angle)

    print(f"  Loaded {len(images)} views at {size}x{size}")
    return images, angles


def main():
    print("=" * 60)
    print("COIL-20 Object Recovery using Robust PCA")
    print("=" * 60)

    # Fetch images for object 1 (duck)
    images, angles = fetch_coil20_object(obj_id=1)

    height, width = images[0].shape
    n_views = len(images)
    print(f"Image size: {height}x{width}, {n_views} views")

    # Stack images as rows of matrix D
    D_clean = np.vstack([img.flatten() for img in images])
    print(f"Data matrix shape: {D_clean.shape} (views x pixels)")

    # Add sparse corruption (5% of pixels set to 0)
    np.random.seed(42)
    D = D_clean.copy()
    corruption_mask = np.random.rand(*D.shape) < 0.05
    D[corruption_mask] = 0
    corruption_pct = 100 * corruption_mask.sum() / corruption_mask.size
    print(f"Added sparse corruption: {corruption_pct:.1f}% of pixels set to 0")

    # Run RPCA
    print("\nRunning Robust PCA...")
    rpca = RobustPCA(D)
    L, S = rpca.fit(max_iter=1000, iter_print=200)

    # Calculate reconstruction error
    reconstruction_error = np.linalg.norm(D - (L + S), 'fro')
    relative_error = reconstruction_error / np.linalg.norm(D, 'fro')
    print(f"\nReconstruction Error ||D - (L+S)||_F: {reconstruction_error:.6f}")
    print(f"Relative Error: {relative_error:.2e}")

    # Recovery quality (compare L to original clean data)
    recovery_error = np.linalg.norm(L - D_clean, 'fro') / np.linalg.norm(D_clean, 'fro')
    print(f"Recovery Error ||L - D_clean||_F / ||D_clean||_F: {recovery_error:.4f}")

    # Visualize: show a few views
    view_indices = [0, 18, 36, 54]  # 0°, 90°, 180°, 270°

    fig = plt.figure(figsize=(14, 10))

    # Use inferno colormap with 0=black for consistency
    cmap = 'inferno'
    vmin, vmax = 0, 1

    for i, idx in enumerate(view_indices):
        # D (corrupted)
        ax = fig.add_subplot(4, 4, i + 1)
        ax.imshow(D[idx].reshape(height, width), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'D ({angles[idx]}°)', fontsize=10)
        ax.axis('off')

        # L (recovered)
        ax = fig.add_subplot(4, 4, i + 5)
        ax.imshow(L[idx].reshape(height, width), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'L ({angles[idx]}°)', fontsize=10)
        ax.axis('off')

        # S (sparse) - show -S so corruption appears bright
        ax = fig.add_subplot(4, 4, i + 9)
        ax.imshow(-S[idx].reshape(height, width), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'S ({angles[idx]}°)', fontsize=10)
        ax.axis('off')

        # Original (for reference)
        ax = fig.add_subplot(4, 4, i + 13)
        ax.imshow(D_clean[idx].reshape(height, width), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'Original ({angles[idx]}°)', fontsize=10)
        ax.axis('off')

    # Add row labels
    fig.text(0.02, 0.78, 'D (Corrupted)', fontsize=12, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.55, 'L (Recovered)', fontsize=12, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.33, 'S (Sparse)', fontsize=12, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.12, 'Original', fontsize=12, fontweight='bold', rotation=90, va='center')

    plt.suptitle('Robust PCA on COIL-20: D = L + S\n(72 views of rotating object)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    plt.savefig(os.path.join(os.path.dirname(__file__), 'coil20_recovery_result.png'),
                dpi=150, bbox_inches='tight')
    print("\nSaved to examples/coil20_recovery_result.png")


if __name__ == "__main__":
    main()
