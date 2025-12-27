"""
Image Watermark/Text Removal using Robust PCA

Demonstrates RPCA's ability to separate sparse corruptions (text overlay) from
the underlying low-rank image structure.

    D = L + S

where:
    D = image with text/watermark overlay
    L = recovered clean image (low-rank structure)
    S = extracted text/watermark (sparse component)

This works because:
- Natural images have low-rank structure (smooth regions, repeated patterns)
- Text/watermarks are sparse (only affect a small fraction of pixels)

How to run:
    python -m examples.image_watermark_removal
"""
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from r_pca import RobustPCA


def create_natural_image(height=120, width=160):
    """Create a synthetic image with natural low-rank structure.

    Uses separable components (outer products) to ensure true low-rank structure.
    This simulates how natural images often have smooth horizontal bands.
    """
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)

    # Create image using outer products (guarantees low rank)
    # Each channel is a sum of rank-1 matrices: u * v^T

    # Red channel: warm gradient
    r1 = np.outer(1 - 0.5*y, np.ones(width))  # Vertical gradient
    r2 = np.outer(np.ones(height), 0.2*np.sin(3*np.pi*x))  # Horizontal wave
    red = 0.6 * r1 + 0.15 * r2 + 0.3

    # Green channel: earth tones
    g1 = np.outer(0.8 - 0.6*y, np.ones(width))
    g2 = np.outer(np.exp(-((y-0.7)**2)/0.02), np.ones(width))  # Horizon band
    green = 0.4 * g1 - 0.2 * g2 + 0.2

    # Blue channel: sky gradient
    b1 = np.outer(1 - 0.8*y, np.ones(width))
    b2 = np.outer(np.ones(height), 0.1*np.cos(2*np.pi*x))
    blue = 0.5 * b1 + 0.1 * b2 + 0.2

    image = np.stack([red, green, blue], axis=2)
    return np.clip(image, 0, 1)


def add_text_watermark(image, text="SAMPLE", opacity=0.7):
    """Add text watermark to image.

    Creates a sparse corruption pattern (text pixels only).
    """
    height, width = image.shape[:2]
    watermarked = image.copy()

    # Create text mask using a simple bitmap font approach
    # Each character is represented as a small binary pattern
    char_patterns = {
        'S': [
            [0,1,1,1,0],
            [1,0,0,0,0],
            [0,1,1,0,0],
            [0,0,0,1,0],
            [1,1,1,0,0],
        ],
        'A': [
            [0,1,1,0,0],
            [1,0,0,1,0],
            [1,1,1,1,0],
            [1,0,0,1,0],
            [1,0,0,1,0],
        ],
        'M': [
            [1,0,0,0,1],
            [1,1,0,1,1],
            [1,0,1,0,1],
            [1,0,0,0,1],
            [1,0,0,0,1],
        ],
        'P': [
            [1,1,1,0,0],
            [1,0,0,1,0],
            [1,1,1,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0],
        ],
        'L': [
            [1,0,0,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0],
            [1,1,1,1,0],
        ],
        'E': [
            [1,1,1,1,0],
            [1,0,0,0,0],
            [1,1,1,0,0],
            [1,0,0,0,0],
            [1,1,1,1,0],
        ],
    }

    # Create the text mask
    text_mask = np.zeros((height, width), dtype=bool)

    # Scale and position
    scale = 4
    char_h, char_w = 5 * scale, 5 * scale
    spacing = 6 * scale

    # Center the text
    total_width = len(text) * spacing
    start_x = (width - total_width) // 2
    start_y = (height - char_h) // 2

    for i, char in enumerate(text):
        if char in char_patterns:
            pattern = np.array(char_patterns[char])
            # Scale up the pattern
            scaled = np.kron(pattern, np.ones((scale, scale)))

            x_pos = start_x + i * spacing
            y_pos = start_y

            # Place the character
            for py in range(min(char_h, height - y_pos)):
                for px in range(min(char_w, width - x_pos)):
                    if py < scaled.shape[0] and px < scaled.shape[1]:
                        if scaled[py, px] > 0:
                            text_mask[y_pos + py, x_pos + px] = True

    # Apply watermark (white text with some transparency)
    watermark_color = np.array([1.0, 1.0, 1.0])
    for c in range(3):
        watermarked[:,:,c][text_mask] = (
            (1 - opacity) * watermarked[:,:,c][text_mask] +
            opacity * watermark_color[c]
        )

    return watermarked, text_mask


def main():
    print("=" * 60)
    print("Image Watermark Removal using Robust PCA")
    print("=" * 60)

    # Create a natural-looking image
    height, width = 120, 160
    original = create_natural_image(height, width)
    print(f"Created synthetic image: {height}x{width}")

    # Add text watermark
    watermarked, text_mask = add_text_watermark(original, "SAMPLE", opacity=0.8)
    corruption_pct = 100 * text_mask.sum() / text_mask.size
    print(f"Added 'SAMPLE' watermark ({corruption_pct:.1f}% of pixels)")

    # Process each color channel with RPCA
    # Use smaller lambda to encourage sparser S (text is very sparse)
    print("\nRunning Robust PCA on each color channel...")
    L_channels = []
    S_channels = []

    for c, name in enumerate(['Red', 'Green', 'Blue']):
        D = watermarked[:,:,c]
        # Smaller lambda = sparser S, which helps isolate the text
        lmbda = 0.5 / np.sqrt(max(D.shape))
        rpca = RobustPCA(D, lmbda=lmbda)
        L, S = rpca.fit(max_iter=1000, iter_print=500)
        L_channels.append(np.clip(L, 0, 1))
        S_channels.append(S)
        print(f"  {name} channel done")

    # Reconstruct images
    L_image = np.stack(L_channels, axis=2)
    S_image = np.stack(S_channels, axis=2)

    # Calculate reconstruction error
    reconstruction = L_image + S_image
    reconstruction_error = np.linalg.norm(watermarked - reconstruction)
    relative_error = reconstruction_error / np.linalg.norm(watermarked)

    print(f"\nReconstruction Error ||D - (L+S)||_F: {reconstruction_error:.6f}")
    print(f"Relative Error ||D - (L+S)||_F / ||D||_F: {relative_error:.2e}")

    # Create visualization matching D = L + S format
    fig = plt.figure(figsize=(14, 4))
    gs = fig.add_gridspec(1, 5, width_ratios=[3, 0.5, 3, 0.5, 3], wspace=0.1)

    # D (watermarked image)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(watermarked)
    ax0.set_title('D (Watermarked)', fontsize=14, fontweight='bold')
    ax0.axis('off')

    # "=" symbol
    ax_eq = fig.add_subplot(gs[0, 1])
    ax_eq.text(0.5, 0.5, '=', fontsize=32, fontweight='bold', ha='center', va='center')
    ax_eq.axis('off')

    # L (recovered image)
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.imshow(L_image)
    ax1.set_title('L (Recovered)', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # "+" symbol
    ax_plus = fig.add_subplot(gs[0, 3])
    ax_plus.text(0.5, 0.5, '+', fontsize=32, fontweight='bold', ha='center', va='center')
    ax_plus.axis('off')

    # S (extracted watermark) - grayscale so white watermark appears white like in D
    ax2 = fig.add_subplot(gs[0, 4])
    S_magnitude = np.linalg.norm(S_image, axis=2)  # Combine RGB channels to single magnitude
    ax2.imshow(S_magnitude, cmap='gray', vmin=0)
    ax2.set_title('S (Watermark)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    plt.suptitle('Robust PCA: D = L + S', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'image_watermark_result.png'),
                dpi=150, bbox_inches='tight')
    print("\nSaved to examples/image_watermark_result.png")


if __name__ == "__main__":
    main()
