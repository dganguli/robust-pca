"""
Video Background/Foreground Separation using Robust PCA

This is the CLASSIC application of Robust PCA! Given a video of a static scene
with moving objects, RPCA separates:

    D = L + S

where:
    L = low-rank component (the static background - same across all frames)
    S = sparse component (the moving foreground objects)

This works because:
- The background is the same in every frame -> rank 1 matrix
- Moving objects occupy only a small portion of each frame -> sparse

Applications:
- Surveillance: separate people/vehicles from background
- Video compression: encode background once
- Object tracking: isolate moving objects
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from r_pca import RobustPCA


def create_synthetic_video(n_frames=30, height=60, width=80):
    """Create a synthetic video with static background and moving object.

    Returns:
        frames: list of 2D arrays (each frame)
        background: the static background image
    """
    # Create a static background
    # Use separable functions to create a truly low-rank background
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))

    background = (
        0.3 +
        0.15 * np.sin(8 * np.pi * x) +  # Vertical stripes (rank 1)
        0.15 * np.cos(6 * np.pi * y)    # Horizontal variation (rank 1)
    )
    background = np.clip(background, 0, 1)

    frames = []
    foreground_masks = []

    for i in range(n_frames):
        frame = background.copy()

        # Create a moving rectangular object (more visible)
        t = i / n_frames
        # Object moves horizontally across the frame
        obj_width, obj_height = 15, 20
        cx = int(width * (0.1 + 0.8 * t))
        cy = height // 2

        # Draw the object (bright rectangle)
        x1, x2 = max(0, cx - obj_width//2), min(width, cx + obj_width//2)
        y1, y2 = max(0, cy - obj_height//2), min(height, cy + obj_height//2)
        mask = np.zeros((height, width), dtype=bool)
        mask[y1:y2, x1:x2] = True
        frame[mask] = 0.95  # Bright object

        frames.append(frame)
        foreground_masks.append(mask)

    return frames, background, foreground_masks


def frames_to_matrix(frames):
    """Convert list of frames to a matrix (each row is a flattened frame)."""
    return np.vstack([f.flatten() for f in frames])


def main():
    print("=" * 60)
    print("Video Background/Foreground Separation using Robust PCA")
    print("=" * 60)

    # Create synthetic video
    n_frames = 40
    height, width = 60, 80
    frames, true_background, foreground_masks = create_synthetic_video(
        n_frames, height, width
    )
    print(f"Created synthetic video: {n_frames} frames of {height}x{width}")

    # Convert to matrix (rows are frames)
    D = frames_to_matrix(frames)
    print(f"Data matrix shape: {D.shape} (frames x pixels)")

    # Apply Robust PCA
    print("\nRunning Robust PCA...")
    rpca = RobustPCA(D)
    L, S = rpca.fit(max_iter=1000, iter_print=200)

    # Extract background (average of L rows, since all should be the same)
    recovered_background = L.mean(axis=0).reshape(height, width)
    recovered_background = np.clip(recovered_background, 0, 1)

    # Simple average baseline (average over all frames)
    avg_frame = D.mean(axis=0).reshape(height, width)

    # Calculate quality
    mse_avg = np.mean((avg_frame - true_background) ** 2)
    mse_rpca = np.mean((recovered_background - true_background) ** 2)

    print(f"\nBackground Recovery MSE:")
    print(f"  Simple Average: {mse_avg:.6f}")
    print(f"  Robust PCA:     {mse_rpca:.6f}")
    print(f"  Improvement:    {mse_avg/mse_rpca:.1f}x")

    # Visualize results
    fig = plt.figure(figsize=(14, 10))

    # Row 1: Sample frames from the video
    frame_indices = [0, n_frames//3, 2*n_frames//3, n_frames-1]
    for i, idx in enumerate(frame_indices):
        ax = fig.add_subplot(3, 4, i+1)
        ax.imshow(frames[idx], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Frame {idx+1}', fontsize=10)
        ax.axis('off')

    # Row 2: Background comparison
    ax = fig.add_subplot(3, 4, 5)
    ax.imshow(true_background, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('True Background', fontsize=10)
    ax.axis('off')

    ax = fig.add_subplot(3, 4, 6)
    ax.imshow(avg_frame, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f'Average of Frames\n(MSE: {mse_avg:.4f})', fontsize=10)
    ax.axis('off')

    ax = fig.add_subplot(3, 4, 7)
    ax.imshow(recovered_background, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f'RPCA Background (L)\n(MSE: {mse_rpca:.4f})', fontsize=10)
    ax.axis('off')

    ax = fig.add_subplot(3, 4, 8)
    diff = np.abs(recovered_background - true_background)
    ax.imshow(diff, cmap='hot', vmin=0, vmax=0.1)
    ax.set_title('Recovery Error', fontsize=10)
    ax.axis('off')

    # Row 3: Foreground extraction
    for i, idx in enumerate(frame_indices):
        ax = fig.add_subplot(3, 4, 9+i)
        foreground = np.abs(S[idx, :]).reshape(height, width)
        ax.imshow(foreground, cmap='hot')
        ax.set_title(f'Foreground (S) Frame {idx+1}', fontsize=10)
        ax.axis('off')

    plt.suptitle('Robust PCA: Video Background/Foreground Separation\n'
                 'D = L (static background) + S (moving objects)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'video_separation_result.png'),
                dpi=150, bbox_inches='tight')
    print("\nSaved result to examples/video_separation_result.png")


if __name__ == "__main__":
    main()
