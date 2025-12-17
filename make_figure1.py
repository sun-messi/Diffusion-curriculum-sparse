"""
Generate Figure 1: Coordinated Training Timeline - Curriculum Learning × Sparsity
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
import os

# ============== Configuration ==============
STAGES = ['EARLY STAGE', 'MIDDLE STAGE', 'LATE STAGE']
STAGE_COLORS = ['#E74C3C', '#F39C12', '#27AE60']  # Red, Orange, Green

# Sparsity settings per stage (based on train_celeba_cs_32.py)
# Stage 1: t ∈ [0.8, 1.0], sparsity=80%
# Stage 2: t ∈ [0.6, 1.0], sparsity=60%
# Stage 3: t ∈ [0.4, 1.0], sparsity=40%
# Stage 4: t ∈ [0.2, 1.0], sparsity=20%
# Stage 5: t ∈ [0.0, 1.0], sparsity=0%
SPARSITY_CONFIG = {
    'EARLY STAGE': {'sparsity': 0.80, 'active': 51, 'total': 256, 't_range': '[0.8, 1.0]', 'feature': '$M_1$ Coarse'},
    'MIDDLE STAGE': {'sparsity': 0.40, 'active': 154, 'total': 256, 't_range': '[0.4, 1.0]', 'feature': 'Intermediate'},
    'LATE STAGE': {'sparsity': 0.00, 'active': 256, 'total': 256, 't_range': '[0.0, 1.0]', 'feature': '$M_2$ Fine'},
}

# Sample images for each stage (using latest images)
SAMPLE_IMAGES = {
    'EARLY STAGE': 'contents_cs_32/ddpm_celeba_s1_e3.png',      # Stage 1 final
    'MIDDLE STAGE': 'contents_cs_32/ddpm_celeba_s3_e5.png',     # Stage 3 final
    'LATE STAGE': 'contents_cs_32/ddpm_celeba_s5_e23.png',      # Stage 5 latest
}

def load_and_crop_sample(image_path, row=3, col=2, crop_size=32):
    """Load image and crop a single sample from the grid."""
    img = Image.open(image_path)
    img_array = np.array(img)

    # Assuming 4x4 grid, crop one sample
    h, w = crop_size, crop_size
    y_start = row * h
    x_start = col * w

    return img_array[y_start:y_start+h, x_start:x_start+w]


def draw_unet_bottleneck(ax, sparsity, active_channels, total_channels, stage_color):
    """Draw UNet bottleneck with channel sparsity visualization."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Draw simplified UNet shape
    # Encoder (left side - trapezoid going down)
    encoder_x = [0.5, 2, 2, 0.5]
    encoder_y = [5.5, 4, 2, 0.5]
    ax.fill(encoder_x, encoder_y, color='#3498DB', alpha=0.3, edgecolor='#2980B9', linewidth=2)
    ax.text(1.25, 3, 'Enc', ha='center', va='center', fontsize=8, fontweight='bold', color='#2980B9')

    # Decoder (right side - trapezoid going up)
    decoder_x = [8, 9.5, 9.5, 8]
    decoder_y = [4, 5.5, 0.5, 2]
    ax.fill(decoder_x, decoder_y, color='#3498DB', alpha=0.3, edgecolor='#2980B9', linewidth=2)
    ax.text(8.75, 3, 'Dec', ha='center', va='center', fontsize=8, fontweight='bold', color='#2980B9')

    # Bottleneck (center) - the key visualization
    bottleneck_x, bottleneck_y = 5, 3
    bottleneck_width, bottleneck_height = 2.5, 2.5

    # Draw bottleneck box
    rect = FancyBboxPatch((bottleneck_x - bottleneck_width/2, bottleneck_y - bottleneck_height/2),
                          bottleneck_width, bottleneck_height,
                          boxstyle="round,pad=0.05",
                          facecolor='white', edgecolor=stage_color, linewidth=3)
    ax.add_patch(rect)

    # Draw channel grid inside bottleneck
    n_cols = 8
    n_rows = 4
    channel_w = bottleneck_width / (n_cols + 1)
    channel_h = bottleneck_height / (n_rows + 1)

    # Calculate which channels are active
    total_vis_channels = n_cols * n_rows
    active_vis = int(total_vis_channels * (1 - sparsity))

    channel_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            x = bottleneck_x - bottleneck_width/2 + (col + 0.5) * channel_w + channel_w/2
            y = bottleneck_y + bottleneck_height/2 - (row + 0.5) * channel_h - channel_h/2

            if channel_idx < active_vis:
                color = stage_color
                alpha = 0.9
            else:
                color = '#BDC3C7'
                alpha = 0.4

            small_rect = Rectangle((x - channel_w/3, y - channel_h/3),
                                   channel_w * 0.6, channel_h * 0.6,
                                   facecolor=color, alpha=alpha, edgecolor='none')
            ax.add_patch(small_rect)
            channel_idx += 1

    # Arrows connecting encoder -> bottleneck -> decoder
    ax.annotate('', xy=(bottleneck_x - bottleneck_width/2 - 0.2, 3),
                xytext=(2.2, 3),
                arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=2))
    ax.annotate('', xy=(7.8, 3),
                xytext=(bottleneck_x + bottleneck_width/2 + 0.2, 3),
                arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=2))

    # Label
    ax.text(5, 0.3, f'Bottleneck\n{active_channels}/{total_channels} active',
            ha='center', va='center', fontsize=8, fontweight='bold')


def draw_capacity_bar(ax, sparsity, stage_color, stage_name):
    """Draw capacity/sparsity progress bar."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')

    # Background bar
    bar_bg = FancyBboxPatch((0.5, 0.5), 9, 1,
                            boxstyle="round,pad=0.02",
                            facecolor='#ECF0F1', edgecolor='#BDC3C7', linewidth=1)
    ax.add_patch(bar_bg)

    # Active capacity bar
    active_ratio = 1 - sparsity
    if active_ratio > 0:
        bar_active = FancyBboxPatch((0.5, 0.5), 9 * active_ratio, 1,
                                    boxstyle="round,pad=0.02",
                                    facecolor=stage_color, edgecolor='none', alpha=0.8)
        ax.add_patch(bar_active)

    # Text
    ax.text(5, 1, f'{int(active_ratio*100)}% Active',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white' if active_ratio > 0.3 else '#2C3E50')

    # Sparsity label below
    ax.text(5, 0.1, f'Sparsity: {int(sparsity*100)}%',
            ha='center', va='top', fontsize=8, color='#7F8C8D')


def create_figure():
    """Create the main infographic figure."""
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('white')

    # Title
    fig.suptitle('Coordinated Training Timeline: Curriculum Learning × Sparsity',
                 fontsize=18, fontweight='bold', y=0.96, color='#2C3E50')

    # Subtitle
    fig.text(0.5, 0.92, 'Synchronizing noise-level curriculum with model capacity release',
             ha='center', fontsize=12, style='italic', color='#7F8C8D')

    # Create grid: 4 rows x 3 cols
    # Row 0: Stage headers
    # Row 1: Generated samples (noise level)
    # Row 2: UNet sparsity visualization
    # Row 3: Capacity progress bar

    gs = GridSpec(4, 3, figure=fig, height_ratios=[0.3, 1.2, 1.2, 0.5],
                  hspace=0.3, wspace=0.15, left=0.08, right=0.92, top=0.88, bottom=0.08)

    # Row labels on the left
    row_labels = ['', 'Noise Level\n& Output', 'UNet\nBottleneck', 'Capacity\nReleased']
    for i, label in enumerate(row_labels):
        if label:
            fig.text(0.03, 0.78 - i*0.22, label, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='#34495E', rotation=90)

    for col, stage in enumerate(STAGES):
        config = SPARSITY_CONFIG[stage]
        color = STAGE_COLORS[col]

        # Row 0: Stage header
        ax_header = fig.add_subplot(gs[0, col])
        ax_header.axis('off')
        ax_header.set_xlim(0, 10)
        ax_header.set_ylim(0, 2)

        # Stage name box
        header_box = FancyBboxPatch((0.5, 0.3), 9, 1.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='none', alpha=0.9)
        ax_header.add_patch(header_box)
        # Stage name on top line
        ax_header.text(5, 1.3, stage, ha='center', va='center',
                      fontsize=13, fontweight='bold', color='white')
        # Time range on second line inside the box
        ax_header.text(5, 0.7, f't ∈ {config["t_range"]}', ha='center', va='center',
                      fontsize=11, color='white', fontweight='bold')

        # Row 1: Generated samples
        ax_sample = fig.add_subplot(gs[1, col])
        ax_sample.axis('off')

        # Load and display sample image
        base_path = '/home/sunj11/Documents/minDiffusion_curriculum'
        img_path = os.path.join(base_path, SAMPLE_IMAGES[stage])

        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax_sample.imshow(img)

            # Add border
            for spine in ax_sample.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(3)

        # Feature label below
        ax_sample.set_title(config['feature'], fontsize=11, pad=5, color='#2C3E50', fontweight='bold')

        # Row 2: UNet bottleneck visualization
        ax_unet = fig.add_subplot(gs[2, col])
        draw_unet_bottleneck(ax_unet, config['sparsity'], config['active'], config['total'], color)

        # Row 3: Capacity bar
        ax_bar = fig.add_subplot(gs[3, col])
        draw_capacity_bar(ax_bar, config['sparsity'], color, stage)

    # Add arrows between stages using fig.text with arrow characters
    arrow_y = 0.50
    for i in range(2):
        start_x = 0.35 + i * 0.28
        fig.text(start_x, arrow_y, '→', fontsize=30, ha='center', va='center',
                color='#95A5A6', fontweight='bold')

    # Add key insight box at bottom
    insight_text = ("Key Insight: High Noise ↔ High Sparsity (few channels learn coarse $M_1$)\n"
                   "Low Noise ↔ Low Sparsity (all channels active, new channels learn fine $M_2$)")
    fig.text(0.5, 0.02, insight_text, ha='center', va='bottom', fontsize=10,
            style='italic', color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#DEE2E6'))

    return fig


if __name__ == '__main__':
    fig = create_figure()

    # Save figure
    output_path = '/home/sunj11/Documents/minDiffusion_curriculum/figure1_timeline.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")

    # Also save PDF for paper
    pdf_path = '/home/sunj11/Documents/minDiffusion_curriculum/figure1_timeline.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {pdf_path}")

    plt.show()
