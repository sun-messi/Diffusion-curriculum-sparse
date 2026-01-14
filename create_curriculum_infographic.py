"""
Create an infographic for Joint Noise-Sparse Curriculum for Diffusion Training.
"""

import argparse
import json
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Ellipse
from matplotlib.lines import Line2D
from PIL import Image

# Color palette
PRIMARY_BLUE = '#3B82F6'
DARK_BLUE = '#1E40AF'
LIGHT_BLUE = '#60A5FA'
LIGHTER_BLUE = '#93C5FD'
DASHED_GREY = '#9CA3AF'
BACKGROUND = '#FFFFFF'
SECTION_BG = '#E8F4FC'  # Light blue-grey for section backgrounds
TEXT_COLOR = '#1F2937'

# CelebA dataset path
CELEBA_PATH = '/home/sunj11/Documents/U-ViT-fresh/assets/datasets/celeba/celeba/img_align_celeba'

DEFAULT_CONFIG_PATH = '/home/sunj11/Documents/U-ViT-fresh/curriculum_infographic_config.json'


def load_config(path):
    """Load JSON config for layout/style overrides."""
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_color_config(config):
    """Override palette globals from config."""
    global PRIMARY_BLUE, DARK_BLUE, LIGHT_BLUE, LIGHTER_BLUE, DASHED_GREY, BACKGROUND, SECTION_BG, TEXT_COLOR
    colors = config.get('colors', {})
    PRIMARY_BLUE = colors.get('primary_blue', PRIMARY_BLUE)
    DARK_BLUE = colors.get('dark_blue', DARK_BLUE)
    LIGHT_BLUE = colors.get('light_blue', LIGHT_BLUE)
    LIGHTER_BLUE = colors.get('lighter_blue', LIGHTER_BLUE)
    DASHED_GREY = colors.get('dashed_grey', DASHED_GREY)
    BACKGROUND = colors.get('background', BACKGROUND)
    SECTION_BG = colors.get('section_bg', SECTION_BG)
    TEXT_COLOR = colors.get('text_color', TEXT_COLOR)


def load_random_celeba_images(n=27, seed=42):
    """Load n random images from CelebA dataset."""
    random.seed(seed)
    np.random.seed(seed)

    all_files = sorted(os.listdir(CELEBA_PATH))
    selected_files = random.sample(all_files, n)

    images = []
    for fname in selected_files:
        img_path = os.path.join(CELEBA_PATH, fname)
        img = Image.open(img_path).convert('RGB')
        # Center crop to 128x128 for face region
        cx, cy = 89, 121
        x1, x2 = cy - 64, cy + 64
        y1, y2 = cx - 64, cx + 64
        img = img.crop((y1, x1, y2, x2))
        img = img.resize((64, 64), Image.LANCZOS)
        images.append(np.array(img))

    return images


def load_specific_celeba_image(filename):
    """Load a specific image from CelebA dataset."""
    img_path = os.path.join(CELEBA_PATH, filename)
    img = Image.open(img_path).convert('RGB')
    cx, cy = 89, 121
    x1, x2 = cy - 64, cy + 64
    y1, y2 = cx - 64, cx + 64
    img = img.crop((y1, x1, y2, x2))
    img = img.resize((64, 64), Image.LANCZOS)
    return np.array(img)


def add_noise(image, noise_level):
    """
    Add noise to image.
    noise_level: 'high', 'low', or 'clean'
    """
    img = image.astype(np.float32) / 255.0

    if noise_level == 'high':
        # Heavy static noise - almost unrecognizable
        noise = np.random.randn(*img.shape) * 0.85
        noisy = img + noise
        # Add salt and pepper noise
        salt_pepper = np.random.rand(*img.shape[:2])
        noisy[salt_pepper < 0.12] = 1.0
        noisy[salt_pepper > 0.88] = 0.0
    elif noise_level == 'low':
        # Light grain noise - recognizable
        noise = np.random.randn(*img.shape) * 0.12
        noisy = img + noise
    else:  # clean
        noisy = img

    noisy = np.clip(noisy, 0, 1)
    return (noisy * 255).astype(np.uint8)


def create_3x3_grid(images, noise_pattern, gap=4):
    """
    Create a 3x3 grid of images with specified noise pattern.
    """
    grid_size = 64
    total_size = grid_size * 3 + gap * 2

    grid = np.ones((total_size, total_size, 3), dtype=np.uint8) * 255

    for idx, (img, noise_level) in enumerate(zip(images, noise_pattern)):
        row, col = idx // 3, idx % 3
        noisy_img = add_noise(img, noise_level)

        y_start = row * (grid_size + gap)
        x_start = col * (grid_size + gap)

        grid[y_start:y_start+grid_size, x_start:x_start+grid_size] = noisy_img

    return grid


def draw_neural_network(ax, stage_name):
    """
    Draw a 5-3-5 bottleneck neural network with symmetric connections.
    """
    layer_x = [0.05, 0.5, 0.95]
    input_y = np.linspace(0.1, 0.9, 5)
    hidden_y = np.linspace(0.25, 0.75, 3)
    output_y = np.linspace(0.1, 0.9, 5)

    neuron_radius = 0.073
    # Compensate for axes aspect ratio (width=0.23, height=0.16) to make circles look circular
    aspect_ratio = 0.23 / 0.16  # ~1.44

    all_connections = [(i, j) for i in range(5) for j in range(3)]

    if stage_name == 'early':
        active_pairs = [(2, 1)]
    elif stage_name == 'middle':
        active_pairs = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (4, 2)]
    else:
        active_pairs = all_connections.copy()

    # Draw connections
    for i in range(5):
        for j in range(3):
            is_active = (i, j) in active_pairs
            color = PRIMARY_BLUE if is_active else DASHED_GREY
            linestyle = '-' if is_active else '--'
            linewidth = 2.9 if is_active else 1.2
            alpha = 1.0 if is_active else 0.35

            ax.plot([layer_x[0], layer_x[1]], [input_y[i], hidden_y[j]],
                   color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=1)
            ax.plot([layer_x[1], layer_x[2]], [hidden_y[j], output_y[i]],
                   color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=1)

    # Draw neurons (using Ellipse to compensate for axes aspect ratio)
    ellipse_width = neuron_radius * 2 / aspect_ratio
    ellipse_height = neuron_radius * 2
    for y in input_y:
        ax.add_patch(Ellipse((layer_x[0], y), ellipse_width, ellipse_height, color=PRIMARY_BLUE, zorder=2))
    for y in hidden_y:
        ax.add_patch(Ellipse((layer_x[1], y), ellipse_width, ellipse_height, color=PRIMARY_BLUE, zorder=2))
    for y in output_y:
        ax.add_patch(Ellipse((layer_x[2], y), ellipse_width, ellipse_height, color=PRIMARY_BLUE, zorder=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Remove aspect='equal' to allow circles to display properly in wide axes
    ax.axis('off')


def draw_gradient_arrow(
    ax,
    y_center,
    height,
    text,
    arrow_head_ratio=0.04,
    font_size=14,
    start_color=None,
    end_color=None,
):
    """Draw a large gradient arrow with text."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def hex_to_rgb(value):
        value = value.lstrip('#')
        return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))

    if start_color and end_color:
        sr, sg, sb = hex_to_rgb(start_color)
        er, eg, eb = hex_to_rgb(end_color)
    else:
        sr, sg, sb = 0x60, 0xA5, 0xFA
        er, eg, eb = 0x3B, 0x82, 0xF6

    # Gradient body
    n_segments = 150
    body_end = 1 - arrow_head_ratio
    for i in range(n_segments):
        t = i / n_segments
        x = t * body_end
        width = body_end / n_segments * 1.02

        # Color interpolation
        r = int(sr + (er - sr) * t)
        g = int(sg + (eg - sg) * t)
        b = int(sb + (eb - sb) * t)
        color = f'#{r:02x}{g:02x}{b:02x}'

        rect = Rectangle((x, y_center - height/2), width, height, color=color, ec='none')
        ax.add_patch(rect)

    # Arrow head
    triangle = Polygon([
        [body_end, y_center - height * 0.9],
        [1.0, y_center],
        [body_end, y_center + height * 0.9]
    ], color=end_color or PRIMARY_BLUE, ec='none')
    ax.add_patch(triangle)

    # Text
    ax.text(0.47, y_center, text, fontsize=font_size, color='white',
            ha='center', va='center', fontweight='bold')
    ax.axis('off')


def create_infographic(config):
    """Create the full infographic."""
    print("Loading CelebA images...")
    images = load_random_celeba_images(n=27, seed=123)  # Different seed for variety

    # Load a good clean face for legend
    legend_clean_img = load_specific_celeba_image('000050.jpg')

    fig_cfg = config.get('figure', {})
    fig = plt.figure(
        figsize=tuple(fig_cfg.get('figsize', (20, 11))),
        facecolor=BACKGROUND
    )

    title_cfg = config.get('title', {})
    fig.text(0.5, title_cfg.get('y', 0.97),
             title_cfg.get('text', 'Joint Noise-Sparse Curriculum for Diffusion Training'),
             fontsize=title_cfg.get('font_size', 26), fontweight='bold',
             color=TEXT_COLOR, ha='center')

    top_arrow_cfg = config.get('top_arrow', {})
    band_axes = top_arrow_cfg.get('band_axes')
    if band_axes:
        band_ax = fig.add_axes(band_axes)
        band_ax.set_facecolor(top_arrow_cfg.get('band_color', LIGHTER_BLUE))
        band_ax.set_xlim(0, 1)
        band_ax.set_ylim(0, 1)
        band_ax.axis('off')
    top_arrow_ax = fig.add_axes(top_arrow_cfg.get('axes', [0.08, 0.91, 0.84, 0.045]))
    draw_gradient_arrow(top_arrow_ax, 0.5, top_arrow_cfg.get('height', 0.95),
                        top_arrow_cfg.get('text', 'Training Iteration'),
                        font_size=top_arrow_cfg.get('font_size', 14),
                        arrow_head_ratio=top_arrow_cfg.get('arrow_head_ratio', 0.04),
                        start_color=top_arrow_cfg.get('start_color'),
                        end_color=top_arrow_cfg.get('end_color'))

    train_labels = config.get('train_labels', {})
    left_label = train_labels.get('left', {})
    right_label = train_labels.get('right', {})
    fig.text(left_label.get('x', 0.08), left_label.get('y', 0.956),
             left_label.get('text', 'Train Start'),
             fontsize=left_label.get('font_size', 13),
             color=TEXT_COLOR, ha='left', fontweight='bold')
    fig.text(right_label.get('x', 0.92), right_label.get('y', 0.956),
             right_label.get('text', 'Train End'),
             fontsize=right_label.get('font_size', 13),
             color=TEXT_COLOR, ha='right', fontweight='bold')

    section1_cfg = config.get('section1', {})
    section1_bg = fig.add_axes(section1_cfg.get('bg_axes', [0.08, 0.44, 0.84, 0.46]), facecolor=SECTION_BG)
    section1_bg.set_xlim(0, 1)
    section1_bg.set_ylim(0, 1)
    section1_bg.axis('off')
    for x in section1_cfg.get('separators_x', []):
        section1_bg.add_patch(Rectangle(
            (x - section1_cfg.get('separator_width', 0.004) / 2, 0),
            section1_cfg.get('separator_width', 0.004),
            1,
            color=section1_cfg.get('separator_color', BACKGROUND),
            ec='none'
        ))

    stages = ['Early Stage', 'Middle Stage', 'Late Stage']
    header_x = [0.165, 0.5, 0.835]

    for stage, x in zip(stages, header_x):
        header_ax = fig.add_axes([
            x * 0.84 + 0.08 - 0.1,
            section1_cfg.get('header_axes_y', 0.855),
            section1_cfg.get('header_axes_w', 0.2),
            section1_cfg.get('header_axes_h', 0.045)
        ])
        header_ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                           boxstyle="round,pad=0.01,rounding_size=0.35",
                           facecolor=PRIMARY_BLUE, edgecolor='none'))
        header_ax.text(0.5, 0.5, stage, fontsize=section1_cfg.get('header_font_size', 14), fontweight='bold',
                      color='white', ha='center', va='center')
        header_ax.set_xlim(0, 1)
        header_ax.set_ylim(0, 1)
        header_ax.axis('off')

    s1_left = section1_cfg.get('left_label', {})
    fig.text(s1_left.get('x', 0.04), s1_left.get('y', 0.71),
             s1_left.get('text', 'Training Noise Curriculum\nfrom High to Low'),
             fontsize=s1_left.get('font_size', 13), color=TEXT_COLOR,
             ha='center', va='center', rotation=90, fontweight='bold', linespacing=1.5)

    early_pattern = ['high'] * 9
    middle_pattern = ['high', 'high', 'high', 'low', 'low', 'low', 'high', 'high', 'high']
    late_pattern = ['high', 'high', 'high', 'low', 'low', 'low', 'clean', 'clean', 'clean']

    early_grid = create_3x3_grid(images[0:9], early_pattern)
    middle_grid = create_3x3_grid(images[9:18], middle_pattern)
    late_grid = create_3x3_grid(images[18:27], late_pattern)

    grids = [early_grid, middle_grid, late_grid]
    grid_labels = ['Only High Noise Data for Training',
                   'High and Low Noise Data for Training',
                   'All Data: High, Low, and Clean for Training']

    grid_x_positions = section1_cfg.get('grid_x_positions', [0.115, 0.39, 0.665])

    for grid, label, x_pos in zip(grids, grid_labels, grid_x_positions):
        ax = fig.add_axes([
            x_pos,
            section1_cfg.get('grid_axes_y', 0.57),
            section1_cfg.get('grid_axes_w', 0.23),
            section1_cfg.get('grid_axes_h', 0.27)
        ])
        ax.imshow(grid)
        ax.axis('off')
        fig.text(x_pos + section1_cfg.get('grid_axes_w', 0.23) / 2,
                 section1_cfg.get('grid_label_y', 0.54),
                 label, fontsize=section1_cfg.get('grid_label_font_size', 11),
                 color=TEXT_COLOR, ha='center', va='top')

    noise_arrow_cfg = config.get('noise_arrow', {})
    noise_arrow_ax = fig.add_axes(noise_arrow_cfg.get('axes', [0.08, 0.44, 0.84, 0.025]))
    draw_gradient_arrow(noise_arrow_ax, 0.5, noise_arrow_cfg.get('height', 0.6),
                        noise_arrow_cfg.get('text', 'Noise progressively decreases'),
                        font_size=noise_arrow_cfg.get('font_size', 12),
                        arrow_head_ratio=noise_arrow_cfg.get('arrow_head_ratio', 0.04),
                        start_color=noise_arrow_cfg.get('start_color'),
                        end_color=noise_arrow_cfg.get('end_color'))

    sync_cfg = config.get('sync_arrows', {})
    sync_x_positions = sync_cfg.get('x_positions', [0.23, 0.505, 0.78])
    for x in sync_x_positions:
        sync_ax = fig.add_axes([x - 0.01, sync_cfg.get('axes_y', 0.35), 0.02, sync_cfg.get('axes_h', 0.05)])
        sync_ax.set_xlim(0, 1)
        sync_ax.set_ylim(0, 1)
        sync_ax.axis('off')
        sync_ax.annotate('', xy=(0.5, 0), xytext=(0.5, 1),
                        arrowprops=dict(arrowstyle='-|>', color=DARK_BLUE,
                                       lw=sync_cfg.get('line_width', 2.5),
                                       mutation_scale=sync_cfg.get('mutation_scale', 18)))

    section2_cfg = config.get('section2', {})
    section2_bg = fig.add_axes(section2_cfg.get('bg_axes', [0.08, 0.09, 0.84, 0.26]), facecolor=SECTION_BG)
    section2_bg.set_xlim(0, 1)
    section2_bg.set_ylim(0, 1)
    section2_bg.axis('off')
    for x in section2_cfg.get('separators_x', []):
        section2_bg.add_patch(Rectangle(
            (x - section2_cfg.get('separator_width', 0.004) / 2, 0),
            section2_cfg.get('separator_width', 0.004),
            1,
            color=section2_cfg.get('separator_color', BACKGROUND),
            ec='none'
        ))

    s2_left = section2_cfg.get('left_label', {})
    fig.text(s2_left.get('x', 0.04), s2_left.get('y', 0.19),
             s2_left.get('text', 'Model Capacity Curriculum\nfrom Sparsity to Density'),
             fontsize=s2_left.get('font_size', 13), color=TEXT_COLOR,
             ha='center', va='center', rotation=90, fontweight='bold', linespacing=1.5)

    nn_labels = ['Sparse Model', 'Medium Capacity', 'Dense Model']
    stage_names = ['early', 'middle', 'late']
    nn_x_positions = section2_cfg.get('nn_x_positions', [0.115, 0.39, 0.665])

    for label, stage, x_pos in zip(nn_labels, stage_names, nn_x_positions):
        ax = fig.add_axes([
            x_pos,
            section2_cfg.get('nn_axes_y', 0.17),
            section2_cfg.get('nn_axes_w', 0.23),
            section2_cfg.get('nn_axes_h', 0.16)
        ])
        draw_neural_network(ax, stage)
        fig.text(x_pos + section2_cfg.get('nn_axes_w', 0.23) / 2,
                 section2_cfg.get('nn_label_y', 0.145), label,
                 fontsize=section2_cfg.get('nn_label_font_size', 11),
                 color=TEXT_COLOR, ha='center', va='top')

    cap_arrow_cfg = config.get('capacity_arrow', {})
    cap_arrow_ax = fig.add_axes(cap_arrow_cfg.get('axes', [0.08, 0.10, 0.84, 0.025]))
    draw_gradient_arrow(cap_arrow_ax, 0.5, cap_arrow_cfg.get('height', 0.6),
                       cap_arrow_cfg.get('text', 'Active neurons and capacity progressively increase'),
                       font_size=cap_arrow_cfg.get('font_size', 12),
                       arrow_head_ratio=cap_arrow_cfg.get('arrow_head_ratio', 0.04),
                       start_color=cap_arrow_cfg.get('start_color'),
                       end_color=cap_arrow_cfg.get('end_color'))

    legend_cfg = config.get('legend', {})
    legend_y = legend_cfg.get('y', 0.005)
    img_size = legend_cfg.get('img_size', 0.032)

    sample_high = add_noise(images[5], 'high')
    sample_low = add_noise(images[10], 'low')
    sample_clean = legend_clean_img

    legend_x_positions = legend_cfg.get('item_x_positions', [0.30, 0.44, 0.56])
    legend_items = [
        (sample_high, 'High Noise', legend_x_positions[0]),
        (sample_low, 'Low Noise', legend_x_positions[1]),
        (sample_clean, 'Clean', legend_x_positions[2]),
    ]

    for sample, label, x_pos in legend_items:
        img_ax = fig.add_axes([x_pos, legend_y, img_size, img_size * 20/14])
        img_ax.imshow(sample)
        img_ax.axis('off')
        fig.text(x_pos + img_size + 0.01, legend_y + img_size * 20/14 / 2,
                label, fontsize=legend_cfg.get('label_font_size', 11),
                color=TEXT_COLOR, ha='left', va='center')

    line_ax1 = fig.add_axes([legend_cfg.get('active_x', 0.68), legend_y + 0.018, 0.05, 0.012])
    line_ax1.plot([0, 1], [0.5, 0.5], color=PRIMARY_BLUE, linewidth=3)
    line_ax1.set_xlim(0, 1)
    line_ax1.set_ylim(0, 1)
    line_ax1.axis('off')
    fig.text(0.74, legend_y + img_size * 20/14 / 2, 'Active',
             fontsize=legend_cfg.get('label_font_size', 11),
             color=TEXT_COLOR, ha='left', va='center')

    line_ax2 = fig.add_axes([legend_cfg.get('inactive_x', 0.80), legend_y + 0.018, 0.05, 0.012])
    line_ax2.plot([0, 1], [0.5, 0.5], color=DASHED_GREY, linewidth=2.5, linestyle='--')
    line_ax2.set_xlim(0, 1)
    line_ax2.set_ylim(0, 1)
    line_ax2.axis('off')
    fig.text(0.86, legend_y + img_size * 20/14 / 2, 'Inactive',
             fontsize=legend_cfg.get('label_font_size', 11),
             color=TEXT_COLOR, ha='left', va='center')

    output_path = '/home/sunj11/Documents/U-ViT-fresh/analysis_outputs/curriculum_infographic.png'
    plt.savefig(output_path, dpi=fig_cfg.get('dpi', 200), bbox_inches='tight', facecolor=BACKGROUND)
    print(f"Saved to {output_path}")

    plt.savefig('/home/sunj11/Documents/U-ViT-fresh/curriculum_infographic.png',
                dpi=fig_cfg.get('dpi', 200), bbox_inches='tight', facecolor=BACKGROUND)
    print("Also saved to current directory")

    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create curriculum infographic with optional config overrides.")
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH, help='Path to JSON config')
    args = parser.parse_args()

    config = load_config(args.config)
    apply_color_config(config)
    create_infographic(config)
