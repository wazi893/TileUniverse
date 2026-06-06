"""
TileUniverse Hero Animation Generator

Generates animated GIF of parallel Game of Life worlds evolving simultaneously.
Perfect for README hero image, landing page, and social media.

Requirements:
    pip install pillow numpy

Output: assets/hero_animation.gif
"""

import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: PIL not found. Install with: pip install pillow")
    exit(1)

# Brand colors (RGB tuples)
COLORS = {
    "midnight_black": (13, 15, 17),
    "slate_gray": (27, 30, 34),
    "soft_white": (234, 234, 234),
    "muted_gray": (102, 102, 102),
    "quantum_cyan": (0, 228, 255),
    "deep_cyan": (0, 153, 170),
}


def game_of_life_step(world):
    """Perform one step of Conway's Game of Life."""
    neighbors = (
        np.roll(np.roll(world, 1, 0), 1, 1) +
        np.roll(np.roll(world, 1, 0), 0, 1) +
        np.roll(np.roll(world, 1, 0), -1, 1) +
        np.roll(np.roll(world, 0, 0), 1, 1) +
        np.roll(np.roll(world, 0, 0), -1, 1) +
        np.roll(np.roll(world, -1, 0), 1, 1) +
        np.roll(np.roll(world, -1, 0), 0, 1) +
        np.roll(np.roll(world, -1, 0), -1, 1)
    )
    birth = (world == 0) & (neighbors == 3)
    survive = (world == 1) & ((neighbors == 2) | (neighbors == 3))
    return (birth | survive).astype(np.uint8)


def world_to_image(world, cell_size=4, alive_color=COLORS["quantum_cyan"], dead_color=COLORS["midnight_black"]):
    """Convert world array to PIL Image."""
    h, w = world.shape
    img = Image.new('RGB', (w * cell_size, h * cell_size), dead_color)
    draw = ImageDraw.Draw(img)

    for y in range(h):
        for x in range(w):
            if world[y, x]:
                draw.rectangle(
                    [x * cell_size, y * cell_size, (x + 1) * cell_size - 1, (y + 1) * cell_size - 1],
                    fill=alive_color
                )

    return img


def create_multi_world_frame(worlds, cell_size=3, padding=8, border_width=2):
    """Create a single frame showing multiple worlds in a grid."""
    n_worlds = len(worlds)
    cols = int(np.ceil(np.sqrt(n_worlds)))
    rows = int(np.ceil(n_worlds / cols))

    world_h, world_w = worlds[0].shape
    img_w = world_w * cell_size
    img_h = world_h * cell_size

    total_w = cols * img_w + (cols + 1) * padding
    total_h = rows * img_h + (rows + 1) * padding + 50  # Extra space for title

    frame = Image.new('RGB', (total_w, total_h), COLORS["midnight_black"])
    draw = ImageDraw.Draw(frame)

    # Draw title
    title = "TileUniverse - Parallel World Simulation"
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Center title
    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((total_w - text_w) // 2, 12), title, fill=COLORS["soft_white"], font=font)

    # Draw each world
    for idx, world in enumerate(worlds):
        row = idx // cols
        col = idx % cols

        x = padding + col * (img_w + padding)
        y = 45 + padding + row * (img_h + padding)

        # Draw border
        draw.rectangle(
            [x - border_width, y - border_width, x + img_w + border_width - 1, y + img_h + border_width - 1],
            outline=COLORS["deep_cyan"],
            width=border_width
        )

        # Draw world
        world_img = world_to_image(world, cell_size)
        frame.paste(world_img, (x, y))

        # Draw world label
        label = f"W{idx}"
        draw.text((x + 2, y + 2), label, fill=COLORS["quantum_cyan"], font=font)

    return frame


def generate_hero_gif(
    n_worlds=9,
    world_size=48,
    n_frames=60,
    cell_size=4,
    fps=15,
    output_path=None
):
    """Generate the hero animation GIF."""
    if output_path is None:
        output_path = Path(__file__).parent.parent / "assets" / "hero_animation.gif"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating hero animation...")
    print(f"  Worlds: {n_worlds}")
    print(f"  World size: {world_size}x{world_size}")
    print(f"  Frames: {n_frames}")
    print(f"  FPS: {fps}")
    print()

    # Initialize worlds with different seeds
    worlds = []
    for i in range(n_worlds):
        np.random.seed(42 + i * 17)
        world = (np.random.random((world_size, world_size)) < 0.35).astype(np.uint8)
        # Pre-evolve each world a different amount for variety
        for _ in range(i * 3):
            world = game_of_life_step(world)
        worlds.append(world)

    # Generate frames
    frames = []
    for frame_idx in range(n_frames):
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx}/{n_frames}...")

        frame = create_multi_world_frame(worlds, cell_size=cell_size)
        frames.append(frame)

        # Evolve all worlds
        worlds = [game_of_life_step(w) for w in worlds]

    # Save GIF
    print(f"\nSaving to {output_path}...")
    duration = int(1000 / fps)  # milliseconds per frame
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )

    print(f"Done! File size: {output_path.stat().st_size / 1024:.1f} KB")
    return output_path


def generate_single_world_gif(
    world_size=128,
    n_frames=100,
    cell_size=4,
    fps=20,
    output_path=None
):
    """Generate a simple single-world animation."""
    if output_path is None:
        output_path = Path(__file__).parent.parent / "assets" / "gol_animation.gif"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating single-world animation...")

    np.random.seed(42)
    world = (np.random.random((world_size, world_size)) < 0.3).astype(np.uint8)

    frames = []
    for frame_idx in range(n_frames):
        if frame_idx % 20 == 0:
            print(f"  Frame {frame_idx}/{n_frames}...")

        frame = world_to_image(world, cell_size)
        frames.append(frame)
        world = game_of_life_step(world)

    print(f"\nSaving to {output_path}...")
    duration = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )

    print(f"Done! File size: {output_path.stat().st_size / 1024:.1f} KB")
    return output_path


def generate_banner_image(output_path=None):
    """Generate a static banner image for social media / headers."""
    if output_path is None:
        output_path = Path(__file__).parent.parent / "assets" / "banner.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Banner dimensions (Twitter/GitHub header ratio)
    width, height = 1500, 500

    img = Image.new('RGB', (width, height), COLORS["midnight_black"])
    draw = ImageDraw.Draw(img)

    # Generate background pattern (subtle grid of GoL worlds)
    np.random.seed(42)
    cell_size = 3
    n_cells_x = width // cell_size
    n_cells_y = height // cell_size

    bg_world = (np.random.random((n_cells_y, n_cells_x)) < 0.15).astype(np.uint8)
    for _ in range(10):
        bg_world = game_of_life_step(bg_world)

    # Draw faded background pattern
    fade_color = tuple(c // 6 for c in COLORS["quantum_cyan"])
    for y in range(n_cells_y):
        for x in range(n_cells_x):
            if bg_world[y, x]:
                draw.rectangle(
                    [x * cell_size, y * cell_size, (x + 1) * cell_size - 1, (y + 1) * cell_size - 1],
                    fill=fade_color
                )

    # Add gradient overlay
    for y in range(height):
        alpha = int(255 * (1 - abs(y - height/2) / (height/2)) * 0.3)
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            factor = 1 + alpha / 255 * 0.2
            img.putpixel((x, y), (min(255, int(r * factor)), min(255, int(g * factor)), min(255, int(b * factor))))

    # Draw title text
    try:
        title_font = ImageFont.truetype("arial.ttf", 72)
        subtitle_font = ImageFont.truetype("arial.ttf", 24)
        stats_font = ImageFont.truetype("arial.ttf", 36)
    except:
        title_font = ImageFont.load_default()
        subtitle_font = title_font
        stats_font = title_font

    # Title
    title = "TileUniverse"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    text_w = bbox[2] - bbox[0]
    draw.text(((width - text_w) // 2, 150), title, fill=COLORS["soft_white"], font=title_font)

    # Subtitle
    subtitle = "GPU-Accelerated Parallel Universe Simulation Engine"
    bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    text_w = bbox[2] - bbox[0]
    draw.text(((width - text_w) // 2, 240), subtitle, fill=COLORS["muted_gray"], font=subtitle_font)

    # Stats
    stats = "40B evals/sec  •  100+ parallel worlds  •  CUDA accelerated"
    bbox = draw.textbbox((0, 0), stats, font=stats_font)
    text_w = bbox[2] - bbox[0]
    draw.text(((width - text_w) // 2, 340), stats, fill=COLORS["quantum_cyan"], font=stats_font)

    img.save(output_path, 'PNG')
    print(f"Generated banner: {output_path}")
    return output_path


def main():
    """Generate all hero visualizations."""
    print("=" * 60)
    print("  TileUniverse Hero Visualization Generator")
    print("=" * 60)
    print()

    generate_hero_gif()
    print()

    generate_single_world_gif()
    print()

    generate_banner_image()
    print()

    print("=" * 60)
    print("  All hero visualizations generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
