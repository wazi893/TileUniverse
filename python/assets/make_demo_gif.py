"""Render a terminal-style animated GIF of the flagship tile-CPU demo.
Pure Pillow (no ffmpeg). Output: python/assets/demo_tilecpu.gif
Numbers are the real captured output of `examples/v2_compiled_hello.rs`.
"""
from PIL import Image, ImageDraw, ImageFont

W, H = 820, 548
BG = (13, 17, 23)
BAR = (22, 27, 34)
G = (63, 185, 80)      # green (prompt / output)
Wt = (230, 237, 243)   # white (command)
C = (139, 148, 158)    # comment gray
CODE = (201, 209, 217)
ACC = (121, 192, 255)  # accent blue (keywords / numbers)
DIM = (110, 118, 129)

FONT = "C:/Windows/Fonts/consola.ttf"
FONTB = "C:/Windows/Fonts/consolab.ttf"
f = ImageFont.truetype(FONT, 18)
fb = ImageFont.truetype(FONTB, 18)

# Each line is a list of segments: (text, color) or (text, color, "bold")
lines = [
    [("$ ", G), ("cargo run --release --example v2_compiled_hello", Wt)],
    [],
    [("// a C-like program, compiled to a CPU built from logic gates", C)],
    [("int ", ACC), ("main", Wt), ("() {", CODE)],
    [("    print_char(72); print_char(69); ...   ", CODE), ('// "HELLO, TILE CPU!"', C)],
    [("    return 0;", CODE)],
    [("}", CODE)],
    [],
    [("→ compiled ", DIM), ("55", ACC), (" instruction words from source", DIM)],
    [("→ running on the physical tile CPU ...", DIM)],
    [("→ halted after ", DIM), ("216", ACC), (" tile-CPU ticks", DIM)],
    [],
    [("  console output (computed in tiles):", DIM)],
    [("  ┌──────────────────┐", DIM)],
    [("  │ ", DIM), ("HELLO, TILE CPU!", G, "bold"), (" │", DIM)],
    [("  └──────────────────┘", DIM)],
    [],
    [("bit-identical to the software reference · 3,000+ tests", C)],
]

X0, Y0, LH = 24, 50, 26


def render(n, cursor=True):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # title bar
    d.rectangle([0, 0, W, 36], fill=BAR)
    for i, col in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([16 + i * 22, 12, 28 + i * 22, 24], fill=col)
    d.text((W // 2, 18), "tile-cpu  —  HELLO from logic gates", font=f, fill=DIM, anchor="mm")
    # lines
    last_x = X0
    last_y = Y0
    for li in range(n):
        y = Y0 + li * LH
        x = X0
        for seg in lines[li]:
            txt, col = seg[0], seg[1]
            fnt = fb if len(seg) == 3 else f
            d.text((x, y), txt, font=fnt, fill=col)
            x += fnt.getlength(txt)
        if lines[li]:
            last_x, last_y = x, y
    if cursor:
        d.rectangle([last_x + 1, last_y + 2, last_x + 11, last_y + 20], fill=G)
    return img


frames, durations = [], []
for n in range(1, len(lines) + 1):
    frames.append(render(n, cursor=True))
    durations.append(90 if not lines[n - 1] else 165)
# blink + hold on final frame
frames.append(render(len(lines), cursor=False)); durations.append(500)
frames.append(render(len(lines), cursor=True));  durations.append(500)
frames.append(render(len(lines), cursor=False)); durations.append(2600)

out = "python/assets/demo_tilecpu.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
               loop=0, optimize=True, disposal=2)
print("wrote", out, "| frames:", len(frames))
