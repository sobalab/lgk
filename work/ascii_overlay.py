#!/usr/bin/env python3
"""Translucent ASCII overlay tinted in Knicks blue/orange, composited over a video."""
import argparse, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

KNICKS_BLUE   = np.array([0, 107, 182], dtype=np.float32)   # #006BB6
KNICKS_ORANGE = np.array([245, 132, 38], dtype=np.float32)  # #F58426
RAMP = " .:-=+*#%@"  # darkest -> brightest; leading space => pure video in dark areas
FONT_PATH = "/System/Library/Fonts/Menlo.ttc"


def build_atlas(cell, font_size):
    """One (cell x cell) grayscale 0..1 mask per ramp glyph."""
    font = ImageFont.truetype(FONT_PATH, font_size)
    atlas = np.zeros((len(RAMP), cell, cell), dtype=np.float32)
    for i, ch in enumerate(RAMP):
        img = Image.new("L", (cell, cell), 0)
        d = ImageDraw.Draw(img)
        bbox = d.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (cell - w) / 2 - bbox[0]
        y = (cell - h) / 2 - bbox[1]
        d.text((x, y), ch, fill=255, font=font)
        atlas[i] = np.asarray(img, dtype=np.float32) / 255.0
    return atlas


def smoothstep(x):
    return x * x * (3 - 2 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cell", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.85, help="overlay opacity 0..1")
    ap.add_argument("--gamma", type=float, default=0.5, help="<1 lifts midtones so dim lights register")
    ap.add_argument("--gain", type=float, default=1.7)
    ap.add_argument("--glow", type=float, default=0.6, help="additive neon glow on top of blend")
    ap.add_argument("--basewash", type=float, default=0.30, help="blue duotone wash on the underlying video")
    ap.add_argument("--desat", type=float, default=0.6, help="pull underlying video toward grayscale so Knicks tint dominates")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--fps", default="30")
    args = ap.parse_args()

    cell = args.cell
    W = (args.width // cell) * cell
    H = (args.height // cell) * cell
    gw, gh = W // cell, H // cell
    atlas = build_atlas(cell, cell + 1)
    N = len(RAMP)

    reader = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", args.input, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", args.fps, "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", args.output],
        stdin=subprocess.PIPE)

    frame_bytes = args.width * args.height * 3
    n = 0
    while True:
        raw = reader.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, np.uint8).reshape(args.height, args.width, 3).astype(np.float32)
        frame = frame[:H, :W]  # crop to cell multiple

        lum = (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2])  # (H,W)

        # desaturate underlying video so the Knicks palette isn't fighting native colors
        gray3 = np.repeat(lum[..., None], 3, axis=2)
        frame = frame * (1 - args.desat) + gray3 * args.desat
        # Knicks-blue duotone wash on the (now neutral) underlying video
        blue_duo = (lum[..., None] / 255.0) * KNICKS_BLUE
        frame = frame * (1 - args.basewash) + blue_duo * args.basewash

        cellL = lum.reshape(gh, cell, gw, cell).mean(axis=(1, 3)) / 255.0  # (gh,gw)
        b = np.clip(np.power(cellL, args.gamma) * args.gain, 0, 1)

        idx = np.clip((b * (N - 1)).round().astype(np.int32), 0, N - 1)
        tiles = atlas[idx]                                    # (gh,gw,cell,cell)
        mask = tiles.transpose(0, 2, 1, 3).reshape(H, W)      # (H,W) glyph coverage 0..1

        tb = np.clip((b - 0.4) / 0.45, 0, 1)                  # only bright cells turn orange
        t = smoothstep(tb)[..., None]                         # bright -> orange, dim -> blue
        tint = KNICKS_BLUE * (1 - t) + KNICKS_ORANGE * t      # (gh,gw,3)
        tint = np.repeat(np.repeat(tint, cell, axis=0), cell, axis=1)  # (H,W,3)

        cov = (mask * args.alpha)[..., None]
        out = frame * (1 - cov) + tint * cov
        out = out + tint * (mask[..., None] * args.glow)      # neon glow
        out = np.clip(out, 0, 255).astype(np.uint8)

        writer.stdin.write(out.tobytes())
        n += 1

    writer.stdin.close()
    reader.wait(); writer.wait()
    print(f"done: {n} frames -> {args.output} ({W}x{H})", file=sys.stderr)


if __name__ == "__main__":
    main()
