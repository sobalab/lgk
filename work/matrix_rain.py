#!/usr/bin/env python3
"""Matrix digital-rain overlay: green falling numbers, white-hot heads, the video
revealed through the code. Inspired by the green 'coding in After Effects' graphic."""
import argparse, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GREEN_BODY = np.array([30, 220, 70], dtype=np.float32)    # matrix green
GREEN_HEAD = np.array([200, 255, 210], dtype=np.float32)  # white-hot leading glyph
DIGITS = "0123456789"
FONT_PATH = "/System/Library/Fonts/Menlo.ttc"


def build_atlas(cell, font_size):
    font = ImageFont.truetype(FONT_PATH, font_size)
    atlas = np.zeros((len(DIGITS), cell, cell), dtype=np.float32)
    for i, ch in enumerate(DIGITS):
        img = Image.new("L", (cell, cell), 0)
        d = ImageDraw.Draw(img)
        bbox = d.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (cell - w) / 2 - bbox[0]
        y = (cell - h) / 2 - bbox[1]
        d.text((x, y), ch, fill=255, font=font)
        atlas[i] = np.asarray(img, dtype=np.float32) / 255.0
    return atlas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--fps", default="30")
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--reveal", type=float, default=0.7, help="how strongly the video's lights show as digits")
    ap.add_argument("--rain", type=float, default=0.85, help="strength of the falling-rain streaks")
    ap.add_argument("--gamma", type=float, default=0.55)
    ap.add_argument("--gain", type=float, default=1.6)
    ap.add_argument("--floor", type=float, default=0.05, help="black level cut so the dark sky stays empty")
    ap.add_argument("--basevis", type=float, default=0.85, help="how visible the underlying video is")
    ap.add_argument("--bright", type=float, default=2.2, help="lift the dark night footage so it reads as real video")
    ap.add_argument("--greentint", type=float, default=0.55, help="0 = natural video color, 1 = full green matrix grade")
    args = ap.parse_args()

    fps = eval(args.fps) if "/" in str(args.fps) else float(args.fps)
    cell = args.cell
    W = (args.width // cell) * cell
    H = (args.height // cell) * cell
    gw, gh = W // cell, H // cell
    atlas = build_atlas(cell, cell + 2)
    N = len(DIGITS)

    # deterministic per-column rain params + random digit field
    rng = np.random.default_rng(7)
    speed = rng.uniform(7, 20, size=gw).astype(np.float32)        # rows/sec
    tail = rng.uniform(9, 24, size=gw).astype(np.float32)         # streak length
    period = gh + tail + rng.uniform(4, 30, size=gw).astype(np.float32)
    phase = rng.uniform(0, 1, size=gw).astype(np.float32) * period
    RF = 512
    randfield = rng.integers(0, N, size=(RF, gw))                 # scrolling digit identities

    rows = np.arange(gh)[:, None].astype(np.float32)              # (gh,1)
    cols = np.arange(gw)[None, :]                                 # (1,gw)

    reader = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", args.input, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", args.output],
        stdin=subprocess.PIPE)

    frame_bytes = args.width * args.height * 3
    n = 0
    while True:
        raw = reader.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, np.uint8).reshape(args.height, args.width, 3).astype(np.float32)[:H, :W]
        t = n / fps

        lum = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        cellL = lum.reshape(gh, cell, gw, cell).mean(axis=(1, 3)) / 255.0          # (gh,gw)
        lifted = np.clip((cellL - args.floor) / (1 - args.floor), 0, 1)           # cut the black sky
        vboost = np.clip(np.power(lifted, args.gamma) * args.gain, 0, 1)          # scene reveal

        # falling streaks
        head = np.mod(speed * t + phase, period)                                   # (gw,)
        dist = head[None, :] - rows                                                # (gh,gw) >=0 is the tail above head
        rain_b = np.clip(1.0 - dist / tail[None, :], 0, 1) * (dist >= 0)
        headness = np.clip(1.0 - np.abs(dist), 0, 1)                               # ~1 only at the leading glyph

        # reveal is ALWAYS on (the scene renders in digits regardless of rain);
        # rain streaks add motion and brighten on top.
        intensity = np.clip(args.reveal * vboost
                            + args.rain * rain_b * (0.45 + 0.7 * vboost), 0, 1)

        # scrolling digit identities
        scroll = head.astype(np.int32)
        ridx = (np.arange(gh)[:, None] + scroll[None, :]) % RF
        digit_idx = randfield[ridx, cols]                                          # (gh,gw)
        tiles = atlas[digit_idx]                                                   # (gh,gw,cell,cell)
        mask = tiles.transpose(0, 2, 1, 3).reshape(H, W)

        # per-cell color: green body, white-hot at the head
        hn = headness[..., None]
        color = GREEN_BODY * (1 - hn) + GREEN_HEAD * hn                            # (gh,gw,3)
        color = color * intensity[..., None]
        color = np.repeat(np.repeat(color, cell, axis=0), cell, axis=1)           # (H,W,3)

        # lift the dark night footage so the real video reads, then green-grade it
        bright = np.clip(frame * args.bright + 6, 0, 255)
        gt = args.greentint
        graded = bright * (np.array([1, 1, 1], np.float32) * (1 - gt)
                           + np.array([0.45, 1.15, 0.6], np.float32) * gt)
        base = np.clip(graded, 0, 255) * args.basevis

        # bright digits sit ON TOP of the visible video; lit areas read as glowing code
        out = np.clip(base + color * mask[..., None], 0, 255).astype(np.uint8)
        writer.stdin.write(out.tobytes())
        n += 1

    writer.stdin.close()
    reader.wait(); writer.wait()
    print(f"done: {n} frames -> {args.output} ({W}x{H})", file=sys.stderr)


if __name__ == "__main__":
    main()
