#!/usr/bin/env python3
"""Matrix digital-rain overlay: green falling numbers, white-hot heads, the video
revealed through the code. Inspired by the green 'coding in After Effects' graphic."""
import argparse, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

GREEN_BODY = np.array([30, 220, 70], dtype=np.float32)    # matrix green
GREEN_HEAD = np.array([200, 255, 210], dtype=np.float32)  # white-hot leading glyph
KNICKS_BLUE_HL = np.array([60, 140, 255], dtype=np.float32)    # ESB blue top -> blue digits
KNICKS_ORANGE_HL = np.array([255, 150, 45], dtype=np.float32)  # warm city lights -> orange digits
FONT_PATH = "/System/Library/Fonts/HelveticaNeue.ttc"


def build_atlas(cell, font_size, chars):
    font = ImageFont.truetype(FONT_PATH, font_size)
    atlas = np.zeros((len(chars), cell, cell), dtype=np.float32)
    for i, ch in enumerate(chars):
        img = Image.new("L", (cell, cell), 0)
        d = ImageDraw.Draw(img)
        bbox = d.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (cell - w) / 2 - bbox[0]
        y = (cell - h) / 2 - bbox[1]
        d.text((x, y), ch, fill=255, font=font)
        atlas[i] = np.asarray(img, dtype=np.float32) / 255.0
    return atlas


def dilate(a, k=1):
    """3x3 (k=1) grayscale dilation so detected features grow into solid shapes."""
    out = a.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            out = np.maximum(out, np.roll(np.roll(a, dy, 0), dx, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--fps", default="30")
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--reveal", type=float, default=0.9, help="how strongly the video's lights show as digits")
    ap.add_argument("--rain", type=float, default=0.0, help="strength of the falling-rain streaks (0 = none)")
    ap.add_argument("--bluegain", type=float, default=7.0, help="render ESB blue lighting as blue digits")
    ap.add_argument("--orangegain", type=float, default=3.5, help="render warm city lights as orange digits")
    ap.add_argument("--chars", default="LGK", help="characters used for the effect")
    ap.add_argument("--glow", type=float, default=0.8, help="bloom/glow strength on the characters")
    ap.add_argument("--glowradius", type=float, default=13.0)
    ap.add_argument("--expand", type=int, default=1, help="grow blue/orange features by N cells to build shape")
    ap.add_argument("--esbstrength", type=float, default=1.6, help="how strongly blue/orange chars build shape")
    ap.add_argument("--ripplefreq", type=float, default=0.7, help="shimmer pulses per second")
    ap.add_argument("--ripplewl", type=float, default=7.0, help="shimmer wavelength in cells")
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
    atlas = build_atlas(cell, cell + 2, args.chars)
    N = len(args.chars)

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

        # per-cell color of the scene -> detect ESB blue + warm lights
        cR = frame[..., 0].reshape(gh, cell, gw, cell).mean(axis=(1, 3)) / 255.0
        cG = frame[..., 1].reshape(gh, cell, gw, cell).mean(axis=(1, 3)) / 255.0
        cB = frame[..., 2].reshape(gh, cell, gw, cell).mean(axis=(1, 3)) / 255.0
        lit = np.clip((cellL - 0.05) / 0.25, 0, 1)
        blueness = np.clip((cB - np.maximum(cR, cG)) * args.bluegain, 0, 1) * lit  # ESB blue top
        warmth = np.clip((np.minimum(cR, cG) - cB) * args.orangegain, 0, 1) * lit  # warm lights

        # grow the blue (ESB) + orange (warm buildings) features so the LGK
        # characters cluster and BUILD their lit shapes, then pulse them with motion
        blueness = dilate(blueness, args.expand)
        warmth = dilate(warmth, args.expand)
        feature = np.maximum(blueness, warmth)
        rr = np.arange(gh)[:, None].astype(np.float32)
        cc = np.arange(gw)[None, :].astype(np.float32)
        pulse = 0.55 + 0.45 * np.sin(2 * np.pi * t * args.ripplefreq
                                     - (rr * 0.6 + cc * 0.35) / args.ripplewl)    # traveling shimmer
        esb_shape = feature * (0.7 + 0.5 * vboost) * args.esbstrength * pulse     # glowing pulsing shape

        # falling streaks
        head = np.mod(speed * t + phase, period)                                   # (gw,)
        dist = head[None, :] - rows                                                # (gh,gw) >=0 is the tail above head
        rain_b = np.clip(1.0 - dist / tail[None, :], 0, 1) * (dist >= 0)
        headness = np.clip(1.0 - np.abs(dist), 0, 1)                               # ~1 only at the leading glyph

        # reveal is ALWAYS on (the scene renders in digits regardless of rain);
        # rain streaks add motion and brighten on top.
        intensity = np.clip(args.reveal * vboost
                            + args.rain * rain_b * (0.45 + 0.7 * vboost), 0, 1)
        intensity = np.maximum(intensity, feature * 0.9)  # blue/orange features render solidly
        intensity = np.clip(np.maximum(intensity, esb_shape), 0, 1)  # ESB field builds the shape

        # scrolling digit identities
        scroll = head.astype(np.int32)
        ridx = (np.arange(gh)[:, None] + scroll[None, :]) % RF
        digit_idx = randfield[ridx, cols]                                          # (gh,gw)
        tiles = atlas[digit_idx]                                                   # (gh,gw,cell,cell)
        mask = tiles.transpose(0, 2, 1, 3).reshape(H, W)

        # per-cell color: green body, white-hot at the head (heads only when raining),
        # then override toward Knicks orange (warm lights) and blue (ESB top)
        hn = (headness * (1.0 if args.rain > 0 else 0.0))[..., None]
        color = GREEN_BODY * (1 - hn) + GREEN_HEAD * hn                            # (gh,gw,3)
        color = color * (1 - warmth[..., None]) + KNICKS_ORANGE_HL * warmth[..., None]
        color = color * (1 - blueness[..., None]) + KNICKS_BLUE_HL * blueness[..., None]
        color = color * intensity[..., None]
        color = np.repeat(np.repeat(color, cell, axis=0), cell, axis=1)           # (H,W,3)

        # lift the dark night footage so the real video reads, then green-grade it
        bright = np.clip(frame * args.bright + 6, 0, 255)
        gt = args.greentint
        graded = bright * (np.array([1, 1, 1], np.float32) * (1 - gt)
                           + np.array([0.45, 1.15, 0.6], np.float32) * gt)
        base = np.clip(graded, 0, 255) * args.basevis

        # characters sit ON TOP of the visible video; lit areas read as glowing code
        char = color * mask[..., None]                                            # (H,W,3) colored glyphs
        cimg = Image.fromarray(np.clip(char, 0, 255).astype(np.uint8))
        glow = np.asarray(cimg.filter(ImageFilter.GaussianBlur(args.glowradius)), np.float32) * args.glow
        out = np.clip(base + char + glow, 0, 255).astype(np.uint8)
        writer.stdin.write(out.tobytes())
        n += 1

    writer.stdin.close()
    reader.wait(); writer.wait()
    print(f"done: {n} frames -> {args.output} ({W}x{H})", file=sys.stderr)


if __name__ == "__main__":
    main()
