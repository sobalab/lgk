#!/usr/bin/env python3
"""Construct the Empire State Building out of LGK characters, anchored to the real
spire visible in the footage, extending the full building downward over the video.
Knicks blue/orange bands, glow, and a top-to-bottom build-out animation."""
import argparse, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import os
FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "CourierPrime-Regular.ttf")
SPIRE_BLUE = np.array([95, 165, 255], np.float32)   # bright glowing spire
SHAFT_BLUE = np.array([48, 92, 245], np.float32)    # deep royal main shaft (matches photo)
KNICKS_ORANGE = np.array([255, 140, 40], np.float32)  # amber crown
GREEN = np.array([30, 200, 70], np.float32)
EDGE = np.array([225, 240, 255], np.float32)   # bright construction front

# Empire State Building half-width profile (yn: 0=spire tip .. 1=base), in cells.
# Shaft stays constant width down to the base (no flare) so bottom rows match the middle.
PROF_Y = np.array([0.00, 0.17, 0.21, 0.34, 0.40, 0.50, 1.00])
PROF_W = np.array([0.40, 0.40, 1.60, 2.40, 4.60, 5.00, 5.00])


def build_atlas(cell, font_size, chars):
    font = ImageFont.truetype(FONT_PATH, font_size)
    atlas = np.zeros((len(chars), cell, cell), np.float32)
    for i, ch in enumerate(chars):
        img = Image.new("L", (cell, cell), 0)
        d = ImageDraw.Draw(img)
        b = d.textbbox((0, 0), ch, font=font)
        d.text(((cell - (b[2] - b[0])) / 2 - b[0], (cell - (b[3] - b[1])) / 2 - b[1]),
               ch, fill=255, font=font)
        atlas[i] = np.asarray(img, np.float32) / 255.0
    return atlas


def read_frames(path, W, H):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    fb = W * H * 3
    while True:
        raw = p.stdout.read(fb)
        if len(raw) < fb:
            break
        yield np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(np.float32)
    p.wait()


def detect_anchor(frame, H):
    """Lock onto the ESB spire = the TOPMOST strong-blue pixel (the tallest blue
    feature in the skyline), not the average of every blue light in frame."""
    R, G, B = frame[..., 0], frame[..., 1], frame[..., 2]
    blue = np.clip(B - np.maximum(R, G), 0, 255) * (((R + G + B) / 3) > 20)
    band = blue.copy()
    band[int(H * 0.60):] = 0
    band[:int(H * 0.03)] = 0
    mx = band.max()
    if mx < 35:
        return None
    thr = max(mx * 0.45, 35)
    ys, xs = np.where(band > thr)
    tip_y = ys.min()
    near_tip = ys < tip_y + 22                      # only the spire top cluster
    tip_x = xs[near_tip].mean()
    return tip_x, tip_y


def smooth(a, win=11):
    a = np.array(a, np.float32)
    k = np.hanning(win); k /= k.sum()
    return np.convolve(np.pad(a, win // 2, mode="edge"), k, "valid")[:len(a)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--fps", default="30")
    ap.add_argument("--cell", type=int, default=20)
    ap.add_argument("--chars", default="LGK")
    ap.add_argument("--bright", type=float, default=2.3, help="lift the dark video")
    ap.add_argument("--basevis", type=float, default=0.5)
    ap.add_argument("--greentint", type=float, default=0.5)
    ap.add_argument("--scene", type=float, default=0.35, help="subtle LGK on the rest of the scene")
    ap.add_argument("--glow", type=float, default=1.1)
    ap.add_argument("--glowradius", type=float, default=15.0)
    ap.add_argument("--buildsecs", type=float, default=2.0, help="time for the building to grow in")
    ap.add_argument("--basey", type=float, default=0.72, help="skyline horizon (fraction of H) where the base sits")
    ap.add_argument("--wscale", type=float, default=0.60, help="building width scale (slimmer = more distant)")
    args = ap.parse_args()

    fps = eval(args.fps) if "/" in str(args.fps) else float(args.fps)
    cell = args.cell
    W = (args.width // cell) * cell
    H = (args.height // cell) * cell
    gw, gh = W // cell, H // cell
    atlas = build_atlas(cell, cell + 2, args.chars)
    N = len(args.chars)
    RF = 512
    rng = np.random.default_rng(7)
    randfield = rng.integers(0, N, size=(RF, gw))

    # ---- pass A: track the spire anchor across the clip ----
    cxs, tys = [], []
    last = (args.width * 0.37, args.height * 0.44)
    for f in read_frames(args.input, args.width, args.height):
        a = detect_anchor(f, args.height)
        if a is None:
            a = last
        last = a
        cxs.append(a[0]); tys.append(a[1])
    cxs = smooth(cxs, 13); tys = smooth(tys, 13)
    print(f"tracked {len(cxs)} frames; spire ~x{cxs.mean():.0f} y{tys.mean():.0f}", file=sys.stderr)

    rows = np.arange(gh)[:, None].astype(np.float32)
    cols = np.arange(gw)[None, :].astype(np.float32)

    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", args.output],
        stdin=subprocess.PIPE)

    # ---- pass B: render ----
    n = 0
    for frame in read_frames(args.input, args.width, args.height):
        frame = frame[:H, :W]
        t = n / fps
        lum = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        cellL = lum.reshape(gh, cell, gw, cell).mean(axis=(1, 3)) / 255.0
        vboost = np.clip(np.power(np.clip((cellL - 0.05) / 0.95, 0, 1), 0.5) * 3.0, 0, 1)

        # scene gets a faint green LGK texture so the world still reads as code
        I = args.scene * vboost
        C = np.broadcast_to(GREEN, (gh, gw, 3)).copy()

        # ---- the constructed Empire State Building ----
        center = cxs[n] / cell                      # building center column (cells)
        tip = tys[n] / cell                         # spire tip row (cells)
        base_row = args.basey * H / cell            # skyline horizon row (cells)
        BH = max(base_row - tip, 6.0)               # building sits tip -> skyline (behind freeway)
        yn = (rows - tip) / BH                       # 0 at tip, 1 at base (per grid row)
        hw = (np.interp(yn[:, 0], PROF_Y, PROF_W) * args.wscale)[:, None]   # half-width (cells)
        inside = (yn >= 0) & (yn <= 1) & (np.abs(cols + 0.5 - center) <= hw)

        front = np.clip(t / args.buildsecs, 0, 1)    # build-out grows downward
        grown = inside & (yn <= front)
        shimmer = 0.80 + 0.20 * np.sin(2 * np.pi * (t * 1.4) - yn * 6 + cols * 0.25)
        edge = np.clip(1 - np.abs(yn - front) / 0.05, 0, 1) * (front < 1)   # bright growing front

        Ib = np.where(grown, np.clip(0.95 * shimmer + 0.6 * edge, 0, 1.3), 0.0)
        # Knicks bands exactly like the photo: BLUE antenna+spire, ORANGE crown,
        # BLUE main shaft + base
        orange_band = (yn >= 0.40) & (yn < 0.59)
        shaft_band = (yn >= 0.59)
        shaft_t = np.clip((yn - 0.59) / (1 - 0.59), 0, 1)                  # 0 at shaft top, 1 at base
        shaft_col = SHAFT_BLUE * (1 - 0.6 * shaft_t)[..., None]            # darken/fade toward the bottom
        Cb = np.where(shaft_band[..., None], shaft_col, SPIRE_BLUE)        # deep blue shaft, bright spire
        Cb = np.where(orange_band[..., None], KNICKS_ORANGE, Cb)           # amber crown
        Cb = Cb * (1 - edge[..., None]) + EDGE * edge[..., None]

        # building overrides the scene where it is drawn
        use_b = grown & (Ib > I)
        I = np.where(use_b, Ib, I)
        C = np.where(use_b[..., None], Cb, C)

        # ---- render LGK glyphs from the combined intensity/color grid ----
        scroll = int(t * 6) % RF
        ridx = (np.arange(gh)[:, None] + scroll) % RF
        didx = randfield[ridx, np.arange(gw)[None, :]]
        mask = atlas[didx].transpose(0, 2, 1, 3).reshape(H, W)
        colI = (C * I[..., None])
        colI = np.repeat(np.repeat(colI, cell, 0), cell, 1)
        char = colI * mask[..., None]

        # base video, lifted + green-graded so it stays visible underneath
        bright = np.clip(frame * args.bright + 6, 0, 255)
        gt = args.greentint
        graded = bright * (np.float32([1, 1, 1]) * (1 - gt) + np.float32([0.45, 1.15, 0.6]) * gt)
        base = np.clip(graded, 0, 255) * args.basevis

        cimg = Image.fromarray(np.clip(char, 0, 255).astype(np.uint8))
        glow = np.asarray(cimg.filter(ImageFilter.GaussianBlur(args.glowradius)), np.float32) * args.glow
        out = np.clip(base + char + glow, 0, 255).astype(np.uint8)
        writer.stdin.write(out.tobytes())
        n += 1

    writer.stdin.close(); writer.wait()
    print(f"done: {n} frames -> {args.output} ({W}x{H})", file=sys.stderr)


if __name__ == "__main__":
    main()
