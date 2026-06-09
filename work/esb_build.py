#!/usr/bin/env python3
"""Construct the Empire State Building out of LGK characters, anchored to the real
spire visible in the footage, extending the full building downward over the video.
Knicks blue/orange bands, glow, and a top-to-bottom build-out animation."""
import argparse, subprocess, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import os


def find_coeffs(dst, src):
    """Perspective coefficients for PIL Image.transform that place the four `src`
    corners (in the source image) at the four `dst` corners (in the output frame)."""
    A = []
    for (xd, yd), (xs, ys) in zip(dst, src):
        A.append([xd, yd, 1, 0, 0, 0, -xs * xd, -xs * yd])
        A.append([0, 0, 0, xd, yd, 1, -ys * xd, -ys * yd])
    res = np.linalg.solve(np.array(A, np.float64),
                          np.array(src, np.float64).reshape(8))
    return res
FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "CourierPrime-Regular.ttf")
SPIRE_BLUE = np.array([95, 165, 255], np.float32)   # bright glowing spire
SHAFT_BLUE = np.array([48, 92, 245], np.float32)    # deep royal main shaft (matches photo)
KNICKS_ORANGE = np.array([255, 140, 40], np.float32)  # amber crown
GREEN = np.array([30, 200, 70], np.float32)
GREEN_HEAD = np.array([205, 255, 215], np.float32)   # white-hot leading glyph of a rain stream
ELECTRIC = np.array([90, 225, 255], np.float32)      # electric cyan spark
EDGE = np.array([225, 240, 255], np.float32)   # bright construction front

# Empire State Building half-width profile (yn: 0=spire tip .. 1=base), in cells.
# Thin spire, a SHORT stepped crown, then a tall CONSTANT-width rectangular shaft
# (the rectangle is ~75% of the height so it clearly reads as a rectangle).
PROF_Y = np.array([0.00, 0.10, 0.13, 0.17, 0.21, 0.25, 1.00])
PROF_W = np.array([0.24, 0.24, 1.20, 2.60, 3.80, 4.40, 4.40])


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


def make_text_layer(W, size):
    """Pre-render 'LET'S GO KNICKS' centered: orange 'LET'S GO', blue 'KNICKS'."""
    fnt = ImageFont.truetype(FONT_PATH, size)
    segs = [("LET'S GO ", KNICKS_ORANGE), ("KNICKS", SPIRE_BLUE)]
    meas = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    widths = [meas.textlength(s, font=fnt) for s, _ in segs]
    total = sum(widths)
    asc, desc = fnt.getmetrics()
    Himg = asc + desc + 24
    img = Image.new("RGB", (W, Himg), (0, 0, 0))
    d = ImageDraw.Draw(img)
    x = (W - total) / 2
    for (s, col), w in zip(segs, widths):
        d.text((x, 12), s, font=fnt, fill=tuple(int(c) for c in col))
        x += w
    return np.asarray(img, np.float32)


LOGO_ORANGE = np.array([255, 120, 12], np.float32)   # vivid saturated orange (high contrast)
LOGO_BLUE = np.array([22, 110, 255], np.float32)      # electric royal blue
LOGO_GRAY = np.array([195, 205, 222], np.float32)     # bright silver accents


def make_logo_layer(canvas_W, logo_path, target_w, cell, chars, seed=3):
    """Render the Knicks logo as LGK glyphs: orange / blue / gray by region.
    `seed` varies only the glyph CHARACTERS (not the colours), so different seeds
    are flicker frames of the same crest. Returns (emblem_rgb, outW, outH)."""
    im = Image.open(logo_path).convert("RGBA")
    arr = np.asarray(im, np.float32)
    rgb, a = arr[..., :3], arr[..., 3]
    present = (a > 40) & (rgb.min(axis=2) < 232)          # opaque and not white background
    ys, xs = np.where(present)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    rgb = rgb[y0:y1 + 1, x0:x1 + 1] * present[y0:y1 + 1, x0:x1 + 1, None]
    pres = present[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    h, w = pres.shape
    cols = max(1, target_w // cell)
    rows = max(1, round(cols * h / w))
    rs_rgb = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((cols, rows)), np.float32)
    rs_pre = np.asarray(Image.fromarray((pres * 255).astype(np.uint8)).resize((cols, rows)), np.float32) / 255
    rs_rgb = rs_rgb / np.clip(rs_pre[..., None], 0.05, None)   # recover true hue under coverage

    outW, outH = cols * cell, rows * cell
    img = Image.new("RGB", (outW, outH), (0, 0, 0))
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(FONT_PATH, cell + 1)
    rng = np.random.default_rng(seed)
    for r in range(rows):
        for c in range(cols):
            if rs_pre[r, c] < 0.38:        # tighter cut -> cleaner shapes, more black contrast
                continue
            R, G, B = rs_rgb[r, c]
            if R > B + 18:
                col = LOGO_ORANGE
            elif B > R + 18:
                col = LOGO_BLUE
            else:
                col = LOGO_GRAY
            ch = chars[int(rng.integers(0, len(chars)))]
            d.text((c * cell, r * cell), ch, font=fnt, fill=tuple(int(x) for x in col))

    return np.asarray(img, np.float32), outW, outH


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
    ap.add_argument("--rain", type=float, default=0.85, help="strength of the falling digital rain")
    ap.add_argument("--raindensity", type=float, default=0.42, help="fraction of columns that rain (lower = sparser)")
    ap.add_argument("--flicker", type=float, default=12.0, help="per-cell digit flicker rate (changes/sec)")
    ap.add_argument("--lights", type=float, default=0.6, help="brightness of city/car lights rendered as binary ascii")
    ap.add_argument("--lightglow", type=float, default=1.7, help="glow/bloom on the city + car lights")
    ap.add_argument("--lightthr", type=float, default=0.20, help="luminance threshold for what counts as a light")
    ap.add_argument("--noiseamt", type=float, default=0.38, help="perlin-noise dither on the ascii thresholds (organic edges)")
    ap.add_argument("--spark", type=float, default=0.0045, help="density of flickering electric-cyan sparks")
    ap.add_argument("--glow", type=float, default=1.1)
    ap.add_argument("--glowradius", type=float, default=15.0)
    ap.add_argument("--buildsecs", type=float, default=2.0, help="time for the building to grow in")
    ap.add_argument("--basey", type=float, default=0.72, help="skyline horizon (fraction of H) where the base sits")
    ap.add_argument("--wscale", type=float, default=0.82, help="building width scale (slimmer = more distant)")
    ap.add_argument("--esbdepth", type=float, default=0.9, help="depth of the building's 3D side face (cells, x half-width)")
    ap.add_argument("--esbshear", type=float, default=0.15, help="how much the 3D side face shears up as it recedes")
    ap.add_argument("--logo", default=os.path.join(os.path.dirname(__file__), "knicks_logo.png"))
    ap.add_argument("--logow", type=int, default=470, help="Knicks crest on-screen width in px (near edge)")
    ap.add_argument("--logocell", type=int, default=10, help="character size in the logo")
    ap.add_argument("--logox", type=float, default=0.65, help="crest center x (fraction of W)")
    ap.add_argument("--logoy", type=float, default=0.31, help="crest center y (fraction of H)")
    ap.add_argument("--logoglow", type=float, default=2.2, help="glow/bloom strength on the logo ascii")
    # ---- the crest sits small + distant, SLANTED in perspective (receding toward
    #      center-right), its ascii chars flickering, the whole sign fading in and out ----
    ap.add_argument("--tilt", type=float, default=0.78, help="vertical foreshortening of the crest")
    ap.add_argument("--recede", type=float, default=0.58, help="far (right) edge scale — perspective recede toward center-right")
    ap.add_argument("--keystone", type=float, default=0.92, help="far (top) edge width fraction")
    ap.add_argument("--rot", type=float, default=5.0, help="in-plane slant of the crest, degrees")
    ap.add_argument("--logovars", type=int, default=8, help="number of glyph flicker frames for the crest")
    ap.add_argument("--logoflicker", type=float, default=11.0, help="crest ascii flicker rate (changes/sec)")
    ap.add_argument("--fadeperiod", type=float, default=2.6, help="seconds for one fade in+out cycle of the whole logo")
    ap.add_argument("--aura", type=float, default=0.55, help="strength of the soft cyber glow around the crest")
    ap.add_argument("--rayglow", type=float, default=0.4, help="strength of the radiance/spotlight from behind the buildings")
    ap.add_argument("--originx", type=float, default=0.52, help="hidden glow origin x, behind the skyline (fraction of W)")
    ap.add_argument("--originy", type=float, default=0.66, help="hidden glow origin y, in the skyline behind the buildings (fraction of H)")
    ap.add_argument("--rayreach", type=float, default=0.55, help="how far the radiance reaches up the sky (fraction of H)")
    ap.add_argument("--conew", type=float, default=0.62, help="angular width of the radiance fan, radians")
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
    bin_atlas = build_atlas(cell, cell + 2, "01")        # 0/1 glyphs for the city + car lights
    randbin = rng.integers(0, 2, size=(RF, gw))
    # smooth value-noise grid (perlin-like): a low-res random field bicubically
    # upsampled, scrolled over time, used to organically dither the ascii thresholds
    nlr = np.random.default_rng(33).random((max(2, RF // 6), max(2, gw // 4)))
    noise_field = np.asarray(Image.fromarray((nlr * 255).astype(np.uint8)).resize(
        (gw, RF), Image.BICUBIC), np.float32) / 255.0

    # per-column rain: each column falls at its own speed, so cells both fall and
    # flicker (the field scrolls past them), like the reference's binary rain
    rrng = np.random.default_rng(11)
    rspeed = rrng.uniform(7.0, 18.0, gw).astype(np.float32)            # rows/sec
    rtail = rrng.uniform(5.0, 14.0, gw).astype(np.float32)             # streak length
    rperiod = (gh + rtail + rrng.uniform(40.0, 160.0, gw)).astype(np.float32)   # long gaps -> sparse
    rphase = (rrng.uniform(0, 1, gw) * rperiod).astype(np.float32)
    ractive = (rrng.random(gw) < args.raindensity).astype(np.float32)[None, :]  # only some columns rain
    rowsg = np.arange(gh)[:, None].astype(np.float32)
    colidx = np.arange(gw)[None, :]

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
    mean_cx, mean_ty = float(cxs.mean()), float(tys.mean())   # crest/glow lock onto the tracked building's drift
    print(f"tracked {len(cxs)} frames; spire ~x{cxs.mean():.0f} y{tys.mean():.0f}", file=sys.stderr)

    rows = np.arange(gh)[:, None].astype(np.float32)
    cols = np.arange(gw)[None, :].astype(np.float32)

    emb0, eW, eH = make_logo_layer(W, args.logo, args.logow, args.logocell, args.chars, seed=10)

    # ---- the crest sits small + distant, SLANTED in perspective: a near (left) edge
    #      with a receding far (right) edge, so the sign stretches toward center-right.
    #      Its ascii chars flicker (cycling glyph variants) and the whole sign fades
    #      in and out. A soft, dimmed cyber spotlight wells up from behind the skyline. ----
    CYBER = np.array([90, 205, 255], np.float32)          # electric cyan-blue cyber glow
    ecx, ecy = args.logox * W, args.logoy * H             # crest center on screen
    Wp = float(args.logow)
    Hp = Wp * (eH / eW) * args.tilt
    rf = args.recede                                      # far (right) edge recedes into the distance
    rot = math.radians(args.rot)
    cr, srot = math.cos(rot), math.sin(rot)
    # local corners (TL, TR, BR, BL): left edge full height, right edge shortened, slanted
    local = [(-Wp / 2, -Hp / 2), (Wp / 2, -Hp / 2 * rf), (Wp / 2, Hp / 2 * rf), (-Wp / 2, Hp / 2)]
    emb_quad = [(ecx + x * cr - y * srot, ecy + x * srot + y * cr) for x, y in local]
    src_quad = [(0, 0), (eW, 0), (eW, eH), (0, eH)]
    coeffs = find_coeffs(emb_quad, src_quad)

    # soft oval mask in the SAME perspective so every flicker frame's glow stays organic
    GS = 256
    gyy, gxx = np.mgrid[0:GS, 0:GS]
    grr = np.sqrt(((gxx - GS / 2) / (GS / 2)) ** 2 + ((gyy - GS / 2) / (GS / 2)) ** 2).astype(np.float32)
    ofill = np.clip((1.06 - grr) / 0.16, 0, 1)
    s = 1.32
    omask_quad = [(ecx + (x - ecx) * s, ecy + (y - ecy) * s) for x, y in emb_quad]
    ocoeffs = find_coeffs(omask_quad, [(0, 0), (GS, 0), (GS, GS), (0, GS)])
    omask = np.asarray(Image.fromarray((ofill * 255).astype(np.uint8)).transform(
        (W, H), Image.PERSPECTIVE, ocoeffs, resample=Image.BILINEAR), np.float32) / 255.0

    def warp_crest(emb_arr):
        im = Image.fromarray(np.clip(emb_arr, 0, 255).astype(np.uint8))
        proj = np.asarray(im.transform((W, H), Image.PERSPECTIVE, coeffs,
                                       resample=Image.BILINEAR), np.float32) * omask[..., None]
        pi = Image.fromarray(np.clip(proj, 0, 255).astype(np.uint8))
        near = (np.asarray(pi.filter(ImageFilter.GaussianBlur(4)), np.float32) * 1.5
                + np.asarray(pi.filter(ImageFilter.GaussianBlur(11)), np.float32) * 1.0
                + np.asarray(pi.filter(ImageFilter.GaussianBlur(26)), np.float32) * 0.6) * args.logoglow
        return proj, proj * 1.3 + near

    # K flicker frames: identical colour/shape, different glyph CHARACTERS
    K = max(1, args.logovars)
    crest_variants = []
    proj0 = None
    for k in range(K):
        embk = emb0 if k == 0 else make_logo_layer(W, args.logo, args.logow, args.logocell, args.chars, seed=10 + k)[0]
        projk, crestk = warp_crest(embk)
        if k == 0:
            proj0 = projk
        crest_variants.append(crestk)

    # cool cyber halo derived from one variant (blur hides the flicker)
    p0 = Image.fromarray(np.clip(proj0, 0, 255).astype(np.uint8))
    wide = np.asarray(p0.filter(ImageFilter.GaussianBlur(40)), np.float32)
    huge = np.asarray(p0.filter(ImageFilter.GaussianBlur(80)), np.float32)
    widel = 0.299 * wide[..., 0] + 0.587 * wide[..., 1] + 0.114 * wide[..., 2]
    hugel = 0.299 * huge[..., 0] + 0.587 * huge[..., 1] + 0.114 * huge[..., 2]
    aura_layer = (widel[..., None] * 3.0 + hugel[..., None] * 4.5) * (CYBER / 255.0) * args.aura
    aura_layer = aura_layer * (1 - 0.55 * omask[..., None])   # halo AROUND the crest; let its colour read

    # radiance welling up from BEHIND the skyline — a soft, sourceless fan whose origin
    # is hidden below/behind the buildings. Broad cone + faint god-ray striations,
    # masked to emerge only above the horizon so no point source is ever seen.
    Yg, Xg = np.mgrid[0:H, 0:W]
    Xf, Yf = Xg.astype(np.float32), Yg.astype(np.float32)
    ox, oy = args.originx * W, args.originy * H
    axx, ayy = ecx - ox, ecy - oy
    La = math.hypot(axx, ayy)
    axu, ayu = axx / La, ayy / La                          # axis: hidden origin -> crest
    vx, vy = Xf - ox, Yf - oy
    dist = np.sqrt(vx * vx + vy * vy)
    dotp = (vx * axu + vy * ayu) / np.maximum(dist, 1.0)   # cos(angle from axis)
    ang = np.arccos(np.clip(dotp, -1.0, 1.0))
    cone = np.exp(-(ang / args.conew) ** 2)                # broad upward fan
    rad = np.exp(-dist / (args.rayreach * H))              # fades up into the sky
    rays = np.clip(1.0 + 0.18 * np.sin(ang * 11.0), 0, 2)  # faint volumetric striations
    horizon = args.basey * H
    emerge = np.clip((horizon + 0.05 * H - Yf) / (0.12 * H), 0, 1)   # fade toward/below the skyline
    originhide = np.clip(dist / (0.14 * H), 0, 1)          # no bright point at the source — origin unknown
    along = vx * axu + vy * ayu                            # distance travelled along the beam from origin
    tocrest = 0.40 + 0.60 * np.clip(along / La, 0, 1)      # the beam strengthens toward the logo
    endcap = np.clip((La - along) / (0.09 * H) + 1.0, 0, 1)  # ...and ENDS at the logo (fades just beyond)
    radiance = rad * cone * rays * np.clip(dotp, 0, 1) * emerge * originhide * tocrest * endcap
    ray_layer = radiance[..., None] * CYBER * args.rayglow

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
        # ---- matrix RAIN: per-column falling streams of LGK with white-hot heads, so
        #      the individual cells fall and flicker like the reference's binary rain ----
        head = np.mod(rspeed * t + rphase, rperiod)            # (gw,) leading-glyph row per column
        dist_r = head[None, :] - rowsg                         # (gh,gw) rows trailing the head
        rain_b = np.clip(1.0 - dist_r / rtail[None, :], 0, 1) * (dist_r >= 0) * ractive
        headness = np.clip(1.0 - np.abs(dist_r), 0, 1) * ractive   # ~1 only at the leading glyph

        # perlin-like noise grid: organically dither the threshold so the ascii field
        # has feathered, clustered edges instead of a hard cutoff
        nz = noise_field[(np.arange(gh)[:, None] + int(t * 6)) % RF, colidx]   # (gh,gw) smooth, animated
        scene_gate = np.clip((vboost * 1.6 + (nz - 0.5) * args.noiseamt - 0.12) / 0.28, 0, 1)
        I = np.maximum(args.scene * vboost * scene_gate, args.rain * rain_b)
        I = np.maximum(I, headness * args.rain)
        C = np.broadcast_to(GREEN, (gh, gw, 3)).copy()
        C = C * (1 - headness[..., None]) + GREEN_HEAD * headness[..., None]

        # sparse Knicks-colour tracers flickering through the rain
        htick = int(t * 12)
        hsel = np.random.default_rng(2000 + htick).random((gh, gw))
        otr = (hsel < 0.004) & (rain_b > 0.04)
        btr = (hsel > 0.996) & (rain_b > 0.04)
        C = np.where(otr[..., None], KNICKS_ORANGE, C)
        C = np.where(btr[..., None], SPIRE_BLUE, C)
        I = np.where(otr | btr, np.maximum(I, 0.9), I)

        # electric crackle: bright cyan sparks flickering fast through the field
        esel = np.random.default_rng(5000 + int(t * 22)).random((gh, gw))
        spark = (esel < args.spark) & (I > 0.03)
        C = np.where(spark[..., None], ELECTRIC, C)
        I = np.where(spark, np.maximum(I, 1.15), I)

        # ---- the Empire State Building as a 3D extruded model: a lit FRONT face + a
        #      shaded, sheared right SIDE face with a bright vertical corner edge ----
        center = cxs[n] / cell                      # building center column (cells)
        tip = tys[n] / cell                         # spire tip row (cells)
        base_row = args.basey * H / cell            # skyline horizon row (cells)
        BH = max(base_row - tip, 6.0)               # building sits tip -> skyline (behind freeway)
        front = np.clip(t / args.buildsecs, 0, 1)    # build-out grows downward

        ub = np.broadcast_to(cols + 0.5 - center, (gh, gw))   # signed offset from the axis
        ynF = (rows - tip) / BH                               # (gh,1) front-face row coord
        hwF = (np.interp(np.clip(ynF[:, 0], 0, 1), PROF_Y, PROF_W) * args.wscale)[:, None]
        rowvalid = (ynF >= 0) & (ynF <= 1) & (ynF <= front)
        front_face = rowvalid & (np.abs(ub) <= hwF)
        s = ub - hwF                                          # depth into the right side (cells)
        depth = np.maximum(1.0, args.esbdepth * hwF)
        ynS = (rows + s * args.esbshear - tip) / BH           # side recedes up-right
        side_face = (ub > hwF) & (s <= depth) & (ynS >= 0) & (ynS <= 1) & (ynS <= front)
        face = front_face | side_face
        yn = np.where(side_face, ynS, ynF)                    # band/shimmer source coord per cell

        # face shading: front lit (brighter left, bright corner), side in shadow
        un = ub / np.maximum(hwF, 0.5)
        shadeF = 0.86 + 0.14 * (0.5 - 0.5 * un)
        shadeF = np.where(np.abs(un) > 0.82, np.maximum(shadeF, 1.18), shadeF)   # crisp rectangle edges
        shadeS = 0.5 - 0.18 * np.clip(s / depth, 0, 1)
        shade = np.where(side_face, shadeS, shadeF)

        shimmer = 0.80 + 0.20 * np.sin(2 * np.pi * (t * 1.4) - yn * 6 + cols * 0.25)
        edge = np.clip(1 - np.abs(yn - front) / 0.05, 0, 1) * (front < 1)   # bright growing front
        Ib = np.where(face, np.clip(0.95 * shimmer + 0.6 * edge, 0, 1.3) * shade, 0.0)
        # Knicks bands: BLUE antenna+spire, ORANGE crown, BLUE main shaft + base
        orange_band = (yn >= 0.40) & (yn < 0.59)
        shaft_band = (yn >= 0.59)
        shaft_t = np.clip((yn - 0.59) / (1 - 0.59), 0, 1)
        shaft_col = SHAFT_BLUE * (1 - 0.6 * shaft_t)[..., None]
        Cb = np.where(shaft_band[..., None], shaft_col, SPIRE_BLUE)
        Cb = np.where(orange_band[..., None], KNICKS_ORANGE, Cb)
        Cb = Cb * (1 - edge[..., None]) + EDGE * edge[..., None]

        # ---- payoff: completion flash + heartbeat once the building is fully built ----
        if t >= args.buildsecs:
            te = t - args.buildsecs
            flash = np.exp(-te / 0.35)                                     # bright surge on completion
            beat = 1 + 0.7 * flash + 0.12 * np.sin(2 * np.pi * 1.1 * te)   # then a steady pulse
            Ib = Ib * beat
            Cb = Cb * (1 - 0.5 * flash) + EDGE * (0.5 * flash)             # whiten at the peak

        # building overrides the scene where it is drawn
        use_b = face & (Ib > I)
        I = np.where(use_b, Ib, I)
        C = np.where(use_b[..., None], Cb, C)

        # ---- render LGK glyphs; per-column scroll makes the cells fall and flicker ----
        col_scroll = ((rspeed + args.flicker) * t).astype(np.int32)   # per-column fall + fast flicker
        ridx = (np.arange(gh)[:, None] + col_scroll[None, :]) % RF
        didx = randfield[ridx, colidx]
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
        ebuzz = 1.0 + 0.06 * math.sin(2 * math.pi * t * 37) + 0.04 * math.sin(2 * math.pi * t * 113)  # electric buzz
        out_f = base + (char + glow) * ebuzz

        # ---- city + car lights: faint, glowing BINARY ascii kept in their native colour ----
        cRf = frame[..., 0].reshape(gh, cell, gw, cell).mean((1, 3))
        cGf = frame[..., 1].reshape(gh, cell, gw, cell).mean((1, 3))
        cBf = frame[..., 2].reshape(gh, cell, gw, cell).mean((1, 3))
        litw = np.clip((cellL - args.lightthr + (nz - 0.5) * args.noiseamt * 0.5) / 0.34, 0, 1) ** 1.3  # noise-dithered
        nat = np.stack([cRf, cGf, cBf], -1)
        nat = nat / (nat.max(-1, keepdims=True) + 1e-3)                    # preserve native hue at full saturation
        bscroll = int(t * args.flicker * 0.7)
        bdidx = randbin[(np.arange(gh)[:, None] + bscroll) % RF, colidx]   # flickering 0/1
        bmask = bin_atlas[bdidx].transpose(0, 2, 1, 3).reshape(H, W)
        lcol = np.repeat(np.repeat(nat * (litw * args.lights * 255.0)[..., None], cell, 0), cell, 1)
        lchar = lcol * bmask[..., None]
        lci = Image.fromarray(np.clip(lchar, 0, 255).astype(np.uint8))
        lglow = np.asarray(lci.filter(ImageFilter.GaussianBlur(args.glowradius * 0.7)), np.float32) * args.lightglow
        out_f += lchar * 0.55 + lglow                                      # faded glyphs + a soft native glow

        # ---- once the building is built, the cyber radiance wells up from behind the
        #      skyline and the Knicks crest resolves within it, glowing majestically.
        #      The crest + glow are LOCKED to the tracked building's drift, so they sit
        #      in the moving scene instead of floating in fixed screen space. ----
        sigstart = args.buildsecs + 0.30
        te = t - sigstart
        if te >= 0:
            on = float(np.clip(te / 0.60, 0, 1)); on = on * on * (3 - 2 * on)  # spotlight ramps up once
            swell = 1 + 0.20 * float(np.exp(-te / 0.5))
            breathe = 1 + 0.05 * float(np.sin(2 * np.pi * 0.40 * te))          # slow elegant breathing
            dx = int(round(cxs[n] - mean_cx)); dy = int(round(tys[n] - mean_ty))   # track the scene's motion
            occl = 1.0 - 0.85 * np.clip(lum / 120.0, 0, 1)                     # the city occludes the glow behind it
            out_f += np.roll(ray_layer, (dy, dx), (0, 1)) * (occl[..., None] * (on * swell * breathe))

            # the whole Knicks sign fades IN and OUT in a slow cycle; its ascii flickers
            fade = on * (0.5 - 0.5 * float(np.cos(2 * np.pi * te / args.fadeperiod)))
            if fade > 0.01:
                vk = (int(te * args.logoflicker) * 5) % len(crest_variants)    # scrambled glyph flicker
                out_f += np.roll(aura_layer, (dy, dx), (0, 1)) * (fade * breathe)
                out_f += np.roll(crest_variants[vk], (dy, dx), (0, 1)) * (fade * breathe)

        out = np.clip(out_f, 0, 255).astype(np.uint8)
        writer.stdin.write(out.tobytes())
        n += 1

    writer.stdin.close(); writer.wait()
    print(f"done: {n} frames -> {args.output} ({W}x{H})", file=sys.stderr)


if __name__ == "__main__":
    main()
