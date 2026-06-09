# LGK — live p5.js version

A real-time browser port of the offline renderer (`../work/esb_build.py`). Same look —
LGK ascii matrix rain, the LGK-built Empire State Building, city/car lights as glowing
native-colour binary, and the small slanted Knicks crest that fades in/out at the end of a
dimmed spotlight — but it runs live off the video clock instead of a pre-rendered mp4, with
sliders to tweak everything in real time (no 20s re-render per change).

## Run
Needs a local server (the sketch reads the video/font/logo/JSON, which `file://` blocks):

```
cd ..            # repo root
python3 -m http.server 8123
# then open http://localhost:8123/p5/
```

Click once if the video doesn't autoplay (browser autoplay policy).

## How it maps to the python
- `spire_track.json` — the ESB spire path, precomputed once (the only look-ahead pass the
  Python did). Regenerate with the snippet in the commit if `og-video.mp4` changes.
- Per-cell grid (96×54), rain, perlin dither, electric sparks, ESB build, glyph flicker,
  lights, fade/flicker, spotlight — all ported to `sketch.js` and driven by `video.time()`.
- Glows use canvas `filter: blur()`; the crest perspective uses a subdivided textured mesh
  in a WEBGL buffer (warped once per variant, glow baked in flat space).

## Known approximations vs. the offline render (~95% parity)
- Base video grade is a 2-pass tint approximation of the exact linear grade.
- The spotlight's per-pixel occlusion-by-city-brightness is not applied live (static fan).
- Crest glow is baked with additive blur passes rather than the exact triple-blur weights.

The offline `esb_build.py` remains the high-quality reference renderer.
