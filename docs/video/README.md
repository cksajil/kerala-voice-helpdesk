# Demo video (LinkedIn, 26 Nov 2026)

`build.py` makes the 1920×1080 demo video (v4, about 150 s): an opening clip, picture-led cards in the
style of Lee LeFever's *The Art of Explanation* (`cards2.py`), the owner's phone screen recording in a
split screen whose right panel reveals label → question → answer → source as the example plays, and
a synthesised background track (no copyright), ducked under the demo audio.

```bash
pip install pillow numpy qrcode            # Pillow needs libraqm for Malayalam shaping; ffmpeg on PATH
cd docs/video
npm pack @fontsource/noto-sans-malayalam @fontsource/inter @fontsource/noto-emoji   # OFL fonts
for t in *.tgz; do tar xzf "$t"; done      # then rename the package dirs to match build.py / cards2.py:
# fonts/fontsource-noto-sans-malayalam-5.3.0, fonts/fontsource-inter-5.3.0, fonts/noto-emoji
python3 build.py                                              # rough cut with placeholders
python3 build.py --demo demo_small.mp4 --marks marks.json     # with the recording
```

`marks.json` holds the [start, end] seconds of each question in the recording (ask, then answer; the
wait in between is cut). The recording and the rendered videos are not in the repo. `icons.json` maps
icon names to Noto Emoji subset files. Malayalam text drops "?" and "," because the Malayalam font
subset has no Latin punctuation, and "₹" is written "Rs" for the same reason.
