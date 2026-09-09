# SAM-3D-Body 3D Alignment Viewer POC

This POC keeps GPU inference and interactive viewing separate:

1. Run SAM-3D-Body on a GPU server and generate two `*.render.npz` files.
2. Align the skeleton sequences with `scripts/render_alignment_side_by_side.py`
   to produce `alignment_report.json`.
3. Export original-view and canonical, DTW-matched mesh buffers with
   `export_assets.py`.
4. Compare both videos first, optionally overlay 3D, then switch to one
   automatically normalized view.

The viewer starts with the two source videos playing side by side. `3D` is a
toggle that overlays the corresponding model on each video without enabling a
free camera. `统一视角` pauses the hidden video decoders, keeps the reference
model's capture orientation fixed, and rotates the user model around the
ground-vertical axis through its calibrated foot center until its horizontal
facing direction matches the reference. Returning to video seeks both sources
once to the current 3D time. In unified mode, two horizontal ground discs stay
at fixed left and right positions. The first frame is treated as the
planted-foot calibration pose: mesh topology separates the two lower-leg
components, both sole contact centers are rigidly leveled, and their midpoint
is placed at the center of the disc. Those calibrated sole vertices are tracked
through the sequence. Each frame receives one shared vertical correction that
keeps the lower support foot on the ground during a squat; horizontal weight
shifts and the other foot's relative lift remain intact because neither foot is
independently snapped to the disc. A horizontal drag applies the same yaw angle
to both models around their own calibrated centers while the discs remain
still. The camera never moves, so the models stay side by side and the operation
is body rotation rather than camera orbiting. Double-click restores the matched
view.

3D assets and the WebGL context are loaded only after the toggle is enabled.
Each colored wireframe is rendered after an invisible solid depth prepass, so
far-side limbs and back-facing lines are occluded by the nearest body surface
without changing the visible wireframe color or opacity.
Turning the toggle off disposes the mesh buffers, materials, renderer, and WebGL
context. The viewer also releases video decoder resources when the page exits.

## Prepare browser assets

Use the module's Linux conda environment, which already provides NumPy:

```bash
conda activate sam_3d_body
python export_assets.py \
  --user-render-npz /path/to/user.render.npz \
  --reference-render-npz /path/to/reference.render.npz \
  --alignment-json /path/to/alignment_report.json \
  --output-dir data \
  --user-label IMG_3194 \
  --reference-label "Fan Zhendong"
```

Place browser-compatible copies of the two source videos beside the exported
buffers. The viewer intentionally keeps video preparation separate from mesh
export:

```bash
ffmpeg -i /path/to/user.mov -an -c:v libx264 -pix_fmt yuv420p data/user.mp4
ffmpeg -i /path/to/reference.mov -an -c:v libx264 -pix_fmt yuv420p data/reference.mp4
```

Video-overlay buffers retain the original vertices plus every source frame's
SAM-3D-Body camera translation and pinhole intrinsics. The browser projects
those vertices into the video's contained content rectangle, so overlay mode
keeps the exact subject size and position from each source instead of applying
manual scale or offset calibration.

Before export, a symmetric three-frame low-pass filter is applied to source
vertices, keypoints, camera translation, and the derived body basis. The
filter uses `0.2 / 0.6 / 0.2` weights with the center frame unchanged in time,
so it reduces prediction jitter without adding playback latency. The window is
intentionally short to preserve fast racket-sport motion.

Unified-view buffers use the MHR70 hip midpoint as the root, the hip axis and
neck as the body basis, and one stable bone-chain scale per person. The scale is
the sequence median of neck-to-pelvis + neck-to-nose + average left/right
leg-chain length (hip-to-knee + knee-to-ankle). Every vertex is scaled uniformly
by `2.4 / bodyScale`, giving both people the same target skeletal-chain size
while preserving their body proportions. There is no subsequent mesh-height
fitting: crouching stays lower and raising a hand does not shrink the body.
The chain is estimated from source keypoints before temporal smoothing, so
blending bent joints cannot shorten the normalization length.

The manifest's `normalization.method` is `skeletal-chain-v1`;
`targetBodyScale`, `userScaleFactor`, and `referenceScaleFactor` describe the
new scaling. These replace the old visual-height diagnostics; the v1 buffer
layout and viewer loading contract remain compatible. Previously generated
assets retain their old scale and must be re-exported to use this method.

Per-frame basis matrices provide the relative rotation from the user
capture into the reference capture orientation; the reference itself is never
rotated to a third canonical camera. The stable scale prevents model size from
pulsing frame by frame.

## Run locally

```bash
npm ci
python server.py
```

Open <http://127.0.0.1:4173>. The viewer itself does not need a GPU.

## Validate

```bash
npm test
python -m unittest -q
```
