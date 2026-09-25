# Color Detection

Real-time color tracking with OpenCV. Point a webcam at something yellow, or red, blue, or any RGB value, and it draws a box around every blob of that color.

No machine learning is involved. The whole thing runs on HSV color-space thresholding, which makes it fast enough for any laptop webcam and a good base for things like object following or sorting by color.

## How it works

1. **Convert to HSV.** In RGB, "yellow" changes with lighting. In HSV the *hue* stays fairly stable while saturation and brightness vary, so color is much easier to isolate.
2. **Build a hue window.** The target RGB color is converted to a hue value and a window of ±10 is set around it. Saturation and value must be at least 100, which filters out greys and dark shadows.
3. **Handle red.** OpenCV stores hue on 0–179, and red sits at *both* ends of that range. When the window crosses 0 or 179, it is split into two ranges and the masks are combined, so red is actually detected.
4. **Clean the mask.** A morphological opening removes single-pixel noise.
5. **Find blobs.** External contours are extracted, anything smaller than `--min-area` is ignored, and each remaining blob gets its own bounding box.

## Usage

```bash
pip install -r requirements.txt

python main.py                          # track yellow on webcam 0
python main.py --color red --show-mask  # track red, and show the binary mask
python main.py --color 0,120,255        # any RGB value
python main.py --source clip.mp4        # run on a video file instead
```

| Flag | Default | Description |
|---|---|---|
| `--color` | `yellow` | preset (`red`, `orange`, `yellow`, `green`, `cyan`, `blue`, `purple`) or `R,G,B` |
| `--source` | `0` | webcam index or video path |
| `--min-area` | `500` | smallest blob, in pixels, that counts as a detection |
| `--show-mask` | off | show the thresholded mask in a second window |

Press **q** to quit.

## Files

- `main.py`: capture loop, blob detection and drawing
- `util.py`: RGB → HSV range conversion, including the red hue wrap-around

## Credits

Started from the color-detection tutorial by [Computer Vision Engineer](https://www.youtube.com/@ComputerVisionEngineer). Compared with the original, this version:

- detects red correctly by splitting the hue window at the wrap-around
- boxes each blob separately instead of drawing one box around all matching pixels
- filters noise with morphological opening and a minimum area
- adds color presets, arbitrary RGB input, and video-file input
- exits cleanly when the camera is unavailable

## Stack

Python · OpenCV · NumPy
