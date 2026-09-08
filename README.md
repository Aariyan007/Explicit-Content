# Shirt-Off Detector

A webcam meme project. Tracks your torso with pose landmarks, watches for a spike in visible skin, and throws up a big "AHH HOT MAN" style overlay when you take your shirt off.

## How it works

1. `mediapipe` pose detection finds your shoulders and hips → defines a torso bounding box.
2. Inside that box, an HSV skin-color mask estimates what % of pixels look like skin.
3. You calibrate a baseline while your shirt is ON.
4. When the skin ratio jumps past baseline by a set threshold, a random funny line flashes on screen for a couple seconds.

## Requirements

- Python 3.9–3.12 (mediapipe does **not** support 3.13/3.14 — see Known Issues)
- `opencv-python`
- `mediapipe`
- `numpy`

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install opencv-python numpy mediapipe==0.10.21
```

`mediapipe==0.10.21` is pinned deliberately — versions 0.10.31+ removed the legacy `mp.solutions` API this script depends on.

## Usage

```bash
python shirt_off_detector.py
```

**Controls:**
- `c` — calibrate baseline (do this with your shirt ON, before removing it)
- `q` — quit

## Tuning

| Variable | What it does | Default |
|---|---|---|
| `SKIN_TRIGGER_DELTA` | How much the skin ratio must jump to count as "shirt off". Lower = more sensitive (more false positives from bare arms/neck). Raise if it's not triggering. | `0.25` |
| `TEXT_DURATION` | How long the overlay stays on screen, in seconds. | `2.0` |
| HSV `lower` / `upper` in `skin_ratio()` | Skin color range. Tuned for lighter skin tones by default — widen this range if it's not detecting your skin tone. | `[0,30,60]` – `[25,180,255]` |

## Known issues

- **Python 3.13/3.14 not supported** — mediapipe only ships wheels for 3.9–3.12. Use `python3.12 -m venv venv` when setting up.
- **Lighting matters a lot** — HSV skin detection is unreliable in dim or heavily tinted lighting. Good, even lighting = fewer false triggers.
- **Bare arms/neck can false-trigger** — since the ROI is the whole torso box (shoulders to hips), sleeveless shirts or wide necklines can push the skin ratio up even with a shirt on. Recalibrate baseline if this happens.

## Roadmap ideas

- Face-detection gate to reduce false positives
- Swap mediapipe for OpenCV's built-in upper-body Haar cascade (drops the mediapipe version dependency entirely)
