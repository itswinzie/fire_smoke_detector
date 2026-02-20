# 🔥 Fire & Smoke Detection System

**Real-time fire and smoke detection using Vision Language Models (VLM) on NVIDIA Jetson Orin NX.**

Powered by [Ollama](https://ollama.com) and `gemma3:4b`, this system continuously analyses a live webcam feed or video file, triggers multi-modal alerts on detection, and streams annotated video to a web dashboard — all running locally on the Jetson with no cloud dependency.

---

## Features

- **Real-time detection** — fire, smoke, or both, with confidence levels (high / medium / low)
- **MJPEG video stream** — live annotated feed at up to 30 fps via browser
- **Multi-modal alerts** — red flashing border on video, popup notification, audible beep, and log file
- **Smoke enhancement** — CLAHE contrast boost + sharpening applied before VLM inference to improve smoke visibility
- **Dual-thread architecture** — video stream never freezes while VLM is processing
- **Flexible input** — USB webcam (real-time) or any video file (loops automatically)
- **Web dashboard** — detection status, confidence bar, session statistics, alert history

---

## System Requirements

### Hardware
| Component | Requirement |
|---|---|
| Platform | NVIDIA Jetson Orin NX (8 GB or 16 GB) |
| Camera | USB webcam or CSI camera compatible with Jetson |
| Power supply | **19 V / 4 A or higher** — insufficient power causes CPU/GPU throttling |
| Storage | At least 10 GB free for the Ollama model |

> **Warning:** If you see *"System throttled due to Over-current"* in Vivaldi, your power adapter is underpowered. This will significantly degrade detection performance.

### Software
| Component | Requirement |
|---|---|
| JetPack | 6.x or 7.0 (JetPack 5.x is **not** supported) |
| Python | 3.10 or later |
| Ollama | Installed automatically by `run.sh` |
| Browser | Vivaldi, Chrome, or Firefox |

---

## Project Structure

```
fire_smoke_detector/
├── detector.py        — Main backend: VLM inference, MJPEG stream server, logging
├── index.html         — Web dashboard UI
├── run.sh             — One-command setup and launch scrip
└── README.md
```

---

## Quick Start

### Step 1 — Copy all files to your Jetson

Place all project files into a single folder on the Jetson, for example:

```
/home/<user>/fire_smoke_detector/
```

### Step 2 — Run the setup script

```bash
cd ~/fire_smoke_detector
chmod +x run.sh
./run.sh
```

`run.sh` will automatically:
1. Install Ollama (skipped if already installed)
2. Pull the `gemma3:4b` model *(first run: ~5–15 minutes depending on connection speed)*
3. Install Python dependencies (`aiohttp`, `opencv-python`)
4. Launch the detection system

### Step 3 — Select video source

```
Select input source:
  [1] Webcam (real-time)
  [2] Video file

Enter choice (1/2):
```

- **Webcam** — enter the camera index (usually `0`)
- **Video file** — enter the full path, e.g. `/home/user/test_fire.mp4`

### Step 4 — Open the dashboard

```
http://localhost:8080
```

The live annotated video stream appears in the centre panel. Allow a few seconds for the camera to warm up and for the first VLM inference to complete.

---

## Architecture

The system runs two independent threads so the video stream remains smooth regardless of how long VLM inference takes.

```
┌─────────────────────────────────────────────────────────────┐
│  Thread A — Frame Loop                              ~30 fps  │
│  Camera → Resize → Draw overlay → JPEG → /stream           │
└──────────────────────────────┬──────────────────────────────┘
                               │  every FRAME_SKIP frames
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Thread B — VLM Worker                          ~1×/second  │
│  Frame → enhance_for_smoke() → Ollama → update state       │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  HTTP Server                                                 │
│  GET /stream     → MJPEG video stream                       │
│  GET /api/state  → JSON detection state                     │
│  GET /           → Web dashboard (index.html)               │
└─────────────────────────────────────────────────────────────┘
```

### Smoke Enhancement Pipeline

Before each frame is sent to the VLM, it passes through `enhance_for_smoke()`:

```
Raw frame
    │
    ├─► Convert to LAB colour space
    │
    ├─► CLAHE on L channel (clipLimit=3.0)
    │     Boosts local contrast without affecting colour
    │
    ├─► Sharpening filter
    │     Helps the VLM identify soft smoke edges
    │
    └─► Blend: 70% enhanced + 30% original  →  to VLM
```

> Enhancement is applied **only to the frame sent to the VLM**, not to the video displayed in the browser.

---

## Alert Types

When fire or smoke is detected, all of the following trigger simultaneously:

| Alert | Description |
|---|---|
| 🔴 Red border flash | Thick red border pulses around the video frame |
| 📢 Popup notification | Slides in from the top-right corner, auto-dismisses after 8 seconds |
| 🔊 Audible beep | Plays via `aplay` or terminal bell |
| 📋 Log entry | Timestamped record written to `logs/detection_<timestamp>.log` |

Detection types returned by the VLM:

| Type | Meaning |
|---|---|
| `fire` | Orange, red, or yellow flames detected |
| `smoke` | Haze, grey/white plumes, or airborne particles detected |
| `both` | Fire and smoke detected simultaneously |
| `none` | No hazard — scene is clear |

---

## Configuration

All tunable settings are located at the top of `detector.py` under the `CONFIG` section.

### Stream settings

| Setting | Default | Description |
|---|---|---|
| `STREAM_WIDTH` | `1120` | Stream resolution width in pixels |
| `STREAM_HEIGHT` | `630` | Stream resolution height — 16:9, fits Vivaldi fullscreen without scrolling |
| `STREAM_QUALITY` | `72` | JPEG quality for the browser stream (1–100) |
| `STREAM_FPS_CAP` | `30` | Maximum frames per second delivered to the browser |

### VLM inference settings

| Setting | Default | Description |
|---|---|---|
| `FRAME_SKIP` | `30` | Run VLM inference every N frames. At 30 fps → ~1 analysis/second |
| `VLM_WIDTH` | `640` | Frame width sent to VLM. Smaller = faster inference |
| `VLM_HEIGHT` | `360` | Frame height sent to VLM (16:9) |
| `VLM_QUALITY` | `60` | JPEG quality of the frame sent to VLM |
| `MAX_TOKENS` | `80` | Maximum tokens in VLM response — do not exceed 100 |
| `MODEL` | `gemma3:4b` | Ollama model name |

### Other settings

| Setting | Default | Description |
|---|---|---|
| `ALERT_SOUND` | `True` | Enable or disable the audible beep |
| `WEB_PORT` | `8080` | HTTP server port |

### Performance tuning guide

| Goal | Recommended change |
|---|---|
| Reduce CPU / GPU load | Increase `FRAME_SKIP` to `60` or `90` |
| Fix laggy video stream | Lower `STREAM_QUALITY` to `55`, lower `STREAM_FPS_CAP` to `15` |
| Detect smoke more frequently | Lower `FRAME_SKIP` to `15` |
| Higher accuracy (at cost of speed) | Change `MODEL` to `llama3.2-vision:11b` |

---

## Expected Performance (Jetson Orin NX)

| Component | Latency |
|---|---|
| Video stream (MJPEG) | ~100–300 ms |
| VLM inference — `gemma3:4b` | ~3–6 seconds per analysis |
| VLM inference — `llama3.2-vision:11b` | ~8–12 seconds per analysis |

---

## Troubleshooting

### Video does not appear in the dashboard

1. Open `http://localhost:8080/stream` directly in a new tab — if video appears here, the stream is working and the problem is in the dashboard page
2. Run `debug_camera.py` (see below) to test the camera independently
3. Check the terminal for the messages `Frame loop started` and `MJPEG client connected`
4. Confirm that `detector.py` and `index.html` are in the **same folder**

### Camera not detected

```bash
# List all video devices recognised by the system
ls /dev/video*

# Get detailed device information
v4l2-ctl --list-devices
```

### Ollama not responding

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama manually if needed
ollama serve &
```

### Port 8080 already in use

Change `WEB_PORT = 8080` to any available port (e.g. `8888`) in `detector.py`, then restart.

### System throttled / degraded performance

- Use a **19 V / 4 A** power adapter as specified for Jetson Orin NX
- Monitor power and thermal state in real time: `sudo jtop`
- Reduce processing load: increase `FRAME_SKIP`, lower `STREAM_FPS_CAP`

---

## Camera Diagnostic Tool

If the video stream is not working, run `debug_camera.py` before launching the main detector:

```bash
# Stop detector.py first (Ctrl+C), then run:
python3 debug_camera.py
```

This tool will:
- Scan `/dev/video0` through `/dev/video5` and report which cameras can successfully read frames
- Start a standalone MJPEG stream on port **8090**
- Print a frame count every 30 frames to confirm the camera is running

Open the following URL to verify the stream works independently of the main system:

```
http://localhost:8090
```

---

## License

This project is provided for educational and research purposes.  
Model: [gemma3:4b](https://ollama.com/library/gemma3) via [Ollama](https://ollama.com).
