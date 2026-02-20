#!/usr/bin/env python3
"""
Fire & Smoke Detection System  —  v4
Jetson Orin NX | Ollama gemma3:4b | MJPEG Stream

Speed design decisions (target < 5 s end-to-end):
  - VLM frame  : 320x180 px  (was 640x360) — ~40% faster inference
  - MAX_TOKENS : 60          (was 120)     — model stops generating sooner
  - FRAME_SKIP : 1           (send every frame) — worker picks latest immediately
  - Persistent aiohttp session — no TCP reconnect overhead per call (~150ms saved)
  - JSON regex fallback — no worker crash on partial model output
  - Smoke enhancement OFF by default — saves ~50ms per frame
  - Camera tested before threads launch — no silent failures

Accuracy design decisions:
  - Prompt enforces explicit priority: FIRE > BOTH > SMOKE > NONE
  - This prevents fire being mis-classified as smoke
  - ENHANCE_SMOKE = True re-enables CLAHE if faint smoke is missed

Architecture:
  Thread A — Frame loop  : reads camera at full speed, draws overlay, serves MJPEG
  Thread B — VLM worker  : grabs latest frame from queue, calls Ollama, updates state
  HTTP Server            : GET /stream  |  GET /api/state  |  GET /
"""

import asyncio
import base64
import json
import logging
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import aiohttp
import cv2
import numpy as np

# ==============================================================================
#  CONFIG  <-- edit this section to tune the system
# ==============================================================================

OLLAMA_URL    = "http://localhost:11434/v1/chat/completions"
MODEL         = "gemma3:4b"
WEB_PORT      = 8080

# -- Stream sent to browser ----------------------------------------------------
STREAM_WIDTH   = 1120   # fits Vivaldi fullscreen (video panel ~1134 px wide)
STREAM_HEIGHT  = 630    # 16:9
STREAM_QUALITY = 72     # JPEG quality 1-100
STREAM_FPS_CAP = 30     # max fps delivered to browser

# -- Frame sent to Ollama ------------------------------------------------------
# 320x180 gives ~40% speed improvement over 640x360 with minimal accuracy loss.
# Increase to 480x270 or 640x360 only if detection quality is insufficient.
VLM_WIDTH     = 320
VLM_HEIGHT    = 180
VLM_QUALITY   = 60     # JPEG quality for VLM frame

# -- Inference frequency -------------------------------------------------------
# FRAME_SKIP = 1 -> queue updated on every frame; worker picks the latest.
# FRAME_SKIP = 5 -> minor CPU saving at the cost of a slightly stale frame.
FRAME_SKIP    = 1

# -- VLM output length ---------------------------------------------------------
# 60 tokens is enough for the required JSON object.
# Lower = faster. Do not exceed 100.
MAX_TOKENS    = 60

# -- Smoke pre-processing ------------------------------------------------------
# CLAHE contrast boost helps detect faint smoke but costs ~50ms per frame.
# Leave False for maximum speed; set True if subtle smoke is being missed.
ENHANCE_SMOKE = False

# -- Alert sound ---------------------------------------------------------------
ALERT_SOUND   = True

# -- Log directory -------------------------------------------------------------
LOG_DIR       = Path("logs")

# ==============================================================================
#  DETECTION PROMPT
#
#  Rules are listed in strict priority order so the model never labels
#  active fire as smoke.  The prompt is intentionally concise to minimise
#  input-token processing time.
# ==============================================================================

DETECTION_PROMPT = """\
You are a fire and smoke detection system. Classify the image strictly by these rules in order:

1. FIRE  -- any visible flames (orange / red / yellow / white), glowing embers, or burning material.
            Use "fire" even when smoke is also present, UNLESS the smoke is a clearly separate dominant hazard.
2. BOTH  -- active flames AND a distinct separate smoke plume are both clearly visible.
3. SMOKE -- haze, grey/white/brown plume, cloudy or milky air, drifting wisps -- with NO visible flames.
4. NONE  -- completely clear scene with no fire or smoke.

Respond with ONLY this JSON (no markdown, no explanation, no extra text):
{"detected":true/false,"type":"fire"|"smoke"|"both"|"none","confidence":"high"|"medium"|"low","description":"one short sentence"}
"""

# ==============================================================================
#  LOGGING
# ==============================================================================

LOG_DIR.mkdir(exist_ok=True)
_log_file = LOG_DIR / f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("FireDetector")

# ==============================================================================
#  SHARED STATE
#  All reads / writes protected by _state_lock.
#  Use get_state() and set_state() -- never access _state directly.
# ==============================================================================

_state_lock = threading.Lock()
_state = {
    "alert":       False,
    "type":        "none",      # fire | smoke | both | none
    "confidence":  "high",
    "description": "Starting up...",
    "latency_ms":  0,
    "frame_count": 0,
    "alert_count": 0,
    "last_alert":  None,        # "HH:MM:SS" string of most recent alert
    "source":      "\u2014",
    "status":      "starting",  # starting | running | stopped | error
    "log_path":    str(_log_file),
}


def get_state():
    with _state_lock:
        return dict(_state)


def set_state(**kwargs):
    with _state_lock:
        _state.update(kwargs)


# Latest JPEG bytes served to MJPEG clients
_frame_lock  = threading.Lock()
_latest_jpeg = None   # bytes | None

# Single-slot queue -- Thread A always overwrites; Thread B always picks newest
_vlm_queue = queue.Queue(maxsize=1)

# ==============================================================================
#  IMAGE UTILITIES
# ==============================================================================

def enhance_for_smoke(frame):
    """
    CLAHE contrast enhancement in LAB colour space + mild sharpening kernel.
    Improves visibility of faint smoke without altering colour balance.
    Only called when ENHANCE_SMOKE = True (costs ~50ms on Jetson Orin NX).
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    kernel = np.array([[ 0,   -0.4,  0  ],
                       [-0.4,  2.6, -0.4],
                       [ 0,   -0.4,  0  ]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return cv2.addWeighted(sharpened, 0.65, frame, 0.35, 0)


def to_jpeg(frame, quality):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def to_b64(frame, quality):
    return base64.b64encode(to_jpeg(frame, quality)).decode()

# ==============================================================================
#  VIDEO OVERLAY
#  Drawn on every frame before it is pushed to the MJPEG stream.
#  Must be fast -- runs inside Thread A at up to 30 fps.
# ==============================================================================

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_FONTB = cv2.FONT_HERSHEY_DUPLEX

# BGR colour per detection type
_COLOUR = {
    "fire":  (0,  50, 255),    # vivid red-orange
    "smoke": (160, 140, 100),  # grey-brown
    "both":  (0,  50, 255),    # same as fire
    "none":  (0, 200,  80),    # green
}


def draw_overlay(frame, s):
    """Composite alert visuals onto a copy of frame and return it."""
    out   = frame.copy()
    h, w  = out.shape[:2]
    alert = s["alert"]
    dtype = s["type"]
    desc  = s["description"]
    lat   = s["latency_ms"]
    ts    = datetime.now().strftime("%H:%M:%S")
    col   = _COLOUR.get(dtype, _COLOUR["none"])

    if alert:
        # Border flashes at 2 Hz (on 0.5 s, off 0.5 s)
        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), col, 10)

        # Colour tint -- red for fire/both, brown for smoke
        tint = out.copy()
        tint[:] = (0, 0, 130) if dtype in ("fire", "both") else (60, 50, 30)
        out = cv2.addWeighted(out, 0.82, tint, 0.18, 0)

        # Large alert text centred in upper region
        label_map = {
            "fire":  "!! FIRE DETECTED !!",
            "smoke": "!! SMOKE DETECTED !!",
            "both":  "!! FIRE & SMOKE !!",
        }
        label = label_map.get(dtype, "!! ALERT !!")
        scale = max(0.8, w / 520)
        (tw, _), _ = cv2.getTextSize(label, _FONTB, scale, 3)
        tx = (w - tw) // 2
        ty = int(h * 0.17)
        cv2.putText(out, label, (tx + 2, ty + 2), _FONTB, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label, (tx, ty),          _FONTB, scale, col,       3, cv2.LINE_AA)

    # HUD bar (always visible at bottom)
    bar_y  = h - 38
    bar_bg = out.copy()
    cv2.rectangle(bar_bg, (0, bar_y), (w, h), (10, 10, 10), -1)
    out = cv2.addWeighted(out, 0.40, bar_bg, 0.60, 0)

    status_text = "CLEAR" if not alert else dtype.upper()
    cv2.putText(out, f"STATUS: {status_text}",
                (10, bar_y + 25), _FONTB, 0.55, col, 1, cv2.LINE_AA)
    cv2.putText(out, f"VLM: {lat}ms" if lat else "VLM: --",
                (w // 3, bar_y + 25), _FONT, 0.48, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(out, ts,
                (w - 105, bar_y + 25), _FONT, 0.48, (100, 100, 100), 1, cv2.LINE_AA)

    # Description just above the HUD bar
    if desc and len(desc) > 3:
        short = desc[:90]
        cv2.putText(out, short, (11, bar_y - 9),  _FONT, 0.43, (0, 0, 0),       2, cv2.LINE_AA)
        cv2.putText(out, short, (10, bar_y - 10), _FONT, 0.43, (200, 200, 200), 1, cv2.LINE_AA)

    return out

# ==============================================================================
#  THREAD A -- FRAME LOOP
#
#  Reads the camera at full speed, applies overlay from current shared state,
#  encodes to JPEG, and stores in _latest_jpeg for the MJPEG server.
#  Pushes a downscaled copy to _vlm_queue every FRAME_SKIP frames.
#  Never blocks on VLM -- the queue drop is non-blocking (put_nowait).
# ==============================================================================

def frame_loop(source):
    global _latest_jpeg

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Cannot open source: {source}")
        set_state(status="error", description="Cannot open camera / video file.")
        return

    # Reduce internal camera buffer to 1 frame for lowest possible latency
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    src_label = "Webcam" if isinstance(source, int) else Path(source).name
    set_state(source=src_label, status="running",
              description="Waiting for first analysis...")
    log.info(f"Frame loop started -- source: {src_label}")

    frame_idx = 0
    interval  = 1.0 / STREAM_FPS_CAP

    while True:
        t0         = time.time()
        ret, frame = cap.read()

        if not ret:
            if isinstance(source, str):      # video file finished -- loop it
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break                            # webcam error -- exit

        frame_idx += 1
        set_state(frame_count=frame_idx)

        # Resize to stream resolution, draw overlay, store JPEG
        s       = get_state()
        display = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT),
                             interpolation=cv2.INTER_LINEAR)
        display = draw_overlay(display, s)

        with _frame_lock:
            _latest_jpeg = to_jpeg(display, STREAM_QUALITY)

        # Prepare and push VLM frame (drops silently if worker is busy)
        if frame_idx % FRAME_SKIP == 0:
            vlm_frame = cv2.resize(frame, (VLM_WIDTH, VLM_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)
            if ENHANCE_SMOKE:
                vlm_frame = enhance_for_smoke(vlm_frame)
            try:
                _vlm_queue.put_nowait(vlm_frame)
            except queue.Full:
                pass  # VLM still busy -- drop this frame, not a problem

        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    cap.release()
    set_state(status="stopped")
    log.info("Frame loop stopped.")

# ==============================================================================
#  THREAD B -- VLM WORKER
#
#  Runs a single long-lived aiohttp ClientSession (persistent TCP connection).
#  This saves ~100-200ms per call compared to opening a new session each time.
#
#  JSON parsing uses a regex fallback so the worker never crashes on
#  partial or markdown-wrapped model output.
# ==============================================================================

_JSON_PATTERN = re.compile(r'\{[^{}]+\}', re.DOTALL)


def _parse_vlm_response(raw):
    """
    Parse model output to dict.
    1. Strip backticks and 'json' prefix if present.
    2. Try strict json.loads().
    3. Fall back to regex -- extract the first {...} block.
    Raises json.JSONDecodeError only if both attempts fail.
    """
    raw = raw.strip().strip("`").strip()
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_PATTERN.search(raw)
        if match:
            return json.loads(match.group())
        raise


async def _vlm_session_loop():
    """
    Long-lived async loop.
    One ClientSession is created once and reused for all Ollama requests --
    avoids reconnect overhead on every inference call.
    """
    connector = aiohttp.TCPConnector(limit=1, force_close=False)
    timeout   = aiohttp.ClientTimeout(total=30, connect=5)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        log.info("VLM session open -- persistent connection to Ollama active.")

        while True:
            # Wait (blocking executor) until Frame Loop puts a frame
            frame = await asyncio.get_event_loop().run_in_executor(
                None, _vlm_queue.get
            )

            b64 = to_b64(frame, VLM_QUALITY)
            payload = {
                "model":      MODEL,
                "max_tokens": MAX_TOKENS,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": DETECTION_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        }},
                    ],
                }],
            }

            t0 = time.time()
            try:
                async with session.post(OLLAMA_URL, json=payload) as resp:
                    data = await resp.json(content_type=None)

                latency     = int((time.time() - t0) * 1000)
                raw         = data["choices"][0]["message"]["content"].strip()
                result      = _parse_vlm_response(raw)

                detected    = bool(result.get("detected", False))
                dtype       = result.get("type", "none")
                confidence  = result.get("confidence", "low")
                description = result.get("description", "")

                # Guard: model said not-detected but gave a hazard type
                if not detected:
                    dtype = "none"

                if detected:
                    log.warning(
                        f"ALERT [{dtype.upper()}] {confidence}"
                        f" -- {description} ({latency}ms)"
                    )
                    _beep()
                    with _state_lock:
                        _state["alert_count"] += 1
                        _state["last_alert"]   = datetime.now().strftime("%H:%M:%S")
                else:
                    log.info(
                        f"Clear [{dtype}] {confidence}"
                        f" -- {description} ({latency}ms)"
                    )

                set_state(
                    alert=detected,
                    type=dtype,
                    confidence=confidence,
                    description=description,
                    latency_ms=latency,
                )

            except json.JSONDecodeError as exc:
                log.warning(f"JSON parse failed: {exc}")
            except asyncio.TimeoutError:
                log.warning("Ollama request timed out (>30s) -- skipping frame.")
                set_state(description="Ollama timeout -- retrying...")
            except Exception as exc:
                log.error(f"VLM error: {exc}")
                set_state(description=f"VLM error: {str(exc)[:80]}")


def vlm_worker():
    """Thread B entry point -- creates event loop and runs the session loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_vlm_session_loop())

# ==============================================================================
#  ALERT SOUND
# ==============================================================================

def _beep():
    """Play alert beep in a daemon thread -- never blocks the VLM worker."""
    if not ALERT_SOUND:
        return

    def _play():
        try:
            subprocess.run(
                ["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
                capture_output=True, timeout=3,
            )
        except Exception:
            try:
                sys.stdout.write("\a")
                sys.stdout.flush()
            except Exception:
                pass

    threading.Thread(target=_play, daemon=True).start()

# ==============================================================================
#  HTTP SERVER
#
#  GET /stream     -- MJPEG multipart stream (continuous)
#  GET /api/state  -- current detection state as JSON
#  GET /           -- dashboard HTML (index.html, same directory as this file)
# ==============================================================================

_BASE_DIR   = Path(__file__).parent.resolve()
_INDEX_HTML = _BASE_DIR / "index.html"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # suppress per-request access logs

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]  # strip query string

        # /stream -- MJPEG --------------------------------------------------------
        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Connection",    "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            log.info("MJPEG client connected.")
            try:
                while True:
                    with _frame_lock:
                        jpeg = _latest_jpeg
                    if jpeg is None:
                        time.sleep(0.033)
                        continue
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                        b"\r\n" + jpeg + b"\r\n"
                    )
                    self.wfile.flush()
                    time.sleep(1.0 / STREAM_FPS_CAP)
            except (BrokenPipeError, ConnectionResetError, OSError):
                log.info("MJPEG client disconnected.")

        # /api/state -- JSON -------------------------------------------------------
        elif path == "/api/state":
            body = json.dumps(get_state()).encode()
            self.send_response(200)
            self.send_header("Content-Type",   "application/json")
            self.send_header("Cache-Control",  "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        # / or /index.html -- Dashboard -------------------------------------------
        elif path in ("/", "/index.html"):
            try:
                body = _INDEX_HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type",   "text/html; charset=utf-8")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            except FileNotFoundError:
                self.send_error(404, f"index.html not found at {_INDEX_HTML}")
        else:
            self.send_error(404)

# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print("\n" + "=" * 60)
    print("  FIRE & SMOKE DETECTION SYSTEM  v4")
    print("  Jetson Orin NX | Ollama gemma3:4b")
    print("=" * 60)

    # Select input source
    print("\nSelect input source:")
    print("  [1]  Webcam (real-time)")
    print("  [2]  Video file")
    choice = input("\nEnter choice (1 / 2): ").strip()

    if choice == "1":
        idx    = input("Camera index [default 0, press Enter to skip]: ").strip()
        source = int(idx) if idx.isdigit() else 0
    elif choice == "2":
        source = input("Full path to video file: ").strip()
        if not Path(source).exists():
            print(f"\n  File not found: {source}")
            sys.exit(1)
    else:
        print("Invalid choice -- defaulting to webcam index 0.")
        source = 0

    # Quick camera / file test before launching threads
    print("\n  Testing camera / file...")
    cap_test = cv2.VideoCapture(source)
    if not cap_test.isOpened():
        print(f"  Cannot open: {source}")
        print("  Run debug_camera.py to diagnose camera issues.")
        sys.exit(1)
    ok, _ = cap_test.read()
    cap_test.release()
    if not ok:
        print(f"  Camera opened but cannot read a frame from: {source}")
        sys.exit(1)
    print("  Camera / file OK")

    # Launch Frame Loop and VLM Worker threads
    threading.Thread(target=frame_loop, args=(source,),
                     daemon=True, name="FrameLoop").start()
    threading.Thread(target=vlm_worker,
                     daemon=True, name="VLMWorker").start()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", WEB_PORT), _Handler)
    threading.Thread(target=server.serve_forever,
                     daemon=True, name="HTTPServer").start()

    url = f"http://localhost:{WEB_PORT}"
    print(f"\n  Dashboard  ->  {url}")
    print(f"  Stream     ->  {url}/stream")
    print(f"  Log        ->  {_log_file}")
    print()
    print(f"  Stream  : {STREAM_WIDTH}x{STREAM_HEIGHT} @ {STREAM_FPS_CAP} fps  Q{STREAM_QUALITY}")
    print(f"  VLM     : {VLM_WIDTH}x{VLM_HEIGHT}  Q{VLM_QUALITY}  max_tokens={MAX_TOKENS}")
    print(f"  Model   : {MODEL}")
    print(f"  Enhance : ENHANCE_SMOKE={ENHANCE_SMOKE}")
    print()
    print("Press Ctrl+C to stop.\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    def _shutdown(sig, _frame):
        print("\nShutting down...")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.pause()


if __name__ == "__main__":
    main()
