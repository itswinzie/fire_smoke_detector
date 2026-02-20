#!/usr/bin/env python3
"""
Fire & Smoke Detection System  —  v2 (MJPEG Stream)
Jetson Orin NX + Ollama (gemma3:4b)

Architecture (2 thread berasingan):
  Thread A — FRAME LOOP  : Baca kamera secepat mungkin, overlay alert,
                            simpan latest_frame untuk MJPEG stream.
  Thread B — VLM WORKER  : Ambil frame setiap FRAME_SKIP, hantar ke Ollama,
                            update shared_state. Tidak block Thread A.
  HTTP Server            : /stream    → MJPEG stream (Thread A frame)
                           /api/state → JSON state
                           /          → Dashboard UI
"""

import cv2
import asyncio
import aiohttp
import base64
import json
import time
import threading
import logging
import sys
import os
import subprocess
import queue
import numpy as np
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
import signal

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  tukar nilai di sini untuk optimum prestasi
# ══════════════════════════════════════════════════════════════════════════════
OLLAMA_URL      = "http://localhost:11434/v1/chat/completions"
MODEL           = "gemma3:4b"
WEB_PORT        = 8080

# MJPEG stream
STREAM_QUALITY  = 72     # JPEG quality
STREAM_WIDTH    = 1120   # Muat tepat dalam video panel Vivaldi (1134px lebar)
STREAM_HEIGHT   = 630    # Muat tepat dalam video panel (672px tinggi), 16:9
STREAM_FPS_CAP  = 30     # Max FPS dihantar ke browser

# VLM analysis
FRAME_SKIP      = 30     # Analisis setiap N frame (30fps → ~1x/saat)
VLM_QUALITY     = 60     # JPEG quality frame yang dihantar ke VLM
VLM_WIDTH       = 640    # Lebih kecil untuk VLM lebih laju
VLM_HEIGHT      = 360

MAX_TOKENS      = 80
ALERT_SOUND     = True

LOG_DIR         = Path("logs")

FIRE_PROMPT = """You are an expert fire and smoke detection AI with high sensitivity to smoke.

SMOKE indicators to look for (even subtle signs):
- Hazy, milky, or cloudy air that reduces visibility or blurs background details
- Grey, white, brown, or yellowish drifting wisps or plumes
- Semi-transparent layers floating in the air
- Unusual haziness around light sources or windows
- Diffuse cloudiness that was not present in a normal scene
- Any airborne particles that obscure or soften edges of objects

FIRE indicators:
- Orange, red, or yellow flames
- Glowing embers or bright flickering light
- Charred or burning objects

Be sensitive: if you see ANY haze, unusual cloudiness, or airborne particles — detect it as smoke.

Answer ONLY in this exact JSON format (no extra text):
{"detected": true/false, "type": "fire"|"smoke"|"both"|"none", "confidence": "high"|"medium"|"low", "description": "one short sentence describing what you see"}

Examples:
{"detected": true, "type": "smoke", "confidence": "high", "description": "Thick grey smoke plume rising from lower left"}
{"detected": true, "type": "smoke", "confidence": "medium", "description": "Faint hazy layer visible near ceiling, air appears cloudy"}
{"detected": true, "type": "fire", "confidence": "high", "description": "Large orange flames on right side of frame"}
{"detected": true, "type": "both", "confidence": "high", "description": "Active flames with heavy smoke filling upper area"}
{"detected": false, "type": "none", "confidence": "high", "description": "Clear indoor scene, no haze or fire visible"}"""

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("FireDetector")

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════
state_lock = threading.Lock()
shared_state = {
    "alert":       False,
    "type":        "none",
    "confidence":  "",
    "description": "Initialising...",
    "latency_ms":  0,
    "frame_count": 0,
    "alert_count": 0,
    "last_alert":  None,
    "source":      "—",
    "status":      "starting",
    "log_path":    str(log_file),
}

# Latest JPEG bytes untuk MJPEG stream
frame_lock  = threading.Lock()
latest_jpeg = None   # bytes | None

# Queue untuk hantar frame ke VLM worker (maxsize=1 → buang frame lama)
vlm_queue = queue.Queue(maxsize=1)

# ══════════════════════════════════════════════════════════════════════════════
#  OVERLAY — lukis alert terus pada frame (OpenCV)
# ══════════════════════════════════════════════════════════════════════════════
FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

def draw_overlay(frame: np.ndarray, state: dict) -> np.ndarray:
    out   = frame.copy()
    h, w  = out.shape[:2]
    alert = state["alert"]
    dtype = state["type"]
    desc  = state["description"]
    latency = state["latency_ms"]
    ts    = datetime.now().strftime("%H:%M:%S")

    if alert:
        # ── Border merah berkedip ─────────────────────────────────────────────
        border_color = (0, 0, 255) if dtype in ("fire", "both") else (140, 120, 80)
        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), border_color, 10)

        # ── Tint overlay merah ────────────────────────────────────────────────
        tint = out.copy()
        if dtype in ("fire", "both"):
            tint[:] = (0, 0, 160)
        else:
            tint[:] = (80, 60, 40)
        out = cv2.addWeighted(out, 0.80, tint, 0.20, 0)

        # ── Teks amaran besar (tengah atas) ───────────────────────────────────
        label = {
            "fire":  "!! FIRE DETECTED !!",
            "smoke": "!! SMOKE DETECTED !!",
            "both":  "!! FIRE & SMOKE !!",
        }.get(dtype, "!! ALERT !!")

        scale = max(0.8, w / 500)
        (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, scale, 3)
        tx = (w - tw) // 2
        ty = int(h * 0.18)
        cv2.putText(out, label, (tx + 2, ty + 2), FONT_BOLD, scale, (0, 0, 0), 4, cv2.LINE_AA)
        txt_col = (60, 120, 255) if dtype in ("fire", "both") else (180, 200, 255)
        cv2.putText(out, label, (tx, ty), FONT_BOLD, scale, txt_col, 3, cv2.LINE_AA)

    # ── HUD bar bawah (selalu papar) ──────────────────────────────────────────
    bar_h = 38
    bar_y = h - bar_h
    bar_bg = out.copy()
    cv2.rectangle(bar_bg, (0, bar_y), (w, h), (8, 8, 8), -1)
    out = cv2.addWeighted(out, 0.40, bar_bg, 0.60, 0)

    status_str   = "CLEAR" if not alert else dtype.upper()
    status_color = (0, 210, 70) if not alert else (40, 80, 255)
    cv2.putText(out, f"STATUS: {status_str}", (10, bar_y + 25),
                FONT_BOLD, 0.55, status_color, 1, cv2.LINE_AA)

    lat_str = f"VLM: {latency}ms" if latency else "VLM: —"
    cv2.putText(out, lat_str, (w // 3, bar_y + 25),
                FONT, 0.48, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(out, ts, (w - 105, bar_y + 25),
                FONT, 0.48, (110, 110, 110), 1, cv2.LINE_AA)

    # ── Deskripsi (atas HUD bar) ──────────────────────────────────────────────
    if desc and len(desc) > 3:
        short = desc[:85]
        cv2.putText(out, short, (11, bar_y - 9), FONT, 0.44, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, short, (10, bar_y - 10), FONT, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

    return out

# ══════════════════════════════════════════════════════════════════════════════
#  THREAD A — FRAME LOOP (laju, tidak tunggu VLM)
# ══════════════════════════════════════════════════════════════════════════════
def frame_loop(source):
    global latest_jpeg
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Cannot open source: {source}")
        with state_lock:
            shared_state["status"] = "error"
        return

    # Kurangkan buffer webcam untuk latency rendah
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    src_label = "Webcam" if isinstance(source, int) else Path(source).name
    with state_lock:
        shared_state["source"] = src_label
        shared_state["status"] = "running"

    log.info(f"Frame loop started — source: {src_label}")
    frame_idx = 0
    frame_interval = 1.0 / STREAM_FPS_CAP

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):   # video fail: ulang dari mula
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_idx += 1
        with state_lock:
            shared_state["frame_count"] = frame_idx
            cur_state = dict(shared_state)

        # Resize untuk stream
        if STREAM_WIDTH > 0:
            stream_frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT),
                                      interpolation=cv2.INTER_LINEAR)
        else:
            stream_frame = frame

        # Lukis overlay
        display = draw_overlay(stream_frame, cur_state)

        # Encode JPEG
        _, buf = cv2.imencode(".jpg", display,
                              [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
        with frame_lock:
            latest_jpeg = buf.tobytes()

        # Hantar ke VLM queue setiap FRAME_SKIP
        if frame_idx % FRAME_SKIP == 0:
            vlm_frame = cv2.resize(frame, (VLM_WIDTH, VLM_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)
            try:
                vlm_queue.put_nowait(vlm_frame)
            except queue.Full:
                pass  # VLM masih sibuk, skip frame ini

        # Kawal FPS — tidur baki masa
        elapsed = time.time() - t_start
        sleep_t = frame_interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    cap.release()
    with state_lock:
        shared_state["status"] = "stopped"
    log.info("Frame loop stopped.")

# ══════════════════════════════════════════════════════════════════════════════
#  THREAD B — VLM WORKER (lambat, tidak block stream)
# ══════════════════════════════════════════════════════════════════════════════
def enhance_for_smoke(frame: np.ndarray) -> np.ndarray:
    """
    Pre-process frame sebelum hantar ke VLM.
    Tingkatkan kontras dan ketajaman supaya asap lebih mudah dikesan.
    """
    # Tukar ke LAB colour space — lebih baik untuk contrast enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE pada channel L (lightness) — tingkatkan kontras tempatan
    # clipLimit tinggi = lebih agresif; tileGridSize kecil = lebih detail
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Gabung semula dan tukar balik ke BGR
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Slight sharpening — bantu VLM nampak tepi asap yang kabur
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Blend — 70% enhanced + 30% original (jangan terlalu agresif)
    result = cv2.addWeighted(sharpened, 0.7, frame, 0.3, 0)
    return result


def frame_to_b64(frame, quality=60, enhance=True):
    """Encode frame ke base64 JPEG. Jika enhance=True, apply smoke enhancement dulu."""
    if enhance:
        frame = enhance_for_smoke(frame)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")

async def _call_vlm(b64: str) -> dict:
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",      "text": FIRE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }]
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json=payload,
                                timeout=aiohttp.ClientTimeout(total=60)) as r:
            data = await r.json()
            raw  = data["choices"][0]["message"]["content"].strip()
            raw  = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)

def vlm_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    log.info("VLM worker started.")

    while True:
        frame = vlm_queue.get()
        b64   = frame_to_b64(frame, VLM_QUALITY)
        t0    = time.time()
        try:
            result      = loop.run_until_complete(_call_vlm(b64))
            latency     = int((time.time() - t0) * 1000)
            detected    = result.get("detected", False)
            dtype       = result.get("type", "none")
            confidence  = result.get("confidence", "low")
            description = result.get("description", "")

            if detected:
                log.warning(f"🔥 ALERT [{dtype.upper()}] {confidence} — {description}")
                _beep()
                with state_lock:
                    shared_state["alert_count"] += 1
                    shared_state["last_alert"]   = datetime.now().strftime("%H:%M:%S")
            else:
                log.info(f"✅ Clear — {description} ({latency}ms)")

            with state_lock:
                shared_state.update({
                    "alert":       detected,
                    "type":        dtype,
                    "confidence":  confidence,
                    "description": description,
                    "latency_ms":  latency,
                })

        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error: {e}")
        except Exception as e:
            log.error(f"VLM error: {e}")
            with state_lock:
                shared_state["description"] = f"VLM Error: {str(e)[:60]}"

# ══════════════════════════════════════════════════════════════════════════════
#  BEEP
# ══════════════════════════════════════════════════════════════════════════════
def _beep():
    if not ALERT_SOUND:
        return
    def __b():
        try:
            subprocess.run(["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
                           capture_output=True, timeout=2)
        except Exception:
            try:
                sys.stdout.write('\a'); sys.stdout.flush()
            except Exception:
                pass
    threading.Thread(target=__b, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════════════
#  HTTP SERVER  —  MJPEG + JSON API + index.html
# ══════════════════════════════════════════════════════════════════════════════
# Path mutlak ke index.html (sama direktori dengan detector.py)
BASE_DIR   = Path(__file__).parent.resolve()
INDEX_HTML = BASE_DIR / "index.html"

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # senyapkan log HTTP

    def do_HEAD(self):
        """Sokong HEAD request — digunakan oleh browser untuk test stream."""
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
        else:
            self.send_response(200)
            self.end_headers()

    def do_GET(self):

        # ── /stream  →  MJPEG ─────────────────────────────────────────────────
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            log.info("MJPEG client connected.")
            try:
                while True:
                    with frame_lock:
                        jpeg = latest_jpeg
                    if jpeg is None:
                        time.sleep(0.05)
                        continue
                    # Format MJPEG yang betul
                    chunk = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                        b"\r\n" + jpeg + b"\r\n"
                    )
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    time.sleep(1.0 / STREAM_FPS_CAP)
            except (BrokenPipeError, ConnectionResetError, OSError):
                log.info("MJPEG client disconnected.")

        # ── /api/state  →  JSON ───────────────────────────────────────────────
        elif self.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with state_lock:
                self.wfile.write(json.dumps(shared_state).encode())

        # ── / atau /index.html  →  Dashboard ─────────────────────────────────
        elif self.path in ("/", "/index.html"):
            try:
                content = INDEX_HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", len(content))
                self.end_headers()
                self.wfile.write(content)
                log.info(f"Served index.html from {INDEX_HTML}")
            except FileNotFoundError:
                self.send_error(404, f"index.html not found at {INDEX_HTML}")
        else:
            self.send_error(404)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*60)
    print("  🔥 FIRE & SMOKE DETECTION SYSTEM  v2 (MJPEG)")
    print("     Jetson Orin NX | Ollama gemma3:4b")
    print("="*60)

    print("\nSelect input source:")
    print("  [1] Webcam (real-time)")
    print("  [2] Video file")
    choice = input("\nEnter choice (1/2): ").strip()

    if choice == "1":
        cam = input("Camera index (default 0, tekan Enter untuk skip): ").strip()
        source = int(cam) if cam.isdigit() else 0
    elif choice == "2":
        source = input("Enter full path to video file: ").strip()
        if not Path(source).exists():
            print(f"❌ File not found: {source}")
            sys.exit(1)
    else:
        print("Invalid, using webcam (0)")
        source = 0

    # Pastikan log dir wujud
    LOG_DIR.mkdir(exist_ok=True)

    # Thread A — Frame loop
    threading.Thread(target=frame_loop, args=(source,), daemon=True).start()
    # Thread B — VLM worker
    threading.Thread(target=vlm_worker, daemon=True).start()

    # HTTP server
    server = HTTPServer(("0.0.0.0", WEB_PORT), StreamHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://localhost:{WEB_PORT}"
    print(f"\n✅ Dashboard  →  {url}")
    print(f"📡 MJPEG     →  http://localhost:{WEB_PORT}/stream")
    print(f"📁 Log       →  {log_file}")
    print(f"\nStream  : {STREAM_WIDTH}×{STREAM_HEIGHT} @ {STREAM_FPS_CAP}fps  Q{STREAM_QUALITY}")
    print(f"VLM     : setiap {FRAME_SKIP} frame  |  {VLM_WIDTH}×{VLM_HEIGHT}  Q{VLM_QUALITY}")
    print("\nPress Ctrl+C to stop.\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    def handle_exit(sig, frame):
        print("\nShutting down...")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    signal.pause()

if __name__ == "__main__":
    main()
