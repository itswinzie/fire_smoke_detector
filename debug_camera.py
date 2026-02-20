#!/usr/bin/env python3
"""
debug_camera.py — Standalone camera and MJPEG stream diagnostic tool.

Run this BEFORE detector.py to verify that:
  1. Your camera is detected and can read frames
  2. The MJPEG stream works correctly in your browser

Usage:
    python3 debug_camera.py

Then open http://localhost:8090 in your browser.
"""

import cv2
import sys
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8090

# ─── Shared JPEG buffer ───────────────────────────────────────────────────────
_lock        = threading.Lock()
_latest_jpeg = None

# ─── Camera scan ─────────────────────────────────────────────────────────────

def find_cameras(max_index: int = 6) -> list[int]:
    print(f"\n[1] Scanning camera indices 0 – {max_index - 1} ...")
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"    ✅  /dev/video{i}  →  {w}×{h}")
                found.append(i)
            else:
                print(f"    ⚠️   /dev/video{i}  →  opens but cannot read frames")
            cap.release()
        else:
            print(f"    ❌  /dev/video{i}  →  cannot be opened")
    return found

# ─── Capture thread ───────────────────────────────────────────────────────────

def capture_loop(cam_idx: int):
    global _latest_jpeg
    print(f"\n[2] Starting capture from /dev/video{cam_idx} ...")
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"    ⚠️   Frame read failed (total captured: {count})")
            time.sleep(0.1)
            continue

        count += 1
        h, w = frame.shape[:2]

        # Annotate frame for easy verification
        cv2.putText(frame, f"Frame: {count}", (10, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 230, 80), 2)
        cv2.putText(frame, f"/dev/video{cam_idx}  {w}x{h}", (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
        cv2.putText(frame, "debug_camera.py", (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _lock:
            _latest_jpeg = buf.tobytes()

        if count % 30 == 0:
            print(f"    📷  {count} frames captured OK")

        time.sleep(0.033)

# ─── HTTP handler ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            print(f"    🌐  Browser connected to MJPEG stream.")
            sent = 0
            try:
                while True:
                    with _lock:
                        jpeg = _latest_jpeg
                    if jpeg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                        b"\r\n" + jpeg + b"\r\n"
                    )
                    self.wfile.flush()
                    sent += 1
                    if sent % 30 == 0:
                        print(f"    📡  {sent} frames streamed OK")
                    time.sleep(0.033)
            except Exception as e:
                print(f"    Stream ended: {e}")

        else:
            html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <title>Camera Debug</title>
  <style>
    body{{background:#080b10;color:#b8cce0;font-family:monospace;padding:28px;}}
    h2{{color:#ff3d1a;letter-spacing:2px;margin-bottom:8px}}
    p{{color:#667788;margin:6px 0}}
    img{{display:block;margin-top:20px;border:2px solid #1a2130;max-width:100%}}
    .ok{{color:#00e87a}}
  </style>
</head>
<body>
  <h2>🔥 Fire & Smoke Detection — Camera Debug</h2>
  <p>If the video below is live and moving, the camera and MJPEG stream are working correctly.</p>
  <p>Stream URL: <span class="ok">http://localhost:{PORT}/stream</span></p>
  <img src="/stream" alt="Camera feed" style="width:640px;height:480px;object-fit:contain"/>
</body>
</html>""".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(html))
            self.end_headers()
            self.wfile.write(html)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("  CAMERA & STREAM DIAGNOSTIC TOOL")
    print("  Fire & Smoke Detection System")
    print("=" * 52)

    cameras = find_cameras()

    if not cameras:
        print("\n❌  No working cameras found.")
        print("    Try: ls /dev/video*")
        print("    Try: v4l2-ctl --list-devices")
        sys.exit(1)

    # Select camera
    cam_idx = cameras[0]
    if len(cameras) > 1:
        opts = "/".join(str(i) for i in cameras)
        choice = input(f"\n    Select camera index [{opts}] (default {cam_idx}): ").strip()
        if choice.isdigit() and int(choice) in cameras:
            cam_idx = int(choice)

    # Start capture thread
    t = threading.Thread(target=capture_loop, args=(cam_idx,), daemon=True)
    t.start()

    # Wait for first frame (up to 5 s)
    print("\n[3] Waiting for first frame...")
    for _ in range(50):
        with _lock:
            if _latest_jpeg is not None:
                print("    ✅  First frame received OK")
                break
        time.sleep(0.1)
    else:
        print("    ❌  No frame after 5 seconds — camera may be broken or in use.")
        sys.exit(1)

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"\n[4] Debug server ready")
    print(f"    Open in browser  →  http://localhost:{PORT}")
    print(f"    Direct stream    →  http://localhost:{PORT}/stream")
    print("\n    Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
