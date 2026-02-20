# 🔥 Fire & Smoke Detection System
### Jetson Orin NX | Ollama gemma3:4b | Real-time Dashboard

---

## Cara Setup & Jalankan

### Langkah 1 — Clone / Copy fail-fail ini ke Jetson anda

```
fire_smoke_detector/
├── detector.py       ← Backend Python (VLM + server)
├── index.html        ← Dashboard UI
├── requirements.txt  ← Python dependencies
├── run.sh            ← Script setup + run automatik
└── README.md
```

### Langkah 2 — Jalankan dengan satu command

```bash
chmod +x run.sh
./run.sh
```

Script ini akan:
1. Install Ollama (jika belum ada)
2. Download model `gemma3:4b`
3. Install Python dependencies
4. Launch sistem detection

### Langkah 3 — Pilih sumber video

```
Select input source:
  [1] Webcam (real-time)
  [2] Video file

Enter choice (1/2): 
```

### Langkah 4 — Buka Dashboard

```
http://localhost:8080
```

Atau dari PC lain di network yang sama:
```
http://<IP_JETSON>:8080
```

---

## Tuning Prestasi

Edit bahagian **Config** dalam `detector.py`:

| Setting | Default | Keterangan |
|---|---|---|
| `FRAME_SKIP` | 30 | Analisis setiap N frame. Rendahkan untuk lebih kerap (tapi lebih lambat) |
| `MAX_TOKENS` | 80 | Token output VLM. 80 cukup untuk detection |
| `ALERT_SOUND` | True | Toggle bunyi beep |
| `MODEL` | gemma3:4b | Tukar ke model lain jika mahu |

## Jangkaan Prestasi (Jetson Orin NX)

| Model | Latency per Frame |
|---|---|
| gemma3:4b | ~3–6 saat |
| llama3.2-vision:11b | ~8–12 saat |

---

## Cara Tukar Alert Interval

Dalam `detector.py`, baris:
```python
FRAME_SKIP = 30   # Analisis setiap 30 frame
```
Kamera 30fps → analisis setiap ~1 saat.
Kamera 30fps + `FRAME_SKIP=60` → analisis setiap ~2 saat.

---

## Troubleshooting

**Ollama tidak respond:**
```bash
ollama serve &
curl http://localhost:11434/api/tags
```

**Kamera tidak dikesan:**
```bash
ls /dev/video*   # Lihat kamera yang tersedia
```

**Port 8080 digunakan:**
Tukar `WEB_PORT = 8080` dalam `detector.py`
