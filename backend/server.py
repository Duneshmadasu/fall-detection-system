"""
Fall Detection Web Server
Streams webcam feed to browser and sends real-time fall alerts to dashboard
Run this instead of fall_detector.py
"""

from flask import Flask, Response, render_template_string
from flask_socketio import SocketIO
import cv2
import numpy as np
import time
import base64
import threading
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/yolov8n-pose.pt"
CONFIDENCE = 0.5
FALL_ASPECT_RATIO = 0.8
INACTIVITY_SECONDS = 2.0
CAMERA_SOURCE = 0

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ── GLOBALS ───────────────────────────────────────────────────────────────────
latest_frame = None
fall_candidates = {}
alert_cooldown = {}
alerts_log = []
frame_lock = threading.Lock()

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("[INFO] Loading YOLOv8 pose model...")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded!")

# ── FALL DETECTION LOGIC ──────────────────────────────────────────────────────
def is_fallen(kp):
    if kp is None or len(kp) < 17:
        return False

    def get_pt(idx):
        x, y, c = kp[idx]
        return (float(x), float(y)) if c > 0.3 else None

    points = [get_pt(i) for i in [0,5,6,11,12,15,16]]
    visible = [p for p in points if p]
    if len(visible) < 3:
        return False

    xs = [p[0] for p in visible]
    ys = [p[1] for p in visible]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if h == 0:
        return False

    if w / h > FALL_ASPECT_RATIO:
        return True

    ls, rs = get_pt(5), get_pt(6)
    lh, rh = get_pt(11), get_pt(12)
    if ls and rs and lh and rh:
        sh_y = (ls[1] + rs[1]) / 2
        hip_y = (lh[1] + rh[1]) / 2
        if abs(sh_y - hip_y) < abs(ls[0] - rs[0]) * 0.5:
            return True

    return False

# ── CAMERA THREAD ─────────────────────────────────────────────────────────────
def camera_thread():
    global latest_frame
    cap = cv2.VideoCapture(CAMERA_SOURCE)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        return

    print("[INFO] Camera started! Open http://localhost:5000 in your browser")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # Run YOLOv8
        results = model.track(frame, persist=True, conf=CONFIDENCE, verbose=False)
        annotated = results[0].plot()

        # Check for falls
        if results[0].keypoints is not None and results[0].boxes.id is not None:
            kps = results[0].keypoints.data.cpu().numpy()
            boxes = results[0].boxes

            for i, kp in enumerate(kps):
                track_id = int(boxes.id[i])
                fallen = is_fallen(kp)

                if fallen:
                    if track_id not in fall_candidates:
                        fall_candidates[track_id] = current_time

                    elapsed = current_time - fall_candidates[track_id]

                    if elapsed >= INACTIVITY_SECONDS:
                        last = alert_cooldown.get(track_id, 0)
                        if current_time - last > 30:
                            alert_cooldown[track_id] = current_time
                            trigger_fall_alert(track_id, annotated)
                else:
                    fall_candidates.pop(track_id, None)

        # Store frame
        with frame_lock:
            latest_frame = annotated.copy()

        time.sleep(0.03)

    cap.release()

# ── TRIGGER ALERT ─────────────────────────────────────────────────────────────
def trigger_fall_alert(person_id, frame):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Encode snapshot
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    snapshot = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    alert = {
        "alert_id": f"fall_{person_id}_{int(time.time())}",
        "person_id": str(person_id),
        "timestamp": timestamp,
        "location": "Camera-1",
        "status": "unacknowledged",
        "snapshot": snapshot
    }

    alerts_log.insert(0, alert)
    print(f"[ALERT] FALL CONFIRMED for person {person_id}!")

    # Send to dashboard via WebSocket
    socketio.emit("fall_alert", alert)

    # Send SMS + WhatsApp
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from alerts.firebase_alert import send_fall_alert
        send_fall_alert(person_id, frame)
    except Exception as e:
        print(f"[ERROR] Alert sending failed: {e}")

# ── VIDEO STREAM ──────────────────────────────────────────────────────────────
def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.1)
            continue

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(0.033)

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/alerts")
def get_alerts():
    from flask import jsonify
    return jsonify(alerts_log)

@app.route("/")
def index():
    return "Fall Detection Server Running! Open dashboard.html in your browser."

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start camera in background thread
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()

    print("[INFO] Starting web server on http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
