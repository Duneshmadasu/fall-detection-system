# 🛡️ Fall Detection System — Setup Guide

## Project Structure
```
fall_detection/
├── backend/
│   └── fall_detector.py        ← Main detection engine (YOLOv8 + OpenCV)
├── alerts/
│   └── firebase_alert.py       ← Firebase + FCM notifications
├── frontend/
│   └── src/Dashboard.jsx       ← React web dashboard
├── models/                     ← YOLOv8 model saved here automatically
├── requirements.txt
└── README.md
```

---

## STEP 1 — Install Python & Dependencies

### Install Python 3.10+
Download from: https://www.python.org/downloads/

### Install project dependencies
```bash
pip install -r requirements.txt
```

This installs: YOLOv8 (ultralytics), OpenCV, Firebase Admin SDK, and more.

---

## STEP 2 — Download YOLOv8 Pose Model

The model downloads **automatically** on first run. But you can also do it manually:

```python
from ultralytics import YOLO
model = YOLO("yolov8n-pose.pt")  # downloads ~6MB model
```

Save it inside the `models/` folder.

---

## STEP 3 — Setup Firebase

1. Go to https://console.firebase.google.com
2. Create a new project (e.g., "FallDetectionSystem")
3. Enable **Realtime Database** (start in test mode)
4. Go to Project Settings → Service Accounts → Generate new private key
5. Save the downloaded JSON file as `firebase_credentials.json` in the project root
6. Copy your Database URL (looks like: `https://your-project-default-rtdb.firebaseio.com`)

### Set environment variables
Create a `.env` file in the project root:
```
FIREBASE_CRED_PATH=firebase_credentials.json
FIREBASE_DB_URL=https://your-project-default-rtdb.firebaseio.com
```

---

## STEP 4 — Run the Fall Detection Backend

```bash
cd fall_detection
python backend/fall_detector.py
```

- Opens your **webcam** automatically
- Detects people and analyzes poses in real time
- Confirms fall if person is horizontal + inactive for 2 seconds
- Sends alert to Firebase + logs locally

**To use a video file instead of webcam:**
```python
detector.run(source="path/to/your/video.mp4")
```

---

## STEP 5 — Run the Web Dashboard

```bash
# Install Node.js from https://nodejs.org first, then:
npx create-react-app fall-dashboard
cd fall-dashboard
# Copy Dashboard.jsx into src/
npm start
```

Opens at http://localhost:3000

---

## How Fall Detection Works

```
Camera Feed
    ↓
OpenCV (frame extraction)
    ↓
YOLOv8 Pose Model
    ↓ (17 body keypoints detected)
Fall Logic Check:
  - Body aspect ratio > 0.8 (wider than tall = lying down)?
  - Shoulders at same height as hips (horizontal body)?
    ↓ YES (possible fall)
Wait 2 seconds of inactivity
    ↓ CONFIRMED
Firebase Alert sent → Push notification to phone/dashboard
```

---

## Performance (from paper)
| Method              | Accuracy | Recall | F1    |
|---------------------|----------|--------|-------|
| Threshold-Based     | 70–78%   | 60–72% | 65–74% |
| Wearable Sensor     | 80–88%   | 75–86% | 78–85% |
| CNN-Based           | 85–90%   | 82–89% | 84–88% |
| **YOLOv8 + Dual Verify** | **91–94%** | **90–95%** | **91–94%** |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Webcam not opening | Change `CAMERA_SOURCE = 1` or `2` |
| Firebase not connecting | Check credentials JSON path in .env |
| Model not found | Run `YOLO("yolov8n-pose.pt")` once to auto-download |
| Slow performance | Use a smaller model: `yolov8n-pose.pt` (nano = fastest) |
| Too many false alerts | Increase `INACTIVITY_SECONDS` to 3–4 |

---

## Next Steps (Future Enhancements)
- [ ] Add Twilio SMS alerts
- [ ] Multi-camera support
- [ ] Mobile app (React Native)
- [ ] Night vision / IR camera support
- [ ] Edge deployment (Raspberry Pi / Jetson Nano)
