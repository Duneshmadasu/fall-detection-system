"""
Fall Detection Engine
Uses YOLOv8 + OpenCV for real-time fall detection from webcam/video
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from alerts.firebase_alert import send_fall_alert

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH = "models/yolov8n-pose.pt"   # pose estimation model
CONFIDENCE_THRESHOLD = 0.5
FALL_ASPECT_RATIO_THRESHOLD = 0.8       # width/height > this = likely fallen
INACTIVITY_SECONDS = 2.0                # seconds of stillness after fall to confirm
CAMERA_SOURCE = 0                        # 0 = webcam, or path to video file

# Keypoint indices (COCO format used by YOLOv8 pose)
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


class FallDetector:
    def __init__(self):
        print("[INFO] Loading YOLOv8 pose model...")
        self.model = YOLO(MODEL_PATH)
        self.fall_candidates = {}   # track_id -> fall_start_time
        self.alert_cooldown = {}    # track_id -> last_alert_time
        self.ALERT_COOLDOWN_SECS = 30  # don't re-alert same person within 30s
        print("[INFO] Model loaded successfully.")

    def is_fallen(self, keypoints):
        """
        Determine if a person has fallen based on pose keypoints.
        Logic:
          1. Body bounding box is wider than it is tall (lying down)
          2. Shoulders are near hip level (horizontal body)
        """
        if keypoints is None or len(keypoints) < 17:
            return False

        kp = keypoints  # shape: (17, 3) -> x, y, confidence

        # Extract key points with confidence check
        def get_point(idx):
            x, y, conf = kp[idx]
            return (float(x), float(y)) if conf > 0.3 else None

        nose = get_point(NOSE)
        l_shoulder = get_point(LEFT_SHOULDER)
        r_shoulder = get_point(RIGHT_SHOULDER)
        l_hip = get_point(LEFT_HIP)
        r_hip = get_point(RIGHT_HIP)
        l_ankle = get_point(LEFT_ANKLE)
        r_ankle = get_point(RIGHT_ANKLE)

        # Method 1: Bounding box aspect ratio
        visible = [p for p in [nose, l_shoulder, r_shoulder, l_hip, r_hip, l_ankle, r_ankle] if p]
        if len(visible) < 3:
            return False

        xs = [p[0] for p in visible]
        ys = [p[1] for p in visible]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        if height == 0:
            return False

        aspect_ratio = width / height
        if aspect_ratio > FALL_ASPECT_RATIO_THRESHOLD:
            return True

        # Method 2: Shoulder-to-hip vertical relationship
        if l_shoulder and r_shoulder and l_hip and r_hip:
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            hip_y = (l_hip[1] + r_hip[1]) / 2
            vertical_diff = abs(shoulder_y - hip_y)
            horizontal_spread = abs(l_shoulder[0] - r_shoulder[0])

            # If shoulders and hips are at similar height -> lying down
            if vertical_diff < horizontal_spread * 0.5:
                return True

        return False

    def process_frame(self, frame):
        """
        Run YOLOv8 pose estimation on a frame and check for falls.
        Returns annotated frame + list of confirmed falls.
        """
        results = self.model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
        confirmed_falls = []
        current_time = time.time()

        if results[0].keypoints is None:
            return frame, confirmed_falls

        keypoints_data = results[0].keypoints.data.cpu().numpy()
        boxes = results[0].boxes

        for i, kp in enumerate(keypoints_data):
            track_id = int(boxes.id[i]) if boxes.id is not None else i

            fallen = self.is_fallen(kp)

            if fallen:
                # Start tracking fall time
                if track_id not in self.fall_candidates:
                    self.fall_candidates[track_id] = current_time
                    print(f"[DETECT] Person {track_id} possible fall, waiting to confirm...")

                elapsed = current_time - self.fall_candidates[track_id]

                if elapsed >= INACTIVITY_SECONDS:
                    # Confirmed fall - check cooldown before alerting
                    last_alert = self.alert_cooldown.get(track_id, 0)
                    if current_time - last_alert > self.ALERT_COOLDOWN_SECS:
                        confirmed_falls.append(track_id)
                        self.alert_cooldown[track_id] = current_time
                        print(f"[ALERT] FALL CONFIRMED for person {track_id}!")

            else:
                # Person recovered / was not fallen
                if track_id in self.fall_candidates:
                    del self.fall_candidates[track_id]

        # Draw annotations on frame
        annotated = results[0].plot()

        # Draw fall warning overlays
        for track_id in confirmed_falls:
            cv2.putText(annotated, f"FALL DETECTED! Person {track_id}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return annotated, confirmed_falls

    def run(self, source=CAMERA_SOURCE):
        """Main loop: capture video and run fall detection"""
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {source}")
            return

        print(f"[INFO] Starting fall detection... Press 'q' to quit.")
        fps_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video stream ended.")
                break

            annotated_frame, falls = self.process_frame(frame)
            frame_count += 1

            # Send alerts for confirmed falls
            for track_id in falls:
                send_fall_alert(track_id, frame)

            # FPS counter
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()
                print(f"[INFO] FPS: {fps:.1f}")

            cv2.imshow("Fall Detection System", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Fall detection stopped.")


if __name__ == "__main__":
    detector = FallDetector()
    detector.run()
