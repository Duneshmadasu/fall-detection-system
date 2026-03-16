"""
Firebase Alert Module
Sends real-time fall alerts to Firebase Realtime Database + FCM push notifications
"""

import firebase_admin
from firebase_admin import credentials, db, messaging
import datetime
import base64
import cv2
import os
from dotenv import load_dotenv
load_dotenv()
import datetime

# Twilio credentials
# Twilio credentials (loaded from .env file)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
TWILIO_TO_NUMBERS = os.getenv("TWILIO_TO_NUMBERS", "").split(",")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")


# ─── INIT ─────────────────────────────────────────────────────────────────────
_firebase_initialized = False

def init_firebase():
    global _firebase_initialized
    if _firebase_initialized:
        return

    cred_path = os.getenv("FIREBASE_CRED_PATH", "firebase_credentials.json")

    if not os.path.exists(cred_path):
        print(f"[WARNING] Firebase credentials not found at {cred_path}")
        print("[WARNING] Alerts will be logged locally only.")
        return

    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        "databaseURL": os.getenv("FIREBASE_DB_URL", "https://falldetectionsystem-b442c-default-rtdb.asia-southeast1.firebasedatabase.app")
    })
    _firebase_initialized = True
    print("[INFO] Firebase initialized successfully.")


def encode_frame(frame):
    """Convert OpenCV frame to base64 string for storage"""
    if frame is None:
        return None
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode("utf-8")


def send_fall_alert(person_id, frame=None, location="Camera-1"):
    """
    Send fall alert to Firebase and push notification.
    
    Args:
        person_id: Tracked person ID
        frame: OpenCV frame at time of fall (optional)
        location: Camera/location identifier
    """
    init_firebase()

    timestamp = datetime.datetime.now().isoformat()
    alert_id = f"fall_{person_id}_{int(datetime.datetime.now().timestamp())}"

    alert_data = {
        "alert_id": alert_id,
        "person_id": str(person_id),
        "timestamp": timestamp,
        "location": location,
        "status": "unacknowledged",
        "snapshot": encode_frame(frame) if frame is not None else None
    }

    # 1. Log to local file (always works even without Firebase)
    log_alert_locally(alert_data)
    send_sms_alert(person_id, timestamp, location)

    # 2. Write to Firebase Realtime Database
    if _firebase_initialized:
        try:
            ref = db.reference(f"fall_alerts/{alert_id}")
            ref.set(alert_data)
            print(f"[FIREBASE] Alert logged: {alert_id}")
        except Exception as e:
            print(f"[ERROR] Firebase DB write failed: {e}")

        # 3. Send FCM Push Notification
        try:
            send_push_notification(alert_id, person_id, timestamp, location)
        except Exception as e:
            print(f"[ERROR] FCM notification failed: {e}")
    else:
        print(f"[LOCAL ALERT] Fall detected - Person {person_id} at {timestamp} ({location})")


def send_push_notification(alert_id, person_id, timestamp, location):
    """Send FCM push notification to all subscribed devices"""
    message = messaging.Message(
        notification=messaging.Notification(
            title="⚠️ FALL DETECTED",
            body=f"Person {person_id} has fallen at {location}. Immediate attention required!"
        ),
        data={
            "alert_id": alert_id,
            "person_id": str(person_id),
            "timestamp": timestamp,
            "location": location
        },
        topic="fall_alerts"  # All devices subscribed to this topic receive it
    )

    response = messaging.send(message)
    print(f"[FCM] Notification sent: {response}")


def log_alert_locally(alert_data):
    """Fallback: save alert to local JSON log file"""
    import json

    log_file = "fall_alerts_log.json"
    logs = []

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                logs = json.load(f)
            except:
                logs = []

    # Don't store base64 image in local log (too large)
    log_entry = {k: v for k, v in alert_data.items() if k != "snapshot"}
    logs.append(log_entry)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"[LOG] Alert saved locally: {alert_data['alert_id']}")


def acknowledge_alert(alert_id, acknowledged_by="staff"):
    """Mark a fall alert as acknowledged in Firebase"""
    if not _firebase_initialized:
        return

    try:
        ref = db.reference(f"fall_alerts/{alert_id}")
        ref.update({
            "status": "acknowledged",
            "acknowledged_by": acknowledged_by,
            "acknowledged_at": datetime.datetime.now().isoformat()
        })
        print(f"[FIREBASE] Alert {alert_id} acknowledged by {acknowledged_by}")
    except Exception as e:
        print(f"[ERROR] Failed to acknowledge alert: {e}")
        
def send_sms_alert(person_id, timestamp, location):
    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        for number in TWILIO_TO_NUMBERS:
            # Send SMS
            sms = client.messages.create(
                body=f"⚠️ FALL DETECTED!\nPerson {person_id} fell at {location}\nTime: {timestamp}\nPlease respond immediately!",
                from_=TWILIO_FROM_NUMBER,
                to=number
            )
            print(f"[SMS] Sent to {number}: {sms.sid}")

            # Send WhatsApp
            whatsapp = client.messages.create(
                body=f"⚠️ *FALL DETECTED!*\n👤 Person {person_id} has fallen\n📍 Location: {location}\n🕐 Time: {timestamp}\n🚨 Please respond immediately!",
                from_=TWILIO_WHATSAPP_FROM,
                to=f"whatsapp:{number}"
            )
            print(f"[WHATSAPP] Sent to {number}: {whatsapp.sid}")

    except Exception as e:
        print(f"[ERROR] Notification failed: {e}")