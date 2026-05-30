import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------- SETUP --------------------

VIDEO_PATH = "cctv.mp4"
CLIP_FOLDER = "clips"
os.makedirs(CLIP_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_buffer = []
BUFFER_SIZE = fps * 10  # 10 seconds buffer before event

# -------------------- DATABASE --------------------

conn = sqlite3.connect("events.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT,
    time TEXT,
    clip_path TEXT
)
""")
conn.commit()

# -------------------- HELPER FUNCTIONS --------------------

def boxes_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return xA < xB and yA < yB

def save_clip(frames, event_time):
    filename = f"{CLIP_FOLDER}/event_{event_time.replace(':','-')}.mp4"
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    return filename

# -------------------- MAIN LOOP --------------------

event_cooldown = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_buffer.append(frame.copy())
    if len(frame_buffer) > BUFFER_SIZE:
        frame_buffer.pop(0)

    results = model(frame)[0]

    detections = []
    bike_boxes = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "person":
            detections.append(([x1, y1, x2 - x1, y2 - y1], box.conf[0], "person"))

        if label in ["bicycle", "motorbike"]:
            bike_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())
        person_box = [l, t, l + w, t + h]

        cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --------- INTERACTION LOGIC ----------
        for bike in bike_boxes:
            if boxes_overlap(person_box, bike) and event_cooldown == 0:
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                event_time = str(datetime.now().time())[:8]

                print(f"[EVENT] Person {track_id} touched bike at {event_time}")

                # Save clip (10 sec before + 10 sec after)
                post_frames = []
                for _ in range(fps * 10):
                    ret2, f2 = cap.read()
                    if not ret2:
                        break
                    post_frames.append(f2)
                clip_frames = frame_buffer + post_frames

                clip_path = save_clip(clip_frames, event_time)

                cursor.execute(
                    "INSERT INTO events (event, time, clip_path) VALUES (?, ?, ?)",
                    ("person touched bike", event_time, clip_path)
                )
                conn.commit()

                event_cooldown = fps * 20  # avoid duplicate triggers

    if event_cooldown > 0:
        event_cooldown -= 1

    cv2.imshow("CCTV Investigator", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
conn.close()
cv2.destroyAllWindows()
