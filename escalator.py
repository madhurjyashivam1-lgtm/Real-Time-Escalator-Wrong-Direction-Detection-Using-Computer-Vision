import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------------------
# CONFIGURATION
# -------------------------------

VIDEO_PATH = "/home/pguha6/uma/codes/interns/escalator/anomaly-20260307T064214Z-3-001/anomaly/33_city_escalator_002.mov"
OUTPUT_PATH = "/home/pguha6/uma/codes/interns/escalator/output_anomaly2.mp4"

CONF_THRESHOLD = 0.4
TRACK_HISTORY = 6

# Escalator ROIs (adjust to your video)
LEFT_ESCALATOR  = (420, 40, 700, 900)
RIGHT_ESCALATOR = (700, 40, 980, 900)

# -------------------------------
# LOAD MODELS
# -------------------------------

detector = YOLO("/home/pguha6/uma/codes/interns/yolov8n.pt")
tracker = DeepSort(max_age=30)

# -------------------------------
# UTILITIES
# -------------------------------

def inside_roi(cx, cy, roi):
    x1,y1,x2,y2 = roi
    return x1 < cx < x2 and y1 < cy < y2


def estimate_direction(flow_roi):
    vx = np.mean(flow_roi[...,0])
    vy = np.mean(flow_roi[...,1])
    vec = np.array([vx,vy])
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


# -------------------------------
# VIDEO SETUP
# -------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width,height)
)

track_history = {}

prev_gray = None

left_dir = None
right_dir = None

# -------------------------------
# PROCESS VIDEO
# -------------------------------

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # Optical Flow
    # -------------------------------

    if prev_gray is not None:

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        # compute ROI flows
        x1,y1,x2,y2 = LEFT_ESCALATOR
        left_flow = flow[y1:y2, x1:x2]

        x1,y1,x2,y2 = RIGHT_ESCALATOR
        right_flow = flow[y1:y2, x1:x2]

        left_dir = estimate_direction(left_flow)
        right_dir = estimate_direction(right_flow)

    prev_gray = gray

    # -------------------------------
    # PERSON DETECTION
    # -------------------------------

    results = detector(frame)[0]

    detections = []

    for box in results.boxes:

        cls = int(box.cls[0])

        if cls != 0:
            continue

        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD:
            continue

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        detections.append(([x1,y1,x2-x1,y2-y1], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    # -------------------------------
    # TRACK ANALYSIS
    # -------------------------------

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l,t,r,b = map(int, track.to_ltrb())

        w = r-l
        h = b-t

        cx = l + w//2
        cy = t + h//2

        # determine escalator
        if inside_roi(cx,cy,LEFT_ESCALATOR):
            escalator_dir = left_dir
        elif inside_roi(cx,cy,RIGHT_ESCALATOR):
            escalator_dir = right_dir
        else:
            continue

        if escalator_dir is None:
            continue

        # track history
        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append((cx,cy))

        if len(track_history[track_id]) > TRACK_HISTORY:
            track_history[track_id].pop(0)

        color = (0,255,0)

        # compute trajectory direction
        if len(track_history[track_id]) >= 2:

            dx = track_history[track_id][-1][0] - track_history[track_id][0][0]
            dy = track_history[track_id][-1][1] - track_history[track_id][0][1]

            motion = np.array([dx,dy])
            norm = np.linalg.norm(motion)

            if norm > 0:
                motion = motion / norm

                dot = np.dot(motion, escalator_dir)

                if dot < -0.25:

                    color = (0,0,255)

                    cv2.putText(
                        frame,
                        "WRONG DIRECTION",
                        (l,t-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,255),
                        2
                    )

        cv2.rectangle(frame,(l,t),(r,b),color,2)

    # draw ROI for visualization
    cv2.rectangle(frame,(LEFT_ESCALATOR[0],LEFT_ESCALATOR[1]),
                  (LEFT_ESCALATOR[2],LEFT_ESCALATOR[3]),(255,0,0),2)

    cv2.rectangle(frame,(RIGHT_ESCALATOR[0],RIGHT_ESCALATOR[1]),
                  (RIGHT_ESCALATOR[2],RIGHT_ESCALATOR[3]),(255,0,0),2)

    writer.write(frame)

# -------------------------------
# CLEANUP
# -------------------------------

cap.release()
writer.release()

print("Finished. Output saved to:", OUTPUT_PATH)