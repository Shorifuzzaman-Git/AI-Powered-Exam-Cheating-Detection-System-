import cv2
import time
import csv
import os
import math
from collections import deque

import numpy as np
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


FACE_MODEL = "models/face_landmarker.task"
# CAMERA_ID = 0

# YOLO
YOLO_CONFIDENCE = 0.50
YOLO_IOU = 0.50

# Head/Gaze suspicious behaviour duration
# Much shorter than the previous 2.5 seconds
BEHAVIOUR_DURATION = 1.5

# Head pose
HEAD_YAW_THRESHOLD = 15.0

# Gaze
GAZE_LEFT_THRESHOLD = 0.38
GAZE_RIGHT_THRESHOLD = 0.62

# Gaze smoothing
GAZE_HISTORY_SIZE = 7

# How long an alert remains visible
ALERT_DISPLAY_TIME = 2.0

# Person state cleanup
PERSON_TIMEOUT = 5.0

# Face -> person matching distance
FACE_MATCH_DISTANCE = 180

PERSON_CLASS = 0
PHONE_CLASS = 67


LOG_DIR = "logs"
LOG_FILE = os.path.join(
    LOG_DIR,
    "suspicious_events.csv"
)

os.makedirs(LOG_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):

    with open(
        LOG_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "person_id",
            "event",
            "duration_seconds"
        ])

yolo_model = YOLO("yolo26n.pt")

base_options = python.BaseOptions(
    model_asset_path=FACE_MODEL
)

face_options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=10,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1040)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,780)

person_states = {}

def create_person_state():
    return {
        "last_seen": 0.0,
        "last_face_seen": 0.0,

        "head": "UNKNOWN",
        "gaze": "UNKNOWN",
        "yaw": None,
        "pitch": None,
        "roll": None,

        "gaze_history": deque(
            maxlen=GAZE_HISTORY_SIZE
        ),

        "phone_detected": False,
        "behaviour_start": None,
        "behaviour_reason": "",

        "alert_until": 0.0,
        "alert_reason": "",
        "alert_duration": 0.0,

        "event_logged": False
    }

def center_of_box(box):
    x1, y1, x2, y2 = box
    return (
        int((x1 + x2) / 2),
        int((y1 + y2) / 2)
    )


def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def point_inside_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )

def get_eye_gaze_ratio(face,left_corner_id,right_corner_id,iris_id):
    left_corner = face[left_corner_id]
    right_corner = face[right_corner_id]
    iris = face[iris_id]

    eye_width = (right_corner.x -left_corner.x)

    if abs(eye_width) < 0.000001:
        return 0.5

    ratio = (iris.x -left_corner.x) / eye_width
    return ratio


def calculate_gaze(face,state):
    left_ratio = get_eye_gaze_ratio(face,33,133,468)
    right_ratio = get_eye_gaze_ratio(face,362,263,473)
    gaze_ratio = (left_ratio +right_ratio) / 2.0

    gaze_ratio = max(0.0,min(1.0, gaze_ratio))

    state["gaze_history"].append(gaze_ratio)

    smoothed_ratio = np.mean(state["gaze_history"])

    if smoothed_ratio < GAZE_LEFT_THRESHOLD:
        direction = "LEFT"

    elif smoothed_ratio > GAZE_RIGHT_THRESHOLD:
        direction = "RIGHT"

    else:
        direction = "CENTER"

    return direction, smoothed_ratio


def calculate_head_pose(face,width,height):
    face_3d = np.array([
        [0.0, 0.0, 0.0],# Nose
        [0.0, -330.0, -65.0],# Chin
        [-225.0, 170.0, -135.0],# Left eye
        [225.0, 170.0, -135.0],# Right eye
        [-150.0, -150.0, -125.0],# Left mouth
        [150.0, -150.0, -125.0]# Right mouth
    ], dtype=np.float64)


    face_2d = np.array([
        (face[1].x * width,face[1].y * height),
        (face[152].x * width,face[152].y * height),
        (face[33].x * width,face[33].y * height),
        (face[263].x * width,face[263].y * height),
        (face[61].x * width,face[61].y * height),
        (face[291].x * width,face[291].y * height)
    ], dtype=np.float64)

    focal_length = width

    camera_matrix = np.array([
        [focal_length,0,width / 2],
        [0,focal_length,height / 2],
        [0,0,1]

    ], dtype=np.float64)


    dist_coeffs = np.zeros((4, 1),dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        face_3d,
        face_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, None

    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(
        rotation_matrix
    )

    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]

    return pitch, yaw, roll


def get_head_direction(yaw):
    if yaw is None:
        return "UNKNOWN"

    if yaw < -HEAD_YAW_THRESHOLD:
        return "LEFT"

    if yaw > HEAD_YAW_THRESHOLD:
        return "RIGHT"

    return "CENTER"

def draw_face_landmarks(frame,face):
    h, w = frame.shape[:2]

    important_points = [
        # Nose
        1,
        # Chin
        152,
        # Left eye
        33,
        133,
        # Right eye
        362,
        263,
        # Left iris
        468,
        # Right iris
        473,
        # Mouth
        61,
        291
    ]

    for landmark_id in important_points:
        landmark = face[landmark_id]

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        cv2.circle(
            frame,
            (x, y),
            3,
            (0, 255, 255),
            -1
        )


def assign_face_to_person(face_center,people):
    best_person_id = None
    best_distance = float("inf")

    for person_id, person in people.items():
        person_box = person["box"]

        if point_inside_box(face_center,person_box):
            return person_id

        person_center = center_of_box(person_box)
        current_distance = distance(face_center,person_center)

        if current_distance < best_distance:
            best_distance = current_distance
            best_person_id = person_id

    if best_distance <= FACE_MATCH_DISTANCE:
        return best_person_id
    
    return None


def log_event(person_id,reason,duration):
    timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(LOG_FILE,"a",newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            person_id,
            reason,
            round(duration, 2)
        ])


def trigger_phone_alert(person_id,state,current_time):
    state["alert_until"] = (current_time +ALERT_DISPLAY_TIME)
    state["alert_reason"] = ("PHONE DETECTED")
    state["alert_duration"] = 0.0

    if not state["event_logged"]:
        log_event(
            person_id,
            "PHONE DETECTED",
            0.0
        )

        state["event_logged"] = True


def process_faces(frame,people,current_time):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return


    for face in result.face_landmarks:
        nose = face[1]
        face_center = (
            int(nose.x * w),
            int(nose.y * h)
        )

        person_id = assign_face_to_person(face_center,people)

        if person_id is None:
            continue

        if person_id not in person_states:
            person_states[
                person_id
            ] = create_person_state()


        state = person_states[person_id]

        state["last_face_seen"] = (current_time)

        gaze, gaze_ratio = calculate_gaze(face,state)
        pitch, yaw, roll = calculate_head_pose(face,w,h)
        head = get_head_direction(yaw)


        state["gaze"] = gaze
        state["head"] = head
        state["yaw"] = yaw
        state["pitch"] = pitch
        state["roll"] = roll

        draw_face_landmarks(frame,face)

        cv2.circle(
            frame,
            face_center,
            5,
            (255, 255, 0),
            -1
        )


        tx = face_center[0] + 10
        ty = face_center[1]

        cv2.putText(
            frame,
            f"ID: {person_id}",
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Head: {head}",
            (tx, ty + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Gaze: {gaze}",
            (tx, ty + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )


        cv2.putText(
            frame,
            f"Gaze ratio: {gaze_ratio:.2f}",
            (tx, ty + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1
        )


def update_behavior(person_id,state,current_time):
    if state["phone_detected"]:
        trigger_phone_alert(
            person_id,
            state,
            current_time
        )

        state["behaviour_start"] = None
        state["behaviour_reason"] = ""

        return

    head = state["head"]
    gaze = state["gaze"]

    suspicious = False
    reason = ""

    if (head == "LEFT"and gaze == "LEFT"):
        suspicious = True
        reason = ("HEAD + EYES LEFT")

    elif (head == "RIGHT"and gaze == "RIGHT"):
        suspicious = True
        reason = ("HEAD + EYES RIGHT")

    elif head in ["LEFT","RIGHT"]:
        suspicious = True
        reason = (f"HEAD {head}")

    elif gaze in ["LEFT","RIGHT"]:
        suspicious = True
        reason = (f"EYES {gaze}")


    if not suspicious:
        state["behaviour_start"] = None
        state["behaviour_reason"] = ""
        state["event_logged"] = False
        return


    if state["behaviour_start"] is None:
        state["behaviour_start"] = (current_time)
        state["behaviour_reason"] = (reason)
        state["event_logged"] = False

        return


    duration = (current_time -state["behaviour_start"])

    if duration >= BEHAVIOUR_DURATION:

        state["alert_until"] = (current_time + ALERT_DISPLAY_TIME)
        state["alert_reason"] = (state["behaviour_reason"])
        state["alert_duration"] = (duration)


        if not state["event_logged"]:
            log_event(
                person_id,
                state["behaviour_reason"],
                duration
            )
            state["event_logged"] = True


def draw_person(frame,person_id,person,state,current_time):
    x1, y1, x2, y2 = person["box"]
    confidence = person["confidence"]


    alert_active = (state["alert_until"] >current_time)

    if alert_active:
        box_color = (0,0,255)

    else:
        box_color = (0,255,0)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        box_color,
        2
    )

    cv2.putText(
        frame,
        f"Person {person_id} | {confidence:.2f}",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        box_color,
        2
    )

    if state["phone_detected"]:
        cv2.putText(
            frame,
            "PHONE DETECTED",
            (x1, y2 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2
        )

    cv2.putText(
        frame,
        f"Head: {state['head']}",
        (x1, y2 + 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Gaze: {state['gaze']}",
        (x1, y2 + 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    if (state["behaviour_start"] is not None):
        duration = (current_time - state["behaviour_start"])

        cv2.putText(
            frame,
            f"Suspicious: {duration:.2f}s",
            (x1, y2 + 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            2
        )

    if alert_active:

        cv2.rectangle(
            frame,
            (x1 - 6, y1 - 6),
            (x2 + 6, y2 + 6),
            (0, 0, 255),
            5
        )

        cv2.putText(
            frame,
            "CHEATING DETECTED",
            (x1, max(55, y1 - 38)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            state["alert_reason"],
            (x1, max(80, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2
        )



previous_time = time.perf_counter()
fps = 0.0


while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read webcam frame.")
        break

    frame = cv2.flip(frame,1)
    current_time = time.perf_counter()

    elapsed = (current_time -previous_time)

    previous_time = current_time

    if elapsed > 0:
        instant_fps = 1.0 / elapsed
        if fps == 0:
            fps = instant_fps

        else:
            fps = (0.90 * fps +0.10 * instant_fps)

    results = yolo_model.track(
        frame,
        persist=True,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_IOU,
        classes=[PERSON_CLASS,PHONE_CLASS],
        verbose=False
    )
    people = {}
    phones = []

    if (results and results[0].boxes is not None):
        boxes = results[0].boxes

        for i in range(len(boxes)):
            box = boxes[i]
            cls = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            coordinates = (box.xyxy[0].cpu().numpy())
            x1, y1, x2, y2 = map(int,coordinates)

            if cls == PERSON_CLASS:
                if box.id is not None:
                    person_id = int(
                        box.id[0].item()
                    )

                else:
                    person_id = i


                people[person_id] = {
                    "box": (x1,y1,x2,y2),
                    "confidence": confidence
                }

            elif cls == PHONE_CLASS:
                phones.append({
                    "box": (x1,y1,x2,y2),
                    "confidence": confidence
                })

    for person_id in people:
        if person_id not in person_states:
            person_states[person_id] = create_person_state()

        person_states[person_id]["last_seen"] = current_time

        # Reset phone status
        person_states[person_id]["phone_detected"] = False


    for phone in phones:
        phone_box = phone["box"]
        phone_center = center_of_box(phone_box)

        best_person = None
        best_distance = float("inf")

        for person_id, person in people.items():
            person_box = person["box"]

            if point_inside_box(phone_center,person_box):
                best_person = person_id
                break

            person_center = center_of_box(person_box)

            current_distance = distance(phone_center,person_center)

            if current_distance < best_distance:
                best_distance = (current_distance)
                best_person = person_id

        if best_person is not None:
            person_states[best_person]["phone_detected"] = True

        px1, py1, px2, py2 = phone_box

        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            (0, 165, 255),
            2
        )


        cv2.putText(
            frame,
            f"PHONE {phone['confidence']:.2f}",
            (px1, max(20, py1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 165, 255),
            2
        )

    process_faces(frame,people,current_time)

    for person_id in people:
        state = person_states[person_id]

        update_behavior(person_id,state,current_time)

    for person_id, person in people.items():
        state = person_states[person_id]

        draw_person(frame,person_id,person,state,current_time)

    old_ids = []

    for person_id, state in person_states.items():
        if (current_time - state["last_seen"]) > PERSON_TIMEOUT:
            old_ids.append(person_id)

    for person_id in old_ids:
        del person_states[person_id]


    # ========================================================
    # GLOBAL INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (255, 75),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Persons: {len(people)}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "AI EXAM MONITORING",
        (285, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI Exam Monitoring",frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
face_landmarker.close()
