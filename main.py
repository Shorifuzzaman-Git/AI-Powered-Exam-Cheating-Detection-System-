"""
ExamWatch - Real-time exam behavior monitoring using YOLO11 Pose Estimation
-----------------------------------------------------------------------------
No training required - uses a PRETRAINED pose model (knows human keypoints
out of the box) plus simple geometry to classify head orientation.

Detects, per student, per frame:
    - "Focused"          : facing forward or looking down (reading/writing)
    - "Looking Sideways"  : head turned significantly left/right

Also tracks each person across frames with a persistent ID (via Ultralytics'
built-in ByteTrack) and counts how many times each person has been flagged,
which is useful for telling "glanced once" apart from "kept looking away".

Install once:
    pip install ultralytics opencv-python

Run on your webcam:
    python examwatch.py --source 0

Run on a video file, save annotated output:
    python examwatch.py --source classroom.mp4 --output result.mp4

Press 'q' to quit while a window is open.
"""

import argparse
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

# ============================================================
# CONFIG / THRESHOLDS (tune after testing on yourself/your footage)
# ============================================================
# How far (as a fraction of shoulder width) the nose can drift sideways
# from the shoulder midpoint before we call it "Looking Sideways".
YAW_RATIO_THRESHOLD = 0.55

# Minimum confidence for a keypoint to be trusted (YOLO pose gives a
# per-keypoint confidence score, e.g. an occluded ear will score low).
KEYPOINT_CONF_THRESHOLD = 0.4

# How many times a track must be flagged "Looking Sideways" (cumulative,
# not necessarily consecutive) before we escalate its on-screen label.
SUSPICIOUS_STREAK_THRESHOLD = 8

# COCO pose keypoint indices (what YOLO11-pose outputs, in this order)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 0, 1, 2, 3, 4, 5, 6


# ============================================================
# HELPER: classify one person's head orientation from their keypoints
# ============================================================
def classify_orientation(keypoints_xy, keypoints_conf):
    """
    keypoints_xy: (17, 2) array of x,y pixel coords
    keypoints_conf: (17,) array of per-keypoint confidence
    Returns: (label_str, yaw_ratio_or_None)
    """
    nose_conf = keypoints_conf[NOSE]
    l_sh_conf = keypoints_conf[L_SHOULDER]
    r_sh_conf = keypoints_conf[R_SHOULDER]

    # Need nose + both shoulders visible to compute a reliable yaw estimate
    if nose_conf < KEYPOINT_CONF_THRESHOLD or l_sh_conf < KEYPOINT_CONF_THRESHOLD or r_sh_conf < KEYPOINT_CONF_THRESHOLD:
        return "Focused", None  # not enough info - default to non-alerting

    nose_x = keypoints_xy[NOSE][0]
    l_sh_x = keypoints_xy[L_SHOULDER][0]
    r_sh_x = keypoints_xy[R_SHOULDER][0]

    shoulder_mid_x = (l_sh_x + r_sh_x) / 2
    shoulder_width = abs(l_sh_x - r_sh_x)

    if shoulder_width < 1e-3:
        return "Focused", None

    yaw_ratio = (nose_x - shoulder_mid_x) / (shoulder_width / 2)

    if abs(yaw_ratio) > YAW_RATIO_THRESHOLD:
        return "Looking Sideways", yaw_ratio
    return "Focused", yaw_ratio


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ExamWatch pose-based exam monitor")
    parser.add_argument("--source", default="0",
                         help="Webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--model", default="yolo11n-pose.pt",
                         help="Pose model: yolo11n-pose.pt (fastest, CPU-friendly) "
                              "or yolo11m-pose.pt (more accurate, needs a GPU for real-time)")
    parser.add_argument("--output", default=None,
                         help="Optional path to save the annotated video, e.g. result.mp4")
    args = parser.parse_args()

    # Webcam index is passed as a string from argparse; convert if numeric
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)  # auto-downloads weights on first run

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: could not open source '{source}'")
        return

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    suspicious_counts = defaultdict(int)  # track_id -> cumulative sideways flags
    prev_time = time.time()

    print("Running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        frame=cv2.resize(frame,(1040,780))
        if not ret:
            break  # end of video file, or webcam disconnected

        # persist=True keeps the same ID for the same person across frames
        results = model.track(frame, persist=True, verbose=False)

        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue

            boxes = r.boxes
            all_kpts_xy = r.keypoints.xy.cpu().numpy()      # (num_people, 17, 2)
            all_kpts_conf = r.keypoints.conf                # (num_people, 17) or None
            track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                track_id = track_ids[i] if track_ids is not None else i

                kpts_xy = all_kpts_xy[i]
                kpts_conf = all_kpts_conf[i].cpu().numpy() if all_kpts_conf is not None else [1.0] * 17

                label, yaw_ratio = classify_orientation(kpts_xy, kpts_conf)

                if label == "Looking Sideways":
                    suspicious_counts[track_id] += 1
                    color = (0, 0, 255)  # red
                    if suspicious_counts[track_id] >= SUSPICIOUS_STREAK_THRESHOLD:
                        label = "Repeated Looking Away"
                else:
                    color = (0, 200, 0)  # green

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                tag = f"ID {track_id}: {label}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
                cv2.putText(frame, tag, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                # Optional: draw the keypoints used for this decision
                for idx in (NOSE, L_SHOULDER, R_SHOULDER):
                    if kpts_conf[idx] >= KEYPOINT_CONF_THRESHOLD:
                        px, py = int(kpts_xy[idx][0]), int(kpts_xy[idx][1])
                        cv2.circle(frame, (px, py), 3, (255, 255, 0), -1)

        # FPS overlay
        now = time.time()
        fps_display = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now
        cv2.putText(frame, f"FPS: {fps_display:.0f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("ExamWatch - press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()