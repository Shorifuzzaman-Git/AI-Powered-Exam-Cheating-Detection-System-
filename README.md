# AI-Powered Exam Monitoring & Suspicious Activity Detection System

An AI-powered real-time exam monitoring system that uses **Computer Vision, YOLO, OpenCV, and MediaPipe** to detect students, prohibited objects, facial behavior, head movement, gaze direction, and hand movement during examinations.

The system combines multiple visual signals with **temporal behavior analysis** to identify potentially suspicious activities and generate real-time alerts.

---

## 🚀 Features

* 👤 **Student Detection & Tracking**

  * Detect students using YOLO.
  * Track individual students using person IDs.

* 📱 **Prohibited Object Detection**

  * Detect mobile phones and other unauthorized objects.
  * Generate alerts when prohibited objects are detected.

* 👁️ **Gaze Direction Detection**

  * Analyze eye landmarks using MediaPipe.
  * Detect:

    * Left
    * Right
    * Center

* 🧑 **Head Pose Estimation**

  * Estimate head orientation using facial landmarks.
  * Detect:

    * Looking Left
    * Looking Right
    * Looking Center

* 🚨 **Suspicious Activity Detection**

  * Combine multiple visual signals.
  * Analyze behavior over time instead of relying on a single frame.

* ⏱️ **Temporal Behavior Analysis**

  * Reduce false alerts from short or normal movements.
  * Detect behaviors that continue for a specific duration.

* 📝 **Event Logging**

  * Store suspicious events with:

    * Timestamp
    * Person ID
    * Event type
    * Duration

* 🎥 **Real-Time Monitoring**

  * Process webcam video in real time.
  * Display detection results and alerts directly on the screen.

---

## 🧠 Technologies Used

| Technology             | Purpose                                |
| ---------------------- | -------------------------------------- |
| **Python**             | Core programming language              |
| **OpenCV**             | Image and video processing             |
| **YOLO / Ultralytics** | Student and object detection           |
| **MediaPipe**          | Face, eye, and hand landmark detection |
| **NumPy**              | Numerical and coordinate processing    |

---

## 🏗️ System Architecture

```text
                    Webcam
                       │
                       ↓
              ┌─────────────────┐
              │  Video Capture  │
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              ↓                 ↓
            YOLO            MediaPipe
              │                 │
       ┌──────┴──────┐    ┌─────┴───────────┐
       ↓             ↓    ↓                 ↓
    Person        Objects Face            Gaze
    Tracking      Phone   │                 │
         ↓                ↓                 ↓
                       Head Pose         Movement
                           │                │
                           └────────────────┘
                                   │
                                   ↓
                        Temporal Analysis
                                   │
                                   ↓
                       Behavior Detection
                                   │
                                   ↓
                       Suspicion / Alert
                                   │
                    ┌──────────────┴──────────────┐
                    ↓                             ↓
              Real-Time Alert               Event Logging
```

---

## 📂 Project Structure

```text
Real-time-exam-cheating-detection-system/
│
├── models/
│   ├── face_landmarker.task
│
├── logs/
│   └── suspicious_events.csv
│
├── main.py
├── test.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shorifuzzaman-Git/ai-powered-exam-cheating-detection-system.git
```

### 2. Enter the project directory

```bash
cd Real-time-exam-cheating-detection-system
```

### 3. Create a virtual environment

```bash
python3 -m venv venv
```

### 4. Activate the virtual environment

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Required Models

### MediaPipe Face Landmarker

Place:

```text
models/face_landmarker.task
```

---

## ▶️ Run the Project

Run the main exam monitoring system:

```bash
python main.py
```

Press:

```text
q
```

to exit the webcam window.

---


## 🚨 Suspicious Activity Logic

The system does not treat every movement as cheating.

Instead, multiple signals can be combined:

```text
Phone detected
      ↓
Immediate Alert

Head turned
      +
Gaze away
      +
Behavior continues
      ↓
Potentially Suspicious
```

This temporal approach helps reduce false alerts caused by normal short-term movements.

---

## 📊 Event Logging

Detected events can be stored in:

```text
logs/suspicious_events.csv
```

Example:

```csv
timestamp,person_id,event,duration_seconds
2026-09-20 00:15:32,1,PHONE_DETECTED,0.00
2026-09-20 00:16:10,2,HEAD_TURNED_RIGHT,1.42
2026-09-20 00:17:04,1,FAST_HAND_MOVEMENT,0.00
```

---

## 🔮 Future Improvements

* [ ] MediaPipe Pose integration
* [ ] Student-to-student interaction detection
* [ ] Passing object detection
* [ ] Cheat-sheet/paper detection
* [ ] Earphone detection
* [ ] Hand-under-desk detection
* [ ] Action recognition using video sequences
* [ ] Suspicion scoring system
* [ ] FastAPI backend
* [ ] Real-time web dashboard
* [ ] Database integration
* [ ] Alert notification system
* [ ] Multi-camera support
* [ ] Custom exam behavior dataset
* [ ] Model performance evaluation

---

## ⚠️ Disclaimer

This project is intended as an **AI-powered exam cheating detection system**. Detected visual behaviors represent potentially suspicious activity and should not be treated as definitive proof of academic misconduct. Human review should be used for final decisions.

---

## 👨‍💻 Author

**Md.Shorifuzzaman**

Software Engineering Student
Interested in **AI Engineering, Computer Vision, Deep Learning, and Image Processing**.
