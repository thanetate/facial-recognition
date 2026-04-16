### Facial Recognition System

CSCE 4240

Option A – face recognition from videos

Build a system that acquires a video (or images) and detects the person in the scope of view. Imagine a
camera installed at an entrance captures videos of people entering and leaving the premises. The
software system will detect the person registered in the system and alert a stranger who is not registered
in the system. You may include more specific details to make the problem more challenging or easier.
Software functions you may consider include detecting moving object(s), isolating the moving objects
from the background, cropping the human face, and recognizing the person (if registered).

#### Idea

Campus Lab Entry Monitor
A fixed camera at a lab entrance detects faces in incoming video, identifies registered students/staff, and flags unknown people in near real time. The system logs time, identity (or unknown), and saves alert snapshots for a human review.

Temporal identity smoothing and unknown-alert reliability
Instead of deciding from a single frame, track a person for a short window and vote across frames before triggering known/unknown. This greatly reduces false alerts from blur, bad angles, and lighting changes.

Basic: detect face, compare with registered database, alert unknown.
Advanced: robust multi-frame decision logic with confidence thresholds and cooldown rules.

#### File Structure

1. main.py
   Video loop
   Face detection
   Face recognition
   Unknown alert
   Event logging

2. enroll.py
   Add registered users from images
   Build/save the local face-recognition model

Plus folders:
data/known_faces/ (training images by person)
data/model.yml (saved face-recognition model)
data/labels.json (label map from IDs to names)
logs/events.csv and alerts/ (logs and unknown snapshots)

#### Data Storage Plan (Local Only)

No cloud database is required for this project. All data stays local:

1. Face model: `data/model.yml`
   Stores the trained local face-recognition model.
2. Label map: `data/labels.json`
   Stores the numeric ID to person-name mapping.
3. Event log: `logs/events.csv`
   Stores timestamp, predicted label, and confidence score.
4. Alert images: `alerts/`
   Stores snapshots for unknown-person detections.

This keeps the project simple, offline, and easy to demo.

#### Basic Requirement

1. Read video input (webcam or video file).
2. Detect faces in each frame.
3. Compare each detected face against registered people.
4. Decide: known person or unknown person.
5. Show result on screen (name or UNKNOWN).
6. Trigger a simple alert for unknown (console message + saved snapshot).
7. Log events (time, label, confidence) to a CSV file.

#### Day 1

Goal: verify the core pipeline starts correctly by opening a video source and detecting faces.

Files used:

1. `main.py` - opens webcam/video and draws face boxes.
2. `requirements.txt` - minimal dependency list.

Dependencies needed before running:

1. Python 3.10+ (3.13 is also fine)
2. OpenCV with face-recognition support (`opencv-contrib-python`)
3. NumPy (`numpy`)

Install dependencies (no virtual environment):

1. `python3 -m pip install --user -r requirements.txt`
2. If you see "externally-managed-environment" on macOS/Homebrew Python, use:
   `python3 -m pip install --user --break-system-packages -r requirements.txt`

Run steps:

0. Train the local model with:
   `python3 enroll.py` or `.venv/bin/python enroll.py`

1. Webcam:
   `python3 main.py --source 0`or ` .venv/bin/python main.py --source 0`

2. Video file:
   `python3 main.py --source path/to/video.mp4` or `.venv/bin/python main.py --source video/test1.mp4`

Verification checklist:

1. A window opens and displays live frames.
2. Face boxes appear around visible faces.
3. Face count text updates as people enter/leave view.
4. Press `Ctrl+C` in the terminal to quit cleanly.
5. Quick dependency check (optional): `python3 -c "import cv2; print(cv2.__version__)"`

#### Day 2

Goal: train a local face-recognition model from your own pictures and use it to label faces in video.

Files used:

1. `enroll.py` - trains the local model from images in `data/known_faces`.
2. `main.py` - loads the saved model and labels known or unknown faces.
3. `data/known_faces/` - one folder per person, containing that person's photos.

Project setup for training:

1. Create folders like `data/known_faces/alex/` and `data/known_faces/sam/`.
2. Put a few clear face photos in each person's folder (JPG Images).
3. Use front-facing photos with good lighting, a visible face, and minimal blur.
4. Include a few different angles or expressions, but keep the face large and clear.
5. Avoid sunglasses, heavy shadows, group photos, and tiny faces if possible.
6. Run `python3 enroll.py` to build `data/model.yml` and `data/labels.json`.
7. Run `python3 main.py --source video/test1.mp4` to test recognition on a video file.

Verification checklist:

1. `data/model.yml` is created after training.
2. `data/labels.json` is created after training.
3. The video window still opens and detects faces.
4. Known people show a name above their face.
5. Unknown people show `UNKNOWN`.
6. Press `Ctrl+C` in the terminal to stop cleanly.

#### Whats Next?

Stranger alert action: right now you label UNKNOWN, but you should also trigger an alert event (at minimum console alert + save snapshot).
Persistent event logging: save timestamp, predicted label, confidence, source to a local CSV.
Clear “registered in system” workflow evidence: document and demo enrollment output (model + labels) as part of final deliverable.
Optional but good to strengthen grading:

Threshold tuning + short evaluation table (false matches, unknown detection rate).
Your “go deeper” feature (temporal smoothing over multiple frames) to reduce one-frame mislabels.
Practical next steps:

Add unknown alert snapshot saving.
Add CSV logging for every recognition event.
Add cooldown so same unknown isn’t logged every frame.
Run 3 demo scenarios: known person, unknown person, mixed scene.
Record results and include a short metrics section in README/report.
