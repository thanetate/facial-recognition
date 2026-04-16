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
   Build/save face encodings file

Plus folders:
data/known_faces/ (training images by person)
data/encodings.pkl (saved embeddings)
logs/events.csv and alerts/ (logs and unknown snapshots)

#### Data Storage Plan (Local Only)

No cloud database is required for this project. All data stays local:

1. Face encodings: `data/encodings.pkl`
   Stores registered user embeddings and labels.
2. Event log: `logs/events.csv`
   Stores timestamp, predicted label, and confidence score.
3. Alert images: `alerts/`
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
