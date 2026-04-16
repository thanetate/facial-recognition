import argparse
import json
from pathlib import Path

import cv2


def parse_source(raw_source: str):
    # Support both camera index ("0") and file path input.
    # Meaning we can test using the camera on my laptop, 
    # AND we can test using video files
    if raw_source.isdigit():
        return int(raw_source)
    return raw_source


def build_parser() -> argparse.ArgumentParser:
    # argparse builds the CLI for this script
    parser = argparse.ArgumentParser(
        description="Goal - open a video source and detect faces in each frame."
    )
    parser.add_argument(
        "--source",
        default="0",
        # "0" means the default webcam; a file path lets us test recorded video
        help="Video source - camera index or path to a video file.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        # Lower values can find more faces but may increase false detections.
        help="Scale factor - Haar cascade detector.",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        # Higher values make detection stricter and usually reduce noise.
        help="Min neighbors - Haar cascade detector.",
    )
    parser.add_argument(
        "--model",
        default="data/model.yml",
        # This file is created by enroll.py after training on known face images.
        help="Path to the saved face-recognition model.",
    )
    parser.add_argument(
        "--labels",
        default="data/labels.json",
        # This file maps numeric training IDs back to person names.
        help="Path to the saved label map.",
    )
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        default=80.0,
        # Lower confidence values are better for LBPH, so values above this are unknown.
        help="Confidence threshold for unknown faces.",
    )
    return parser


def load_label_map(labels_path: Path) -> dict[int, str]:
    # Convert the saved JSON keys back into integers for lookup during recognition.
    if not labels_path.exists():
        return {}

    with labels_path.open("r", encoding="utf-8") as file_handle:
        raw_labels = json.load(file_handle)

    return {int(label_id): name for label_id, name in raw_labels.items()}


def create_face_recognizer():
    # LBPH lives in opencv-contrib, so recognition only works if that package is installed.
    if not hasattr(cv2, "face"):
        return None
    return cv2.face.LBPHFaceRecognizer_create()


def main() -> int:
    # Read the CLI options entered by the user
    args = build_parser().parse_args()
    source = parse_source(args.source)

    # Use OpenCV's built-in Haar cascade model file
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    # The classifier is the face detector model that scans each frame
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        print(f"Error: could not load cascade file at {cascade_path}")
        return 1

    # Open webcam or video file based on --source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source: {source}")
        return 1

    model_path = Path(args.model)
    labels_path = Path(args.labels)
    label_map = load_label_map(labels_path)
    recognizer = None

    # Load the trained model only if it exists.
    if model_path.exists() and label_map:
        recognizer = create_face_recognizer()
        if recognizer is None:
            print("Warning: cv2.face is not available, so recognition is disabled.")
        else:
            # This loads the model that was created by enroll.py.
            recognizer.read(str(model_path))
            print(f"Loaded recognition model from {model_path}")
    else:
        print("No trained model found yet, so this run will only detect faces.")

    # Match file playback to the video's own frame rate when possible.
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps and source_fps > 0:
        frame_delay_ms = max(1, int(round(1000 / source_fps)))
    else:
        frame_delay_ms = 1

    # Keep the app running until the user stops it with Ctrl+C or the video ends.
    print("Running face detection. Press Ctrl+C in the terminal to stop.")

    # Create the window once so OpenCV can show frames consistently.
    cv2.namedWindow("Facial Recognition - Milestone 1", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of stream or failed frame read.")
                break

            # Detect faces in grayscale for faster and more stable detection.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detectMultiScale returns rectangles for every face it finds.
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=args.scale_factor,
                minNeighbors=args.min_neighbors,
                minSize=(40, 40),
            )

            for x, y, w, h in faces:
                # Draw a box around each detected face.
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 200, 50), 2)

                # If we have a trained model, use it to label the face.
                if recognizer is not None:
                    # Crop the detected face and resize it to match training images.
                    face_region = gray[y : y + h, x : x + w]
                    if face_region.size > 0:
                        face_region = cv2.resize(face_region, (200, 200))
                        label_id, confidence = recognizer.predict(face_region)
                        person_name = label_map.get(label_id, "UNKNOWN")
                        is_known = confidence <= args.unknown_threshold and person_name != "UNKNOWN"

                        label_text = person_name if is_known else "UNKNOWN"
                        label_color = (50, 200, 50) if is_known else (0, 0, 255)
                        cv2.putText(
                            frame,
                            f"{label_text} ({confidence:.1f})",
                            (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            label_color,
                            2,
                            cv2.LINE_AA,
                        )

            # Show the current number of detected faces in the corner.
            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Display the processed frame in a window.
            cv2.imshow("Facial Recognition - Milestone 1", frame)
            # Use the video's FPS for file playback so it feels closer to normal speed.
            # For webcams, this still keeps the window responsive.
            cv2.waitKey(frame_delay_ms)
    except KeyboardInterrupt:
        print("Stopped by user with Ctrl+C.")
    finally:
        # Release the camera/video and close the window cleanly.
        cap.release()
        cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())