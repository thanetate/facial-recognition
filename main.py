import argparse
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
    return parser


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
            # waitKey keeps the window responsive even when we are not reading keys.
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print("Stopped by user with Ctrl+C.")
    finally:
        # Release the camera/video and close the window cleanly.
        cap.release()
        cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())