import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    # This script trains a local model from labeled images on disk.
    parser = argparse.ArgumentParser(
        description="Train a local face-recognition model from images in data/known_faces."
    )
    parser.add_argument(
        "--input",
        default="data/known_faces",
        help="Folder with one subfolder per person, each containing face images.",
    )
    parser.add_argument(
        "--model",
        default="data/model.yml",
        help="Where to save the trained face-recognition model.",
    )
    parser.add_argument(
        "--labels",
        default="data/labels.json",
        help="Where to save the numeric label-to-name mapping.",
    )
    return parser


def create_face_recognizer():
    # LBPH is a simple OpenCV recognizer that works well for a class project.
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face is not available. Install opencv-contrib-python to train the model."
        )
    return cv2.face.LBPHFaceRecognizer_create()


def largest_face(gray_image, detector):
    # If an image contains more than one face, use the biggest one as the main subject.
    faces = detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None

    return max(faces, key=lambda face: face[2] * face[3])


def main() -> int:
    # Read the folder and output locations from the command line.
    args = build_parser().parse_args()
    input_dir = Path(args.input)
    model_path = Path(args.model)
    labels_path = Path(args.labels)

    if not input_dir.exists():
        print(f"Error: input folder does not exist: {input_dir}")
        return 1

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        print(f"Error: could not load cascade file at {cascade_path}")
        return 1

    images = []
    labels = []
    label_map: dict[int, str] = {}
    next_label_id = 0

    # Each subfolder name becomes a person label, like data/known_faces/alex/.
    for person_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        person_name = person_dir.name
        label_id = next_label_id
        next_label_id += 1
        label_map[label_id] = person_name

        # Read every image in that person's folder and extract one face from each image.
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skipping unreadable file: {image_path}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_box = largest_face(gray, detector)
            if face_box is None:
                print(f"No face found in {image_path}, skipping it.")
                continue

            x, y, w, h = face_box
            # Training works best when all faces are the same size.
            face_crop = gray[y : y + h, x : x + w]
            face_crop = cv2.resize(face_crop, (200, 200))

            images.append(face_crop)
            labels.append(label_id)

    if not images:
        print(f"Error: no usable face images found in {input_dir}")
        return 1

    # Train the model and save it locally so main.py can use it later.
    recognizer = create_face_recognizer()
    recognizer.train(images, np.array(labels, dtype=np.int32))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    recognizer.save(str(model_path))
    with labels_path.open("w", encoding="utf-8") as file_handle:
        json.dump(label_map, file_handle, indent=2)

    print(f"Trained model saved to {model_path}")
    print(f"Label map saved to {labels_path}")
    print(f"Processed {len(images)} training images for {len(label_map)} people")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())