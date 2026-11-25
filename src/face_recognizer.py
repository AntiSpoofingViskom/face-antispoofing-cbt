import time
from pathlib import Path

import cv2
import numpy as np


class FaceRecognizer:
    """Local face recognizer using OpenCV LBPH with live registration support."""

    def __init__(
        self,
        reference_dir="data/raw/custom/bonafide",
        registered_dir="data/registered_faces",
        cascade_path="haarcascade_frontalface_default.xml",
    ):
        if not hasattr(cv2, "face"):
            raise RuntimeError("OpenCV face module tidak ada. Install opencv-contrib-python.")
        self.legacy_names = {"person1": "Dean", "person2": "Misael", "person3": "Zahran"}
        self.reference_dir = Path(reference_dir)
        self.registered_dir = Path(registered_dir)
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.face_size = (128, 128)
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.id_to_name = {}
        self.is_trained = False
        self.reload()

    def reload(self):
        """Muat ulang seluruh data wajah referensi dan latih LBPH."""
        faces, labels = [], []
        label_ids = {}

        # 1. Data legacy dari data/raw/custom/bonafide (person1/2/3)
        if self.reference_dir.exists():
            for path in sorted(self.reference_dir.glob("*.jpg")):
                prefix = path.stem.split("_")[0]
                person_name = self.legacy_names.get(prefix)
                if not person_name:
                    continue
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                detected = self.detector.detectMultiScale(image, 1.1, 5, minSize=(60, 60))
                if len(detected) == 0:
                    continue
                x, y, w, h = max(detected, key=lambda box: box[2] * box[3])
                person_id = label_ids.setdefault(person_name, len(label_ids))
                faces.append(cv2.resize(image[y:y + h, x:x + w], self.face_size))
                labels.append(person_id)

        # 2. Data hasil registrasi webcam dari data/registered_faces/<Nama>/*.jpg
        if self.registered_dir.exists():
            for person_folder in sorted(self.registered_dir.iterdir()):
                if not person_folder.is_dir():
                    continue
                person_name = person_folder.name
                for path in sorted(person_folder.glob("*.jpg")):
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                    person_id = label_ids.setdefault(person_name, len(label_ids))
                    faces.append(cv2.resize(image, self.face_size))
                    labels.append(person_id)

        self.id_to_name = {idx: name for name, idx in label_ids.items()}
        if faces:
            self.model = cv2.face.LBPHFaceRecognizer_create()
            self.model.train(faces, np.asarray(labels, dtype=np.int32))
            self.is_trained = True
        else:
            self.is_trained = False

    def register_samples(self, person_name, face_crops):
        """Simpan kumpulan crop wajah baru dan latih ulang recognizer."""
        clean_name = "".join(c for c in person_name.strip() if c.isalnum() or c in (" ", "_", "-")).strip()
        if not clean_name or not face_crops:
            return False

        target_dir = self.registered_dir / clean_name
        target_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())

        for idx, crop in enumerate(face_crops, start=1):
            save_path = target_dir / f"face_{ts}_{idx:02d}.jpg"
            cv2.imwrite(str(save_path), crop)

        self.reload()
        return True

    def predict(self, face_crop):
        if not self.is_trained or face_crop is None or face_crop.size == 0:
            return "Tidak dikenal", 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, self.face_size)
        label_id, distance = self.model.predict(face)
        if distance > 100:
            return "Tidak dikenal", distance
        return self.id_to_name.get(label_id, "Tidak dikenal"), distance
