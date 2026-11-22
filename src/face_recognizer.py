from pathlib import Path

import cv2
import numpy as np


class FaceRecognizer:
    """Simple local face recognizer using OpenCV LBPH."""

    def __init__(self, reference_dir="data/raw/custom/bonafide", cascade_path="haarcascade_frontalface_default.xml"):
        if not hasattr(cv2, "face"):
            raise RuntimeError("OpenCV face module tidak ada. Install opencv-contrib-python.")
        self.names = {"person1": "Dean", "person2": "Misael", "person3": "Zahran"}
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.face_size = (128, 128)
        self._train(Path(reference_dir))

    def _train(self, reference_dir):
        faces, labels = [], []
        label_ids = {}
        for path in sorted(reference_dir.glob("*.jpg")):
            person = path.stem.split("_")[0]
            if person not in self.names:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            detected = self.detector.detectMultiScale(image, 1.1, 5, minSize=(60, 60))
            if len(detected) == 0:
                continue
            x, y, w, h = max(detected, key=lambda box: box[2] * box[3])
            person_id = label_ids.setdefault(person, len(label_ids))
            face = image[y:y + h, x:x + w]
            faces.append(cv2.resize(face, self.face_size))
            labels.append(person_id)

        if not faces:
            raise RuntimeError(f"Tidak ada wajah referensi ditemukan di {reference_dir}.")
        self.id_to_name = {value: self.names[key] for key, value in label_ids.items()}
        self.model.train(faces, np.asarray(labels, dtype=np.int32))

    def predict(self, face_crop):
        # face_crop sudah dipotong oleh detector webcam; jangan deteksi ulang.
        if face_crop is None or face_crop.size == 0:
            return "Tidak dikenal", 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, self.face_size)
        label_id, distance = self.model.predict(face)
        # LBPH distance: lower is better. 70 is stricter unknown cutoff.
        if distance > 100:
            return "Tidak dikenal", distance
        return self.id_to_name.get(label_id, "Tidak dikenal"), distance
