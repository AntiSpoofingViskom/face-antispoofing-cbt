import collections
import os
import urllib.request
from pathlib import Path

import cv2

from src.inference import LivenessDetector
from src.face_recognizer import FaceRecognizer
from src.model_paths import resolve_default_model_path


HISTORY_LENGTH = 30
MAX_MISSED_FRAMES = 12
MIN_FACE_SIZE = 120
MIN_BRIGHTNESS = 35
MAX_BRIGHTNESS = 220
MIN_SHARPNESS = 40


def quality_message(face_crop):
    """Return warning text, or None when face quality is acceptable."""
    if face_crop.shape[1] < MIN_FACE_SIZE or face_crop.shape[0] < MIN_FACE_SIZE:
        return "Dekatkan wajah ke kamera"

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    if brightness < MIN_BRIGHTNESS:
        return "Pencahayaan kurang"
    if brightness > MAX_BRIGHTNESS:
        return "Pencahayaan terlalu terang"

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharpness < MIN_SHARPNESS:
        return "Wajah kurang fokus"
    return None


def main():
    print("Memuat model AI...")
    model_path = resolve_default_model_path(Path(__file__).resolve().parent / "models")

    if not model_path.exists():
        print(f"Error: Model tidak ditemukan di {model_path}")
        print("Letakkan file model (.keras / .h5) di folder models/.")
        return

    print(f"Menggunakan model: {model_path}")
    detector = LivenessDetector(model_path=model_path)
    recognizer = FaceRecognizer()
    print("Model liveness dan face recognition berhasil dimuat!")

    cascade_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        print("Mengunduh file pelacak wajah...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
            cascade_path,
        )

    face_cascade = cv2.CascadeClassifier(cascade_path)
    label_buffer = collections.deque(maxlen=HISTORY_LENGTH)
    conf_buffer = collections.deque(maxlen=HISTORY_LENGTH)
    identity_buffer = collections.deque(maxlen=HISTORY_LENGTH)
    missed_frames = 0

    print("\nMembuka kamera... Tekan 'q' untuk menutup program.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )

        if len(faces) == 0:
            missed_frames += 1
            if missed_frames > MAX_MISSED_FRAMES:
                label_buffer.clear()
                conf_buffer.clear()
                identity_buffer.clear()
                missed_frames = 0
            cv2.putText(frame, "Mencari wajah...", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            missed_frames = 0
            x, y, w_box, h_box = max(faces, key=lambda box: box[2] * box[3])
            face_crop = frame[y:y + h_box, x:x + w_box]
            warning = quality_message(face_crop)

            if warning:
                color = (0, 165, 255)
                text = warning
                # Tetap prediksi liveness untuk update buffer (tanpa masukkan ke buffer)
                raw_label, raw_confidence = detector.predict(face_crop)
                # Jangan masukkan ke buffer saat warning
                # Jangan clear buffer saat warning - biarkan buffer tetap
            else:
                raw_label, raw_confidence = detector.predict(face_crop)
                label_buffer.append(1 if "Real" in raw_label else 0)
                conf_buffer.append(raw_confidence)

                if len(label_buffer) < HISTORY_LENGTH:
                    color = (0, 165, 255)
                    text = f"Menganalisis... {len(label_buffer)}/{HISTORY_LENGTH}"
                else:
                    real_votes = sum(label_buffer)
                    is_real = real_votes > HISTORY_LENGTH / 2
                    final_label = "Real (Bonafide)" if is_real else "Fake (Spoof)"
                    color = (0, 255, 0) if is_real else (0, 0, 255)
                    avg_conf = sum(conf_buffer) / len(conf_buffer)
                    if is_real:
                        identity, _ = recognizer.predict(face_crop)
                        identity_buffer.append(identity)
                        known_identities = [name for name in identity_buffer if name != "Tidak dikenal"]
                        if known_identities:
                            stable_identity, votes = collections.Counter(known_identities).most_common(1)[0]
                            text = f"{final_label} - {stable_identity} ({avg_conf:.1f}%)" if votes >= 10 else f"{final_label} - mengenali... ({avg_conf:.1f}%)"
                        else:
                            text = f"{final_label} - Tidak dikenal ({avg_conf:.1f}%)"
                    else:
                        identity_buffer.clear()
                        text = f"{final_label} ({avg_conf:.1f}%)"

            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, text, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.imshow("Live Liveness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
