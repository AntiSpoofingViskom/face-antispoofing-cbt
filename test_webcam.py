import collections
import os
import urllib.request

import cv2

from src.face_recognizer import FaceRecognizer
from src.inference import LivenessDetector


HISTORY_LENGTH = 30
MAX_MISSED_FRAMES = 12
MIN_FACE_SIZE = 120
MIN_BRIGHTNESS = 35
MAX_BRIGHTNESS = 220
MIN_SHARPNESS = 40
REGISTER_TARGET_SAMPLES = 20
REGISTER_FRAME_INTERVAL = 3


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
    model_path = "models/dual_fold_1.keras"
    if not os.path.exists(model_path):
        print(f"Error: Model tidak ditemukan di {model_path}")
        return

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

    # State registrasi wajah: "NORMAL", "TYPING_NAME", "CAPTURING"
    mode = "NORMAL"
    typed_name = ""
    captured_crops = []
    capture_frame_tick = 0
    status_toast = ""
    status_toast_frames = 0

    print("\nMembuka kamera... Tekan 'r' untuk daftar wajah baru, 'q' untuk keluar.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_frame, w_frame = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )

        if mode == "TYPING_NAME":
            cv2.rectangle(frame, (20, 20), (w_frame - 20, 125), (30, 30, 30), -1)
            cv2.rectangle(frame, (20, 20), (w_frame - 20, 125), (0, 200, 255), 2)
            cv2.putText(
                frame,
                "REGISTRASI WAJAH BARU - Ketik Nama (Enter=Mulai, Esc=Batal):",
                (35, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Nama: {typed_name}_",
                (35, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
            )

        elif mode == "CAPTURING":
            if len(faces) == 0:
                cv2.putText(
                    frame,
                    f"Merekam {typed_name}: Arahkan wajah ke kamera ({len(captured_crops)}/{REGISTER_TARGET_SAMPLES})",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2,
                )
            else:
                x, y, w_box, h_box = max(faces, key=lambda box: box[2] * box[3])
                face_crop = frame[y:y + h_box, x:x + w_box]
                warning = quality_message(face_crop)
                capture_frame_tick += 1

                if warning:
                    color = (0, 165, 255)
                    text = f"Tahan: {warning} ({len(captured_crops)}/{REGISTER_TARGET_SAMPLES})"
                else:
                    raw_label, _ = detector.predict(face_crop)
                    if "Real" not in raw_label:
                        color = (0, 0, 255)
                        text = f"Ditolak: Wajah terdeteksi Spoof! ({len(captured_crops)}/{REGISTER_TARGET_SAMPLES})"
                    else:
                        color = (255, 200, 0)
                        if capture_frame_tick % REGISTER_FRAME_INTERVAL == 0:
                            captured_crops.append(face_crop.copy())
                        text = f"Merekam {typed_name}: {len(captured_crops)}/{REGISTER_TARGET_SAMPLES}"

                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.putText(frame, text, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                if len(captured_crops) >= REGISTER_TARGET_SAMPLES:
                    recognizer.register_samples(typed_name, captured_crops)
                    status_toast = f"Wajah '{typed_name}' berhasil didaftarkan!"
                    status_toast_frames = 90
                    mode = "NORMAL"
                    label_buffer.clear()
                    conf_buffer.clear()
                    identity_buffer.clear()

        else:  # mode == "NORMAL"
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
                                text = (
                                    f"{final_label} - {stable_identity} ({avg_conf:.1f}%)"
                                    if votes >= 10
                                    else f"{final_label} - mengenali... ({avg_conf:.1f}%)"
                                )
                            else:
                                text = f"{final_label} - Tidak dikenal ({avg_conf:.1f}%)"
                        else:
                            identity_buffer.clear()
                            text = f"{final_label} ({avg_conf:.1f}%)"

                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.putText(frame, text, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        if status_toast_frames > 0:
            status_toast_frames -= 1
            cv2.putText(frame, status_toast, (20, h_frame - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(
            frame,
            "[R] Daftar Wajah Baru   [Q] Keluar",
            (20, h_frame - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
        )

        cv2.imshow("Live Liveness Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if mode == "TYPING_NAME":
            if key == 27:  # ESC
                mode = "NORMAL"
            elif key in (13, 10):  # ENTER
                if typed_name.strip():
                    mode = "CAPTURING"
                    captured_crops = []
                    capture_frame_tick = 0
            elif key in (8, 127):  # BACKSPACE
                typed_name = typed_name[:-1]
            elif 32 <= key <= 126 and len(typed_name) < 25:
                typed_name += chr(key)
        else:
            if key == ord("q"):
                break
            elif key == ord("r") and mode == "NORMAL":
                mode = "TYPING_NAME"
                typed_name = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
