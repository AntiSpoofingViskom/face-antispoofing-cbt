import cv2
import numpy as np
import mediapipe as mp
import yaml
import os

class LivenessPreprocessor:
    def __init__(self, config_path="configs/config.yaml"):
        # Muat konfigurasi
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        self.target_size = tuple(self.config['preprocessing']['target_size'])
        self.margin_ratio = self.config['preprocessing']['margin_ratio']
        
        # Inisialisasi MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=0.5
        )

    def crop_face(self, image_bgr):
        """Mendeteksi wajah, menambahkan margin, dan memotong (crop) gambar."""
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)

        # Jika wajah tidak terdeteksi
        if not results.detections:
            return None

        # Ambil wajah pertama dengan confidence tertinggi
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box

        # Konversi bounding box relatif ke piksel absolut
        xmin = int(bboxC.xmin * w)
        ymin = int(bboxC.ymin * h)
        box_w = int(bboxC.width * w)
        box_h = int(bboxC.height * h)

        # Hitung margin untuk menangkap area tepi (bezel HP jika ada)
        dw = int(box_w * self.margin_ratio)
        dh = int(box_h * self.margin_ratio)

        # Tentukan batas crop agar tidak keluar dari dimensi gambar
        xmin_m = max(0, xmin - dw)
        ymin_m = max(0, ymin - dh)
        xmax_m = min(w, xmin + box_w + dw)
        ymax_m = min(h, ymin + box_h + dh)

        cropped_face = image_bgr[ymin_m:ymax_m, xmin_m:xmax_m]
        
        if cropped_face.size == 0:
            return None

        return cv2.resize(cropped_face, self.target_size)

    def extract_fft(self, face_bgr):
        """Mengekstrak fitur spektrum frekuensi 2D menggunakan Fast Fourier Transform."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        
        # 2D FFT & geser frekuensi rendah ke tengah (center)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # Hitung Magnitude Spectrum dengan skala logaritmik
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        
        # Normalisasi ke 0-255 (format citra)
        normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        fft_uint8 = normalized.astype(np.uint8)
        
        # Konversi ke pseudo-RGB (3 channel) agar bisa masuk ke MobileNetV3
        fft_rgb = cv2.cvtColor(fft_uint8, cv2.COLOR_GRAY2RGB)
        return fft_rgb