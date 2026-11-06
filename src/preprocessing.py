import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FacePreprocessor:
    def __init__(self, model_asset_path='blaze_face_short_range.tflite', margin_ratio=0.15):
        self.margin = margin_ratio
        
        # Inisialisasi MediaPipe Tasks API
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.7)
        self.detector = vision.FaceDetector.create_from_options(options)

    def process_frame(self, frame):
        """Memproses frame numpy array, crop wajah dengan margin, dan ekstrak 2D FFT."""
        if frame is None:
            return None

        h_img, w_img, _ = frame.shape
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Konversi numpy array ke format mp.Image untuk Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

        detection_result = self.detector.detect(mp_image)

        if not detection_result.detections:
            return None

        bbox = detection_result.detections[0].bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

        margin_w = int(w * self.margin)
        margin_h = int(h * self.margin)

        x_min = max(0, x - margin_w)
        y_min = max(0, y - margin_h)
        x_max = min(w_img, x + w + margin_w)
        y_max = min(h_img, y + h + margin_h)

        face_crop = frame[y_min:y_max, x_min:x_max]
        
        # Hindari error jika hasil crop kosong (dimensi 0) akibat boundary box di luar batas gambar
        if face_crop.size == 0:
            return None
            
        fft_magnitude = self._compute_2d_fft(face_crop)

        return face_crop, fft_magnitude

    def _compute_2d_fft(self, face_image):
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        fft_resized = cv2.resize(magnitude_normalized, (224, 224))
        fft_3channel = cv2.cvtColor(fft_resized, cv2.COLOR_GRAY2BGR)
        
        return fft_3channel