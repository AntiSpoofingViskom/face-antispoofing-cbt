import cv2
import numpy as np
import tensorflow as tf
from src.preprocessing import FacePreprocessor

class LivenessDetector:
    def __init__(self, model_path="models/antispoofing_model.keras"):
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessor = FacePreprocessor(margin_ratio=0.15)
        
    def predict(self, input_image):
        # Cek apakah input berupa teks path file atau langsung array gambar (frame)
        if isinstance(input_image, str):
            img = cv2.imread(input_image)
        else:
            img = input_image.copy()
            
        if img is None:
            return "Gambar tidak ditemukan", 0.0
            
        # --- PERBAIKAN FATAL: KONVERSI WARNA BGR KE RGB ---
        # Mengubah warna dari standar OpenCV (BGR) ke standar TensorFlow (RGB)
        # agar model AI tidak melihat wajahmu berwarna biru.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # --------------------------------------------------
            
        result = self.preprocessor.process_frame(img)
        if result is None:
            return "Wajah tidak terdeteksi", 0.0
            
        rgb_face, fft_spectrum = result 
        
        # Paksa ukuran gambar menjadi 224x224
        rgb_face = cv2.resize(rgb_face, (224, 224))
        fft_spectrum = cv2.resize(fft_spectrum, (224, 224))
        
        rgb_input = np.expand_dims(rgb_face, axis=0)
        fft_input = np.expand_dims(fft_spectrum, axis=0)
        
        inputs = {
            'rgb_input': rgb_input,
            'fft_input': fft_input
        }
        
        prediction = self.model.predict(inputs, verbose=0)[0][0]
        
        if prediction < 0.5:
            label = "Real (Bonafide)"
            confidence = (1.0 - prediction) * 100
        else:
            label = "Fake (Spoof)"
            confidence = prediction * 100
            
        return label, confidence