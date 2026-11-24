import cv2
import numpy as np
import tensorflow as tf
from src.preprocessing import FacePreprocessor

import json
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from src.preprocessing import FacePreprocessor

class LivenessDetector:
    def __init__(self, model_path=None, face_model_path=None):
        root = Path(__file__).resolve().parents[1]
        if model_path is None:
            model_path = root / "models" / "model.weights.h5"
        else:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = root / model_path

        if face_model_path is None:
            face_model_path = root / "blaze_face_short_range.tflite"
        else:
            face_model_path = Path(face_model_path)
            if not face_model_path.is_absolute():
                face_model_path = root / face_model_path

        self.model = self._load_model(model_path)
        self.preprocessor = FacePreprocessor(
            model_asset_path=str(face_model_path),
            margin_ratio=0.15
        )

        # Static graph compilation
        @tf.function(reduce_retracing=True)
        def _predict_step(feed):
            return self.model(feed, training=False)
        self._predict_step = _predict_step

    def _load_model(self, model_path: Path):
        try:
            return tf.keras.models.load_model(str(model_path), compile=False)
        except Exception as err:
            config_candidates = [
                model_path.parent / "config.json",
                Path("models/config.json"),
            ]
            for cfg_path in config_candidates:
                if cfg_path.exists():
                    try:
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        model = tf.keras.models.model_from_json(json.dumps(cfg))
                        model.load_weights(str(model_path))
                        return model
                    except Exception:
                        continue
            raise err
        
    def predict(self, input_image):
        # Cek apakah input berupa teks path file atau langsung array gambar (frame)
        if isinstance(input_image, str):
            img = cv2.imread(input_image)
        else:
            img = input_image.copy()
            
        if img is None:
            return "Gambar tidak ditemukan", 0.0
            
        result = self.preprocessor.process_frame(img)
        if result is None:
            return "Wajah tidak terdeteksi", 0.0
            
        rgb_face, fft_spectrum = result 
        
        # Paksa ukuran gambar menjadi 224x224
        rgb_face = cv2.resize(rgb_face, (224, 224))
        rgb_face = cv2.cvtColor(rgb_face, cv2.COLOR_BGR2RGB)
        fft_spectrum = cv2.resize(fft_spectrum, (224, 224))
        fft_spectrum = cv2.cvtColor(fft_spectrum, cv2.COLOR_BGR2RGB)
        
        rgb_input = tf.convert_to_tensor(np.expand_dims(rgb_face.astype(np.float32), axis=0))
        fft_input = tf.convert_to_tensor(np.expand_dims(fft_spectrum.astype(np.float32), axis=0))
        
        inputs = {
            'rgb_input': rgb_input,
            'fft_input': fft_input
        }
        
        try:
            output = self._predict_step(inputs)
        except Exception:
            output = self.model(inputs, training=False)
            
        prediction = float(np.asarray(output).reshape(-1)[0])
        
        if prediction < 0.5:
            label = "Real (Bonafide)"
            confidence = (1.0 - prediction) * 100
        else:
            label = "Fake (Spoof)"
            confidence = prediction * 100
            
        return label, confidence