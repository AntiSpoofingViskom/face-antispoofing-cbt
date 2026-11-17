# face-antispoofing-cbt

Deteksi wajah palsu (*face anti-spoofing* / liveness detection) menggunakan
MediaPipe Face Detection, 2D FFT, dan MobileNetV3.

## Struktur singkat

| Folder | Isi |
| --- | --- |
| `src/` | Pipeline data, preprocessing, model, training, inference |
| `app/` | Web UI Streamlit (Role 3) |
| `configs/` | Konfigurasi project |
| `models/` | Bobot model hasil training (tidak di-commit) |
| `reports/` | Grafik evaluasi dan dokumentasi |

## Menjalankan Web UI (Role 3)

1. Install dependency:

   ```bash
   pip install -r requirements.txt
   ```

2. Pastikan Role 2 sudah menyimpan bobot terbaik, misalnya
   `models/antispoofing_model.h5`, dan file
   `blaze_face_short_range.tflite` tersedia di root project.

3. Jalankan aplikasi dari root project:

   ```bash
   streamlit run app/streamlit_app.py
   ```

4. Buka URL yang ditampilkan Streamlit, tekan **START** pada panel kamera, dan
   izinkan akses webcam.

Fitur UI:

- Capture webcam real-time melalui `streamlit-webrtc`
  (fallback ke snapshot `st.camera_input` + OpenCV bila WebRTC tidak tersedia).
- Pipeline frame → `FacePreprocessor` (MediaPipe + FFT) → model Keras →
  status **ASLI / PALSU** dengan skor spoof, confidence, dan FPS.
- Overlay bounding box dan status langsung pada video.
- Moving-average smoothing agar status tidak berkedip, dengan ambang yang bisa
  diatur dari sidebar.
- Panel pratinjau crop RGB & spektrum FFT serta riwayat skor spoof.
