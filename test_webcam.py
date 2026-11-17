import cv2
import os
import urllib.request
import collections
from src.inference import LivenessDetector

def main():
    print("Memuat model AI...")
    model_path = "models/antispoofing_model.h5"
    
    if not os.path.exists(model_path):
        print(f"Error: Model tidak ditemukan di {model_path}")
        return

    detector = LivenessDetector(model_path=model_path)
    print("Model berhasil dimuat!")

    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print("Mengunduh file pelacak wajah...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml", cascade_path)
        
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # --- TAMBAHAN: Buffer untuk menghaluskan tebakan (Moving Average) ---
    # Menyimpan 10 hasil terakhir agar angka di layar tidak lompat-lompat
    history_length = 10
    label_buffer = collections.deque(maxlen=history_length)
    conf_buffer = collections.deque(maxlen=history_length)
    # --------------------------------------------------------------------

    print("\nMembuka kamera... Tekan 'q' untuk menutup program.")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        for (x, y, w_box, h_box) in faces:
            # Dapatkan tebakan mentah dari frame saat ini
            raw_label, raw_confidence = detector.predict(frame)
            
            # Masukkan ke dalam buffer
            label_score = 1 if "Real" in raw_label else 0
            label_buffer.append(label_score)
            conf_buffer.append(raw_confidence)
            
            # Hitung rata-ratanya
            avg_label_score = sum(label_buffer) / len(label_buffer)
            avg_conf = sum(conf_buffer) / len(conf_buffer)
            
            # Tentukan label dan warna akhir berdasarkan rata-rata
            final_label = "Real (Bonafide)" if avg_label_score > 0.5 else "Fake (Spoof)"
            color = (0, 255, 0) if avg_label_score > 0.5 else (0, 0, 255)
            
            # Gambar UI yang sudah dihaluskan
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            text = f"{final_label} ({avg_conf:.1f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow('Live Liveness Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()