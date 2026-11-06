import cv2
import os
from src.inference import LivenessDetector

def main():
    print("Memuat model AI, mohon tunggu sebentar...")
    # Pastikan path model ini sesuai dengan lokasi modelmu
    model_path = "models/antispoofing_model.h5"
    
    if not os.path.exists(model_path):
        print(f"Error: Model tidak ditemukan di {model_path}")
        return

    # Inisialisasi detektor
    detector = LivenessDetector(model_path=model_path)
    print("Model berhasil dimuat!\n")

    # Buka kamera laptop (angka 0 biasanya untuk webcam bawaan)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Tidak dapat mengakses webcam.")
        return

    print("="*40)
    print("PANDUAN PENGUJIAN:")
    print("1. Arahkan wajahmu ke kamera.")
    print("2. Tekan tombol 's' di keyboard untuk mengambil foto dan menebak.")
    print("3. Coba tes pakai wajah asli, lalu coba tes pakai foto dari HP.")
    print("4. Tekan tombol 'q' untuk menutup program.")
    print("="*40)

    temp_image_path = "temp_test_image.jpg"

    while True:
        # Baca frame dari webcam
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame kamera.")
            break

        # Tampilkan kamera di jendela baru
        cv2.imshow('Uji Coba Liveness Detection (Role 2)', frame)

        # Tunggu input tombol dari keyboard
        key = cv2.waitKey(1) & 0xFF
        
        # Jika tombol 'q' ditekan, keluar dari loop
        if key == ord('q'):
            print("Menutup pengujian...")
            break
            
        # Jika tombol 's' ditekan, lakukan prediksi
        elif key == ord('s'):
            # Simpan frame ke file sementara karena fungsi predict() butuh path file
            cv2.imwrite(temp_image_path, frame)
            print("\nMemproses gambar...")
            
            # Panggil fungsi dari inference.py
            label, confidence = detector.predict(temp_image_path)
            
            # Tampilkan hasil di terminal
            if "Real" in label:
                print(f"✅ HASIL: {label} | Akurasi: {confidence:.2f}% (Wajah Asli)")
            else:
                print(f"❌ HASIL: {label} | Akurasi: {confidence:.2f}% (Wajah Palsu/Layar)")

    # Bersihkan memori setelah selesai
    cap.release()
    cv2.destroyAllWindows()
    
    # Hapus file foto sementara jika ada
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

if __name__ == "__main__":
    main()