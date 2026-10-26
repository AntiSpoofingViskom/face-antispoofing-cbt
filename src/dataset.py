import os
import cv2
import yaml
from preprocessing import LivenessPreprocessor

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def extract_and_process(video_path, output_rgb_dir, output_fft_dir, preprocessor, frame_skip):
    """Membaca video, memproses tiap N-frame, dan menyimpannya sebagai gambar."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Gagal membuka video: {video_path}")
        return

    frame_count = 0
    saved_count = 0
    # Ambil nama file tanpa ekstensi untuk penamaan frame
    video_name = os.path.basename(video_path).split('.')[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Terapkan frame skipping (misal: ambil 1 frame tiap 5 frame)
        if frame_count % frame_skip == 0:
            # 1. Crop wajah menggunakan MediaPipe
            cropped_face = preprocessor.crop_face(frame)
            
            if cropped_face is not None:
                # 2. Ekstrak FFT dari wajah yang sudah di-crop
                fft_face = preprocessor.extract_fft(cropped_face)
                
                # 3. Bentuk path penyimpanan
                rgb_filename = os.path.join(output_rgb_dir, f"{video_name}_f{frame_count}.jpg")
                fft_filename = os.path.join(output_fft_dir, f"{video_name}_f{frame_count}.jpg")
                
                # 4. Simpan gambar ke direktori processed
                cv2.imwrite(rgb_filename, cropped_face)
                cv2.imwrite(fft_filename, fft_face)
                saved_count += 1

        frame_count += 1

    cap.release()
    print(f"[{video_name}] Selesai: {saved_count} pasang frame (RGB & FFT) diekstrak.")

def main():
    print("Membaca konfigurasi dan inisialisasi preprocessor...")
    config = load_config()
    preprocessor = LivenessPreprocessor(config_path="configs/config.yaml")
    frame_skip = config['preprocessing']['frame_skip']

    # Pemetaan direktori dataset lokal (Custom Dataset)
    # Format: (Folder_Raw_Sumber, Folder_Split_Tujuan, Label_Tujuan)
    dataset_mapping = [
        ("data/raw/custom/bonafide", "train", "bonafide"),
        ("data/raw/custom/spoof_replay", "train", "spoof"),
        ("data/raw/custom/spoof_print", "train", "spoof")
    ]

    print("Memulai ekstraksi dataset...")
    for raw_dir, split, label in dataset_mapping:
        # Gunakan path relatif yang sesuai dengan OS
        raw_dir = os.path.normpath(raw_dir)
        
        if not os.path.exists(raw_dir):
            print(f"Folder tidak ditemukan (lewati): {raw_dir}")
            continue

        # Siapkan folder output (RGB dan FFT)
        output_rgb = os.path.join(config['paths']['processed_rgb'], split, label)
        output_fft = os.path.join(config['paths']['processed_fft'], split, label)

        os.makedirs(output_rgb, exist_ok=True)
        os.makedirs(output_fft, exist_ok=True)

        # Iterasi semua file video (.mp4 atau .avi)
        video_files = [f for f in os.listdir(raw_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        if not video_files:
            print(f"Tidak ada video di dalam {raw_dir}")
            
        for video_file in video_files:
            video_path = os.path.join(raw_dir, video_file)
            extract_and_process(video_path, output_rgb, output_fft, preprocessor, frame_skip)

    print("Pipeline data preprocessing selesai!")

if __name__ == "__main__":
    main()