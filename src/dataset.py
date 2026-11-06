import os
import cv2
import glob
from preprocessing import FacePreprocessor

def process_video_to_frames(video_path, output_dir_rgb, output_dir_fft, label, preprocessor, frame_skip=5):
    """Membaca video, mengekstrak frame, crop wajah, hasilkan FFT."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    frame_count = 0
    saved_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            result = preprocessor.process_frame(frame)
            if result is not None:
                rgb_face, fft_spectrum = result
                filename = f"{label}_video_{video_name}_frame{frame_count}.jpg"
                cv2.imwrite(os.path.join(output_dir_rgb, filename), rgb_face)
                cv2.imwrite(os.path.join(output_dir_fft, filename), fft_spectrum)
                saved_count += 1
        frame_count += 1

    cap.release()
    return saved_count

def process_single_image(image_path, output_dir_rgb, output_dir_fft, label, preprocessor):
    """Membaca gambar statis tunggal, crop wajah, hasilkan FFT."""
    img = cv2.imread(image_path)
    if img is None:
        return 0
        
    result = preprocessor.process_frame(img)
    if result is not None:
        rgb_face, fft_spectrum = result
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"{label}_img_{image_name}.jpg"
        
        cv2.imwrite(os.path.join(output_dir_rgb, filename), rgb_face)
        cv2.imwrite(os.path.join(output_dir_fft, filename), fft_spectrum)
        return 1
    return 0

def main():
    preprocessor = FacePreprocessor(margin_ratio=0.15)
    PROCESSED_DIR = "data/processed"
    
    dataset_mapping = [
        ("data/raw/custom/bonafide", "train", "bonafide"),
        ("data/raw/custom/spoof_replay", "train", "spoof"),
        ("data/raw/custom/spoof_print", "train", "spoof"),
        ("data/raw/oulu_npu/true", "train", "bonafide"),
        ("data/raw/oulu_npu/false", "train", "spoof")
    ]
    
    total_extracted = 0
    print("Memulai pipeline ekstraksi data (RGB & FFT)...")
    
    for raw_dir, split, label in dataset_mapping:
        if not os.path.exists(raw_dir):
            continue
            
        out_rgb = os.path.join(PROCESSED_DIR, "rgb", split, label)
        out_fft = os.path.join(PROCESSED_DIR, "fft", split, label)
        os.makedirs(out_rgb, exist_ok=True)
        os.makedirs(out_fft, exist_ok=True)
        
        # Cari file video
        video_files = glob.glob(os.path.join(raw_dir, "*.mp4")) + glob.glob(os.path.join(raw_dir, "*.avi"))
        # Cari file gambar statis (JPG/PNG)
        image_files = glob.glob(os.path.join(raw_dir, "*.jpg")) + glob.glob(os.path.join(raw_dir, "*.png"))
        
        if not video_files and not image_files:
            continue
            
        for video_path in video_files:
            print(f"Memproses video: {video_path}...")
            saved = process_video_to_frames(video_path, out_rgb, out_fft, label, preprocessor, frame_skip=5)
            total_extracted += saved
            
        if image_files:
            print(f"Memproses {len(image_files)} gambar statis dari {raw_dir}...")
            img_saved_count = 0
            for img_path in image_files:
                img_saved_count += process_single_image(img_path, out_rgb, out_fft, label, preprocessor)
            total_extracted += img_saved_count

    print(f"Pipeline selesai! Total pasang gambar yang berhasil diekstrak: {total_extracted}")

if __name__ == "__main__":
    main()