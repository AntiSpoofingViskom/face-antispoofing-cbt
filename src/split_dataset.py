import os
import glob
import random
import shutil

def split_processed_dataset(base_dir="data/processed", val_ratio=0.10, test_ratio=0.10, seed=42):
    """
    Membagi data dari folder train ke folder val dan test secara sinkron
    antara direktori RGB dan FFT.
    """
    random.seed(seed)
    classes = ["bonafide", "spoof"]
    
    print(f"Memulai pembagian dataset (Train: {int((1-val_ratio-test_ratio)*100)}%, Val: {int(val_ratio*100)}%, Test: {int(test_ratio*100)}%)...")
    
    for label in classes:
        # Path sumber (Train)
        rgb_train_dir = os.path.join(base_dir, "rgb", "train", label)
        fft_train_dir = os.path.join(base_dir, "fft", "train", label)
        
        if not os.path.exists(rgb_train_dir):
            print(f"[SKIP] Folder {rgb_train_dir} tidak ditemukan.")
            continue
            
        # Buat folder tujuan (Val & Test) untuk RGB dan FFT
        for split in ["val", "test"]:
            os.makedirs(os.path.join(base_dir, "rgb", split, label), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "fft", split, label), exist_ok=True)
            
        # Ambil semua daftar nama file gambar di rgb/train/{label}
        rgb_files = [os.path.basename(f) for f in glob.glob(os.path.join(rgb_train_dir, "*.jpg"))]
        
        if not rgb_files:
            print(f"[INFO] Tidak ada file .jpg di {rgb_train_dir} untuk dipindahkan.")
            continue
            
        # Acak urutan file agar distribusi video lokal dan OULU-NPU merata
        random.shuffle(rgb_files)
        
        total_files = len(rgb_files)
        val_count = int(total_files * val_ratio)
        test_count = int(total_files * test_ratio)
        
        # Tentukan daftar file yang akan dipindah ke Val dan Test
        val_files = rgb_files[:val_count]
        test_files = rgb_files[val_count : val_count + test_count]
        
        # Fungsi pembantu untuk memindahkan pasangan RGB & FFT sekaligus
        def move_pair(file_list, target_split):
            moved = 0
            for filename in file_list:
                src_rgb = os.path.join(rgb_train_dir, filename)
                src_fft = os.path.join(fft_train_dir, filename)
                
                dst_rgb = os.path.join(base_dir, "rgb", target_split, label, filename)
                dst_fft = os.path.join(base_dir, "fft", target_split, label, filename)
                
                # Pastikan kedua file (RGB dan FFT) ada sebelum dipindahkan
                if os.path.exists(src_rgb) and os.path.exists(src_fft):
                    shutil.move(src_rgb, dst_rgb)
                    shutil.move(src_fft, dst_fft)
                    moved += 1
            return moved

        moved_val = move_pair(val_files, "val")
        moved_test = move_pair(test_files, "test")
        remaining_train = total_files - moved_val - moved_test
        
        print(f"Kelas [{label.upper()}] (Total Awal: {total_files} pasang):")
        print(f"  -> Train : {remaining_train} pasang")
        print(f"  -> Val   : {moved_val} pasang")
        print(f"  -> Test  : {moved_test} pasang\n")

    print("Pembagian dataset selesai! Struktur folder train, val, dan test sudah siap.")

if __name__ == "__main__":
    split_processed_dataset()