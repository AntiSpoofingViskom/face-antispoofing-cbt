"""
Generate split files untuk OULU-NPU Protocol IV (Leave-One-Camera-Out).
Menghasilkan fold_X_train.txt, fold_X_dev.txt, fold_X_test.txt di data/splits/
"""
import os
import glob

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_TRUE = os.path.join(BASE_DIR, 'data', 'raw', 'oulu_npu', 'true')
RAW_FALSE = os.path.join(BASE_DIR, 'data', 'raw', 'oulu_npu', 'false')
SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')

# Protocol IV: Leave-One-Camera-Out, 6 fold
# Format: {test_camera: dev_camera}, train = sisanya
FOLDS = {
    1: {'test': 1, 'dev': 2},
    2: {'test': 2, 'dev': 3},
    3: {'test': 3, 'dev': 4},
    4: {'test': 4, 'dev': 5},
    5: {'test': 5, 'dev': 6},
    6: {'test': 6, 'dev': 1},
}

def get_camera(filename):
    """Ambil nomor kamera dari nama file: {kamera}_{session}_{user}_{PAI}_{frame}.jpg"""
    return int(os.path.basename(filename).split('_')[0])

def get_all_files():
    """Kumpulkan semua file dengan label (path, label)."""
    files = []
    for f in glob.glob(os.path.join(RAW_TRUE, '*.jpg')):
        files.append((f, 'bonafide'))
    for f in glob.glob(os.path.join(RAW_FALSE, '*.jpg')):
        files.append((f, 'spoof'))
    return files

def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)
    all_files = get_all_files()
    print(f"Total file: {len(all_files)} (bonafide: {sum(1 for _,l in all_files if l=='bonafide')}, spoof: {sum(1 for _,l in all_files if l=='spoof')})")

    for fold_id, cameras in FOLDS.items():
        test_cam = cameras['test']
        dev_cam = cameras['dev']
        train_cams = [c for c in range(1, 7) if c != test_cam and c != dev_cam]

        train_files = [(p, l) for p, l in all_files if get_camera(p) in train_cams]
        dev_files = [(p, l) for p, l in all_files if get_camera(p) == dev_cam]
        test_files = [(p, l) for p, l in all_files if get_camera(p) == test_cam]

        for split_name, split_files in [('train', train_files), ('dev', dev_files), ('test', test_files)]:
            out_path = os.path.join(SPLITS_DIR, f'fold_{fold_id}_{split_name}.txt')
            with open(out_path, 'w') as f:
                for filepath, label in sorted(split_files):
                    filename = os.path.basename(filepath)
                    f.write(f"{filename}\t{label}\n")

        print(f"Fold {fold_id}: train={len(train_files)}, dev={len(dev_files)}, test={len(test_files)} | test_cam={test_cam}, dev_cam={dev_cam}, train_cams={train_cams}")

    print(f"\nSplit files disimpan di: {SPLITS_DIR}")

if __name__ == '__main__':
    main()
