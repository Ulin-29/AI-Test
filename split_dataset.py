import os
import shutil
import random

# --- Konfigurasi ---
SOURCE_DIR = 'datasets_images'
OUTPUT_DIR = 'datasets'
SPLIT_RATIO = 0.8  # 80% data latihan, 20% data ujian

def split_data():
    """Membagi gambar menjadi folder train dan validation."""
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Folder '{SOURCE_DIR}' tidak ditemukan. Jalankan 'prepare_dataset.py' dulu.")
        return
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR) # Hapus folder lama agar bersih

    train_dir = os.path.join(OUTPUT_DIR, 'train')
    validation_dir = os.path.join(OUTPUT_DIR, 'validation')
    
    print(f"Memulai pembagian dataset dari '{SOURCE_DIR}'...")
    for category_name in os.listdir(SOURCE_DIR):
        category_source_path = os.path.join(SOURCE_DIR, category_name)
        if os.path.isdir(category_source_path):
            os.makedirs(os.path.join(train_dir, category_name), exist_ok=True)
            os.makedirs(os.path.join(validation_dir, category_name), exist_ok=True)
            
            files = [f for f in os.listdir(category_source_path)]
            if len(files) < 5:
                print(f"[Peringatan] Kategori '{category_name}' hanya punya {len(files)} gambar (disarankan min 5). Semua akan dipakai untuk latihan.")
                for file_name in files:
                    shutil.copy(os.path.join(category_source_path, file_name), os.path.join(train_dir, category_name, file_name))
                continue

            random.shuffle(files)
            split_point = int(len(files) * SPLIT_RATIO)
            
            for file_name in files[:split_point]:
                shutil.copy(os.path.join(category_source_path, file_name), os.path.join(train_dir, category_name, file_name))
            for file_name in files[split_point:]:
                shutil.copy(os.path.join(category_source_path, file_name), os.path.join(validation_dir, category_name, file_name))
                
    print("\n✅ Fase B Selesai: Dataset terstruktur siap di folder 'datasets'.")

if __name__ == '__main__':
    split_data()