import os
import fitz  # PyMuPDF
from tqdm import tqdm
import cv2
import numpy as np

# === KONFIGURASI ===
PDF_SOURCE_DIR = 'pdf_sources'
DATASET_DIR = 'dataset'
DPI = 350  # detail tinggi
IMG_SIZE = (224, 224)  # konsisten dengan model
MIN_CONTENT_RATIO = 0.005  # ambang batas isi halaman (0.5% pixel non-putih)
SAVE_GRAYSCALE = True  # simpan dalam grayscale agar efisien untuk AI

def is_blank_page(image):
    """
    Deteksi apakah halaman kosong (terlalu putih / sedikit konten)
    """
    # Hitung proporsi piksel bukan putih
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    non_white_ratio = np.count_nonzero(gray < 240) / gray.size
    return non_white_ratio < MIN_CONTENT_RATIO

def preprocess_image(pix):
    """
    Membersihkan dan meningkatkan kualitas gambar hasil konversi PDF.
    """
    # Konversi ke numpy array
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if img_np.shape[2] == 4:  # hilangkan alpha channel
        img_np = img_np[:, :, :3]

    # Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Denoise halus
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Adaptive threshold → teks lebih kontras
    processed = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Resize agar seragam dengan input model
    resized = cv2.resize(processed, IMG_SIZE)

    return resized


def convert_pdfs_to_images():
    """
    Mengubah semua PDF di dalam PDF_SOURCE_DIR menjadi dataset gambar siap latih.
    """
    if not os.path.exists(PDF_SOURCE_DIR):
        print(f"❌ Folder sumber '{PDF_SOURCE_DIR}' tidak ditemukan.")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)

    category_dirs = [d for d in os.listdir(PDF_SOURCE_DIR) if os.path.isdir(os.path.join(PDF_SOURCE_DIR, d))]
    if not category_dirs:
        print("⚠️ Tidak ada subfolder kategori di dalam 'pdf_sources'.")
        print("Buat struktur seperti: pdf_sources/ktp/, pdf_sources/ijazah/, dst.")
        return

    total_saved = 0
    total_skipped = 0

    print(f"🚀 Mulai konversi PDF dari '{PDF_SOURCE_DIR}' → '{DATASET_DIR}'")

    for category_name in category_dirs:
        category_source = os.path.join(PDF_SOURCE_DIR, category_name)
        category_output = os.path.join(DATASET_DIR, category_name)
        os.makedirs(category_output, exist_ok=True)

        pdf_files = [f for f in os.listdir(category_source) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"⚠️ Tidak ada PDF di folder kategori '{category_name}'")
            continue

        for pdf_filename in tqdm(pdf_files, desc=f"  → {category_name}", colour="green"):
            pdf_path = os.path.join(category_source, pdf_filename)

            try:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=DPI)

                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4:
                        img_np = img_np[:, :, :3]

                    # Skip halaman kosong
                    if is_blank_page(img_np):
                        total_skipped += 1
                        continue

                    processed = preprocess_image(pix)

                    # Simpan
                    output_name = f"{os.path.splitext(pdf_filename)[0]}_page_{page_num + 1}.png"
                    output_path = os.path.join(category_output, output_name)

                    if SAVE_GRAYSCALE:
                        cv2.imwrite(output_path, processed)
                    else:
                        cv2.imwrite(output_path, cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR))

                    total_saved += 1

                doc.close()
            except Exception as e:
                print(f"⚠️ Gagal memproses '{pdf_filename}': {e}")

    print("\n========================================================")
    print(f"✅ Dataset selesai dibuat!")
    print(f"📊 Total gambar tersimpan : {total_saved}")
    print(f"🧹 Halaman kosong dilewati : {total_skipped}")
    print(f"📁 Lokasi dataset: '{DATASET_DIR}'")
    print("========================================================")


if __name__ == "__main__":
    convert_pdfs_to_images()