import os
import fitz  # PyMuPDF
from tqdm import tqdm
import cv2
import numpy as np

# --- Konfigurasi ---
# Folder sumber berisi sub-folder kategori dengan PDF
PDF_SOURCE_DIR = 'pdf_sources'

# Folder tujuan untuk menyimpan gambar hasil konversi
DATASET_DIR = 'dataset'

# Kualitas gambar (dots per inch) untuk hasil detail lebih baik
DPI = 300


def preprocess_image(pix):
    """
    Membersihkan dan meningkatkan kualitas gambar hasil konversi PDF.
    """
    # 1. Konversi gambar dari format pixmap ke format yang bisa dibaca OpenCV
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

    # 2. Konversi ke grayscale (skala abu-abu)
    gray_image = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # 3. Terapkan Adaptive Thresholding untuk membuat gambar hitam-putih yang tajam
    processed_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return processed_image


def convert_pdfs_to_images():
    """
    Membaca semua PDF dari PDF_SOURCE_DIR, mengubah setiap halaman menjadi gambar berkualitas tinggi,
    membersihkannya, dan menyimpannya di DATASET_DIR.
    """
    if not os.path.exists(PDF_SOURCE_DIR):
        print(f"❌ Error: Folder sumber '{PDF_SOURCE_DIR}' tidak ditemukan.")
        print("Pastikan Anda sudah membuat folder tersebut dan mengisinya dengan sub-folder kategori PDF.")
        return

    print(f"🚀 Memulai konversi PDF dari '{PDF_SOURCE_DIR}' ke gambar di '{DATASET_DIR}'...")

    for category_name in os.listdir(PDF_SOURCE_DIR):
        category_source_path = os.path.join(PDF_SOURCE_DIR, category_name)

        if not os.path.isdir(category_source_path):
            continue

        print(f"\n📂 Memproses kategori: {category_name}")

        category_dest_path = os.path.join(DATASET_DIR, category_name)
        os.makedirs(category_dest_path, exist_ok=True)

        pdf_files = [f for f in os.listdir(category_source_path) if f.lower().endswith('.pdf')]

        for pdf_filename in tqdm(pdf_files, desc=f"  → Konversi {category_name}"):
            pdf_path = os.path.join(category_source_path, pdf_filename)

            try:
                doc = fitz.open(pdf_path)

                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=DPI)

                    processed_image = preprocess_image(pix)

                    output_image_name = f"{os.path.splitext(pdf_filename)[0]}_page_{page_num + 1}.png"
                    output_image_path = os.path.join(category_dest_path, output_image_name)

                    cv2.imwrite(output_image_path, processed_image)

                doc.close()

            except Exception as e:
                print(f"\n⚠️ [Peringatan] Gagal memproses file '{pdf_filename}': {e}")

    print("\n========================================================")
    print("✅ Proses persiapan dataset selesai!")
    print(f"📁 Semua gambar siap di folder '{DATASET_DIR}' untuk pelatihan.")
    print("========================================================")


if __name__ == '__main__':
    convert_pdfs_to_images()
