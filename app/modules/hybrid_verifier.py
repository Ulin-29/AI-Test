import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import pytesseract
import json
import keras

@keras.saving.register_keras_serializable()
def grayscale_to_rgb(x):
    """Konversi gambar grayscale (1 channel) ke RGB (3 channel)."""
    return tf.image.grayscale_to_rgb(x)

# --- Konfigurasi ---
# Pastikan path ini sesuai dengan lokasi file di proyek Anda
MODEL_PATH = 'document_classifier_model.keras'
CLASS_NAMES_PATH = 'class_names.json'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# --- 1. Memuat Model AI dan Nama Kelas ---
# Kode ini memuat model klasifikasi gambar yang sudah Anda latih.
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
except Exception as e:
    print(f"Error: Gagal memuat model atau class_names.json. Pastikan file ada di path yang benar. Detail: {e}")
    # Keluar jika model tidak bisa dimuat
    exit()

# --- 2. Peta Keyword  ---
KEYWORD_MAP = {
    # --- Berita Acara & Laporan ---
    "BAUT": ["berita acara uji terima", "baut"],
    "BACT": ["berita acara commissioning test", "bact"],
    "LAPORAN_100%": ["laporan hasil pekerjaan", "pekerjaan selesai 100"],
    "LAPORAN_UT": ["laporan uji terima", "laporan ut"],
    "BERITA_ACARA_BARANG_TIBA": ["berita acara barang tiba", "bba"],
    "BA_LAPANGAN": ["berita acara lapangan"],

    # --- Administrasi ---
    "SURAT_PERMINTAAN": ["surat permintaan uji terima"],
    "SK_TEAM": ["sk team uji terima", "penunjukan team"],
    "NOTA_DINAS": ["nota dinas pelaksanaan uji"],
    "DAFTAR_HADIR_UT": ["daftar hadir uji terima"],
    "DAFTAR_HADIR_CT": ["daftar hadir commissioning test"],

    # --- Teknis & Pengukuran ---
    "RLD": ["as built drawing", "red line drawing", "rld"],
    "BOQ_UT": ["bill of quantity", "boq"],
    "BOQ_CT": ["bill of quantity", "boq", "commissioning"],
    "OTDR_REPORT": ["otdr report", "pengukuran otdr"],
    "FORM_OPM": ["form opm", "data pengukuran opm", "hasil ukur opm"],

    # --- Kategori Foto/Dokumentasi ---
    "FOTO_PENGUKURAN_OPM": ["foto pengukuran opm"],
    "FOTO_KEGIATAN": ["kegiatan uji terima", "dokumentasi test comm"],
    "FOTO_MATERIAL": ["material terpasang"],
    "FOTO_ROLL_METER": ["roll meter", "fault locator"],
    "FOTO_SURVEY_ADDRESS": ["survey address"],
    
    # --- Lain-lain (jika ada di model Anda) ---
    "SIGNATURE": ["tanda tangan", "mengetahui", "disetujui oleh"],
    
}

# --- 3. Fungsi Preprocessing Gambar ---
# Fungsi ini menyiapkan gambar agar formatnya sama seperti saat Anda melatih model.
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    # Konversi ke grayscale lalu kembali ke BGR (3 channel) agar sesuai input model
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_resized = cv2.resize(img_bgr, target_size)
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = img_array / 255.0  # Normalisasi
    return img_array

# --- 4. Fungsi Ekstraksi Teks (OCR) ---
# Fungsi ini "membaca" semua teks yang ada di dalam gambar.
def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        # Ekstrak teks menggunakan bahasa Indonesia
        text = pytesseract.image_to_string(image, lang='ind')
        # Ubah semua teks menjadi huruf kecil agar pencarian keyword tidak sensitif (NIK vs nik)
        return text.lower()
    except Exception as e:
        print(f"Error saat proses OCR: {e}")
        return ""

# --- 5. Fungsi Inti Verifikasi Hybrid ---
def verify_document_hybrid(image_path):
    """
    Menggabungkan prediksi Model AI dengan verifikasi Keyword untuk hasil yang akurat.
    """
    # === LANGKAH 1: Prediksi Jenis Dokumen dengan Model AI ===
    print("Memulai prediksi dengan Model AI...")
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[str(predicted_index)]
    confidence = np.max(predictions[0])
    print(f"-> Hasil Model AI: '{predicted_class}' (Kepercayaan: {confidence:.2%})")

    # === LANGKAH 2: Cek Apakah Keyword untuk Dokumen Ini Sudah Didefinisikan ===
    if predicted_class not in KEYWORD_MAP:
        return {
            "status": "Gagal",
            "prediction": predicted_class,
            "reason": f"Definisi keyword untuk '{predicted_class}' belum ada di KEYWORD_MAP."
        }

    # === LANGKAH 3: Ekstrak Seluruh Teks dari Gambar ===
    print("\nMemulai ekstraksi teks dengan OCR...")
    extracted_text = extract_text_from_image(image_path)
    if not extracted_text:
        return {
            "status": "Gagal",
            "prediction": predicted_class,
            "reason": "OCR tidak dapat mendeteksi teks apa pun pada gambar."
        }
    print("-> Teks berhasil diekstrak.")

    # === LANGKAH 4: Cocokkan Keyword dengan Teks Hasil OCR ===
    print(f"\nMemeriksa keyword yang wajib ada untuk '{predicted_class}'...")
    required_keywords = KEYWORD_MAP[predicted_class]
    found_keywords = []
    missing_keywords = []

    for keyword in required_keywords:
        if keyword in extracted_text:
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    print(f"-> Keyword ditemukan: {found_keywords}")
    print(f"-> Keyword tidak ditemukan: {missing_keywords}")

    # === LANGKAH 5: Buat Keputusan Akhir ===
    # Aturan: Jika lebih dari 70% keyword wajib ditemukan, maka dokumen dianggap terverifikasi.
    # Anda bisa membuat aturan ini lebih ketat (misal: 100% atau `len(missing_keywords) == 0`).
    if len(found_keywords) / len(required_keywords) >= 0.7:
        result = {
            "status": "Terverifikasi",
            "prediction": predicted_class,
            "reason": f"Prediksi '{predicted_class}' dikonfirmasi dengan ditemukannya keyword: {found_keywords}."
        }
    else:
        result = {
            "status": "Tidak Terverifikasi",
            "prediction": predicted_class,
            "reason": f"Prediksi '{predicted_class}' tidak dapat dikonfirmasi karena keyword penting tidak ada: {missing_keywords}."
        }

    return result

# --- Contoh Penggunaan Langsung dari Terminal ---
if __name__ == '__main__':
    # Ganti path ini dengan path gambar yang ingin Anda uji
    test_image_path = 'path/ke/gambar_dokumen.png'
    try:
        hasil_verifikasi = verify_document_hybrid(test_image_path)
        print("\n=====================")
        print(" HASIL AKHIR VERIFIKASI ")
        print("=====================")
        for key, value in hasil_verifikasi.items():
            print(f"- {key.capitalize()}: {value}")
    except FileNotFoundError:
        print(f"\n[ERROR] File tidak ditemukan di '{test_image_path}'. Mohon periksa kembali path file Anda.")