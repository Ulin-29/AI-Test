import cv2
import pytesseract
import fitz  # PyMuPDF
import os
import re

def ocr_image(image_path: str) -> str:
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    _, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
    return pytesseract.image_to_string(binary_img, lang='ind')

def extract_data_from_document(doc_path: str) -> dict:
    raw_text = ""
    try:
        if doc_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            raw_text = ocr_image(doc_path)
        elif doc_path.lower().endswith('.pdf'):
            doc = fitz.open(doc_path)
            full_text_from_pdf = ""
            for i, page in enumerate(doc):
                text_from_page = page.get_text("text")
                if not text_from_page.strip():
                    pix = page.get_pixmap()
                    temp_image_path = f"temp_page_{i+1}.png"
                    pix.save(temp_image_path)
                    text_from_page = ocr_image(temp_image_path)
                    os.remove(temp_image_path)
                full_text_from_pdf += text_from_page + "\n"
            raw_text = full_text_from_pdf
            doc.close()
        else:
            return None

        extracted_data = {}
        pola_jumlah = re.search(r"Jumlah: (Rp ?[\d.,]+)", raw_text, re.IGNORECASE)
        if pola_jumlah: extracted_data['jumlah_pembayaran'] = pola_jumlah.group(1)

        pola_tanggal = re.search(r"Tanggal: (.*)", raw_text, re.IGNORECASE)
        if pola_tanggal: extracted_data['tanggal_dokumen'] = pola_tanggal.group(1).strip()
            
        pola_nama = re.search(r"Nama: (.*)", raw_text, re.IGNORECASE)
        if pola_nama:
            nama_ditemukan = pola_nama.group(1).strip()
            extracted_data['nama_peserta'] = "KOSONG" if "___" in nama_ditemukan else nama_ditemukan

        return extracted_data

    except Exception as e:
        print(f"[ERROR] Gagal memproses dokumen: {e}")
        return None