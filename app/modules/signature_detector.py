import cv2
import numpy as np
import fitz  # PyMuPDF
import os

def _detect_signature_on_image(image_bytes):
    """
    Fungsi internal untuk mendeteksi tanda tangan pada satu gambar.
    Menerima input berupa bytes gambar.
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if area > 500 and w > 0 and h > 0 and (w / h) > 1.5:
                return True
        return False
    except Exception:
        return False

def check_signatures_in_pdf(pdf_path: str):
    """
    Fungsi utama untuk memeriksa keberadaan tanda tangan dalam seluruh file PDF.
    Berhenti setelah tanda tangan pertama ditemukan.
    """
    try:
        doc = fitz.open(pdf_path)
        found_signature = False

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=150)
            image_bytes = pix.tobytes("png")
            
            if _detect_signature_on_image(image_bytes):
                found_signature = True
                break  # Hentikan pencarian jika sudah ditemukan
        doc.close()

        if found_signature:
            return {"status": "Ditemukan"}
        else:
            return {"status": "Tidak Ditemukan"}
    except Exception as e:
        return {"status": "Error", "message": f"Gagal memproses PDF: {str(e)}"}

# Ganti nama fungsi di __main__ untuk konsistensi
if __name__ == '__main__':
    test_pdf_path = 'path/ke/dokumen_anda.pdf'
    if os.path.exists(test_pdf_path):
        result = check_signatures_in_pdf(test_pdf_path)
        print(f"Hasil pengecekan tanda tangan: {result}")
    else:
        print(f"File tidak ditemukan: {test_pdf_path}")