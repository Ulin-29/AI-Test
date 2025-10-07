import re
from rapidfuzz import fuzz

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'100\s*[%o]', ' 100 ', text) 
    text = re.sub(r'[^a-z0-9 ]', ' ', text)  
    text = re.sub(r'\s+', ' ', text) 
    return text.strip()

def fuzzy_contains(text: str, keyword: str, threshold: int = 85) -> bool:
    if len(keyword.split()) <= 2:
        score = fuzz.partial_ratio(keyword.lower(), text, score_cutoff=threshold)
    else:
        score = fuzz.token_set_ratio(keyword.lower(), text, score_cutoff=threshold)
    return score >= threshold

def classify_page_by_keywords(page_text: str) -> str:
    text_norm = normalize_text(page_text)

    # FOTO
    if fuzzy_contains(text_norm, "foto pengukuran opm", threshold=80):
        return "FOTO_PENGUKURAN_OPM"
    if fuzzy_contains(text_norm, "kegiatan uji terima", threshold=80) or fuzzy_contains(text_norm, "dokumentasi test comm", threshold=80):
        return "FOTO_KEGIATAN"
    if fuzzy_contains(text_norm, "material terpasang", threshold=80):
        return "FOTO_MATERIAL"
    if fuzzy_contains(text_norm, "roll meter", threshold=80) or fuzzy_contains(text_norm, "fault locator", threshold=80):
        return "FOTO_ROLL_METER"
    if fuzzy_contains(text_norm, "survey address", threshold=80):
        return "FOTO_SURVEY_ADDRESS"
    
    if fuzzy_contains(text_norm, "form opm", threshold=80) or fuzzy_contains(text_norm, "data pengukuran opm", threshold=80) or fuzzy_contains(text_norm, "hasil ukur opm", threshold=80):
        return "FORM_OPM"
        
    # Berita Acara & Laporan
    if fuzzy_contains(text_norm, "berita acara uji terima") or fuzzy_contains(text_norm, "baut", threshold=90):
        return "BAUT"
    if fuzzy_contains(text_norm, "berita acara commissioning test") or fuzzy_contains(text_norm, "bact", threshold=90):
        return "BACT"
    if fuzzy_contains(text_norm, "laporan hasil pekerjaan") or fuzzy_contains(text_norm, "pekerjaan selesai 100"):
        return "LAPORAN_100%"
    if fuzzy_contains(text_norm, "laporan uji terima") or fuzzy_contains(text_norm, "laporan ut"):
        return "LAPORAN_UT"
    if fuzzy_contains(text_norm, "berita acara barang tiba") or fuzzy_contains(text_norm, "bba", threshold=90):
        return "BERITA_ACARA_BARANG_TIBA"
    if fuzzy_contains(text_norm, "berita acara lapangan"):
        return "BA_LAPANGAN"

    # Administrasi
    if fuzzy_contains(text_norm, "surat permintaan uji terima"):
        return "SURAT_PERMINTAAN"
    if fuzzy_contains(text_norm, "sk team uji terima"):
        return "SK_TEAM"
    if fuzzy_contains(text_norm, "nota dinas pelaksanaan uji"):
        return "NOTA_DINAS"
    if fuzzy_contains(text_norm, "daftar hadir uji terima"):
        return "DAFTAR_HADIR_UT"
    if fuzzy_contains(text_norm, "daftar hadir commissioning test"):
        return "DAFTAR_HADIR_CT"

    # Teknis & Pengukuran
    if fuzzy_contains(text_norm, "as built drawing") or fuzzy_contains(text_norm, "red line drawing") or fuzzy_contains(text_norm, "rld"):
        return "RLD"
    if fuzzy_contains(text_norm, "bill of quantity") or fuzzy_contains(text_norm, "boq", threshold=90):
        if "commissioning" in text_norm or "ct" in text_norm:
            return "BOQ_CT"
        return "BOQ_UT"
    if fuzzy_contains(text_norm, "otdr report") or fuzzy_contains(text_norm, "pengukuran otdr"):
        return "OTDR_REPORT"

    if fuzzy_contains(text_norm, "foto", threshold=75):
        return "EVIDENCE_PHOTO_UMUM"

    return "UNKNOWN"