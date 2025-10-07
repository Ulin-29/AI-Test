import re
from typing import List

def generate_summary(pages_text: List[str], page_classes: List[str]) -> str:
    """
    Menghasilkan ringkasan dokumen dengan mencari informasi pada halaman yang sudah 
    diklasifikasikan dan dengan pola Regex yang lebih baik.
    """
    
    def find_text_by_class(target_classes: List[str]):
        for cls in target_classes:
            try:
                index = page_classes.index(cls)
                return pages_text[index]
            except ValueError:
                continue
        return None

    full_text = " ".join(pages_text)
    cleaned_full_text = re.sub(r'\s+', ' ', full_text).strip()

    baut_text = find_text_by_class(['BAUT'])
    bact_text = find_text_by_class(['BACT'])
    laporan_ut_text = find_text_by_class(['LAPORAN_UT'])
    opm_text = find_text_by_class(['FORM_OPM', 'FOTO_PENGUKURAN_OPM'])
    
    main_context_text = baut_text or bact_text or laporan_ut_text or (pages_text[0] if pages_text else "")

    # --- Ekstraksi Informasi yang Disempurnakan ---

    # Judul
    judul_match = re.search(r'(BERITA ACARA UJI TERIMA|PROYEK|PEKERJAAN)(.{0,250})', main_context_text, re.IGNORECASE | re.DOTALL)
    judul = re.sub(r'\s+', ' ', judul_match.group(0).strip()) if judul_match else "Judul dokumen tidak ditemukan."

    # Kontrak
    kontrak_match = re.search(r'(?:KONTRAK|SURAT PESANAN|SP)\s*(?:No\.?|Nomor)?\s*:?\s*(.*?)(?:\n|WITEL|PELAKSANA)', main_context_text, re.IGNORECASE | re.DOTALL)
    kontrak = f"Kontrak No. {re.sub(r'<[^>]+>', '', kontrak_match.group(1)).strip()}" if kontrak_match else "Nomor kontrak tidak ditemukan."

    # Lokasi
    lokasi_match = re.search(r'LOKASI\s*:?\s*(.*?)(?:\n|PELAKSANA|WITEL)', main_context_text, re.IGNORECASE | re.DOTALL)
    lokasi = re.sub(r'\s+', ' ', lokasi_match.group(1)).strip() if lokasi_match else "tidak disebutkan."

    # Pelaksana (Dinamis)
    pelaksana_match = re.search(r'PELAKSANA\s*:?\s*(.*?)(?:\n|TANGGAL|PADA HARI INI)', main_context_text, re.IGNORECASE | re.DOTALL)
    pelaksana = re.sub(r'\s+', ' ', pelaksana_match.group(1)).strip() if pelaksana_match else "PT. Telkom Akses"
    pemilik_pekerjaan = "PT. Telkom Indonesia, Tbk." # Ini biasanya tetap

    # Tanggal
    tanggal_match = re.search(r'(\d{1,2}\s*(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s*\d{4})', main_context_text, re.IGNORECASE)
    tanggal = tanggal_match.group(0) if tanggal_match else "tidak disebutkan."

    # Hasil Redaman
    search_text_redaman = opm_text or cleaned_full_text
    redaman_match = re.search(r'redaman.*?(\d+[.,]\d+)\s*dB', search_text_redaman, re.IGNORECASE)
    redaman = f"Hasil uji redaman: {redaman_match.group(1)} dB" if redaman_match else "Data redaman tidak tersedia."
    
    # Hasil Grounding
    grounding_match = re.search(r'(\d+[.,]\d+)\s*Ohm', cleaned_full_text, re.IGNORECASE)
    grounding = f"Pengukuran grounding: {grounding_match.group(1)} Ohm" if grounding_match else "Data grounding tidak tersedia."
    
    # Kesimpulan
    kesimpulan_match = re.search(r'(DITERIMA|OK|BAIK|LULUS|SESUAI)', main_context_text, re.IGNORECASE)
    kesimpulan = f"Hasil pekerjaan dinyatakan <strong>{kesimpulan_match.group(1).upper()}</strong>." if kesimpulan_match else "Status pekerjaan tidak dinyatakan secara eksplisit."

    # --- Format HTML ---
    summary_html = f"""
    <div class="space-y-4">
        <div>
            <h3 class="text-lg font-semibold text-gray-800 mb-2">📌 Judul</h3>
            <p class="text-gray-600">{judul}</p>
        </div>
        <div>
            <h3 class="text-lg font-semibold text-gray-800 mb-2">📄 Latar Belakang</h3>
            <ul class="list-disc list-inside space-y-1 text-gray-600">
                <li>{kontrak}</li>
                <li>Lingkup pekerjaan adalah instalasi OSP FTTH di wilayah {lokasi}</li>
                <li>Pelaksana: <strong>{pelaksana}</strong>, Pemilik Pekerjaan: <strong>{pemilik_pekerjaan}</strong></li>
            </ul>
        </div>
        <div>
            <h3 class="text-lg font-semibold text-gray-800 mb-2">🎯 Tujuan</h3>
            <ul class="list-disc list-inside space-y-1 text-gray-600">
                <li>Memastikan hasil pekerjaan sesuai dengan spesifikasi kontrak.</li>
                <li>Memverifikasi kelengkapan administrasi (BAUT, laporan uji, BoQ).</li>
                <li>Menilai kualitas instalasi jaringan fiber optic.</li>
            </ul>
        </div>
        <div>
            <h3 class="text-lg font-semibold text-gray-800 mb-2">🔍 Poin-Poin Penting</h3>
            <ul class="list-disc list-inside space-y-1 text-gray-600">
                <li>Dokumen administratif lengkap dan sah.</li>
                <li>{redaman}</li>
                <li>{grounding}</li>
                <li>Pelaksanaan uji terima berlangsung pada {tanggal}</li>
            </ul>
        </div>
        <div>
            <h3 class="text-lg font-semibold text-gray-800 mb-2">✅ Kesimpulan / Penutup</h3>
            <ul class="list-disc list-inside space-y-1 text-gray-600">
                <li>{kesimpulan}</li>
                <li>Dokumen pendukung dan hasil pengukuran teknis dianggap sah dan memenuhi standar Telkom.</li>
            </ul>
        </div>
    </div>
    """
    return summary_html