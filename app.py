import streamlit as st
import joblib
import time

# ==============================================================================
# 1. KONFIGURASI HALAMAN DAN TEMA FORMAL HIGH-CONTRAST
# ==============================================================================
st.set_page_config(
    page_title="Bensam Analytics | Pantai Benteng Samudra", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# CSS Tingkat Tinggi untuk Tampilan Profesional, Bersih, dan Kontras Tinggi
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        color: #111111 !important;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }

    /* Tombol Utama - Gradien Biru Profesional */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0046BF 0%, #0066FF 100%) !important;
        color: #FFFFFF !important;
        border-radius: 8px;
        border: none;
        padding: 16px 28px;
        font-weight: 700;
        font-size: 16px;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(0, 102, 255, 0.15);
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0, 102, 255, 0.3);
        color: #FFFFFF !important;
    }

    /* Kotak Informasi Dengan Garis Batas Kiri */
    .premium-card {
        background: #FFFFFF !important;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        border-left: 6px solid #0066FF;
        padding: 24px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.02);
        margin-bottom: 20px;
    }
    
    /* Area Input Teks */
    textarea {
        border-radius: 8px !important;
        border: 2px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
        font-size: 15px !important;
        color: #111111 !important;
    }
    textarea:focus {
        border-color: #0066FF !important;
        box-shadow: 0 0 0 4px rgba(0, 102, 255, 0.15) !important;
    }

    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. HEADER UTAMA (SINKRON DENGAN JUDUL PENELITIAN PANTAI BENSAM)
# ==============================================================================
st.markdown("""
    <div style='text-align: center; padding: 30px 0 40px 0;'>
        <p style='color: #0066FF; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 5px; font-size: 14px;'>DASBOR ANALITIK DATA SAINS</p>
        <h1 style='font-weight: 800; font-size: 42px; margin-top: 0; color: #0F172A;'>Sistem Komputasi Analisis Sentimen</h1>
        <p style='color: #475569; font-size: 16px; max-width: 800px; margin: 10px auto 0 auto; font-weight: 500; line-height: 1.6;'>
            Uji Klasifikasi Komparatif terhadap Dataset Opini Publik Pantai Benteng Samudra (Bensam) Kabupaten Pesawaran Berbasis Algoritma Naive Bayes.
        </p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. MEMUAT MODEL MACHINE LEARNING
# ==============================================================================
@st.cache_resource
def load_model_ai():
    model = joblib.load('model_naive_bayes.pkl')
    vec = joblib.load('vectorizer.pkl')
    return model, vec

try:
    model_ai, vectorizer = load_model_ai()
except:
    st.error("Gagal memuat file model. Pastikan file model_naive_bayes.pkl dan vectorizer.pkl berada di folder yang sama.")

# ==============================================================================
# 4. BAGIAN INPUT TEKS ULASAN (DI ATAS UTAMA & RESPONSIVE)
# ==============================================================================
st.markdown("### Modul Pengujian Real-Time")
col_in, col_out = st.columns([1, 1.2])

with col_in:
    ulasan_input = st.text_area(
        "Masukkan kalimat ulasan untuk menguji akurasi prediksi model:", 
        height=110, 
        placeholder="Contoh: Pantai Bensam sangat bersih, pemandangannya indah dan cocok untuk liburan keluarga..."
    )
    btn_analisis = st.button("Jalankan Analisis Sistem")

with col_out:
    if btn_analisis:
        if ulasan_input.strip() == "":
            st.toast("Kolom teks tidak boleh kosong.")
        else:
            with st.spinner("Memproses transformasi matriks teks..."):
                time.sleep(0.4)
                teks_vektor = vectorizer.transform([ulasan_input])
                prediksi = model_ai.predict(teks_vektor)
                probabilitas = model_ai.predict_proba(teks_vektor)[0]
            
            if prediksi[0] == 'Positif':
                confidence = probabilitas[2] if len(probabilitas) > 2 else probabilitas[1]
                st.markdown("""
                    <div style='background-color: #DCFCE7; padding: 22px; border-radius: 8px; border-left: 8px solid #16A34A; height: 110px;'>
                        <h3 style='margin: 0; color: #14532D; font-weight:800; font-size:18px;'>SENTIMEN TERDETEKSI: POSITIF</h3>
                        <p style='margin: 8px 0 0 0; color: #166534; font-size: 14px; font-weight:500;'>Kalimat menunjukkan indikasi kepuasan pengunjung terhadap aspek Pantai Bensam.</p>
                    </div>
                """, unsafe_allow_html=True)
            elif prediksi[0] == 'Negatif':
                confidence = probabilitas[0]
                st.markdown("""
                    <div style='background-color: #FEE2E2; padding: 22px; border-radius: 8px; border-left: 8px solid #DC2626; height: 110px;'>
                        <h3 style='margin: 0; color: #7F1D1D; font-weight:800; font-size:18px;'>SENTIMEN TERDETEKSI: NEGATIF</h3>
                        <p style='margin: 8px 0 0 0; color: #991B1B; font-size: 14px; font-weight:500;'>Kalimat menunjukkan ulasan kritis, keluhan, atau ketidakpuasan operasional di area pantai.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                confidence = probabilitas[1] if len(probabilitas) > 2 else probabilitas[0]
                st.markdown("""
                    <div style='background-color: #FEF3C7; padding: 22px; border-radius: 8px; border-left: 8px solid #D97706; height: 110px;'>
                        <h3 style='margin: 0; color: #78350F; font-weight:800; font-size:18px;'>SENTIMEN TERDETEKSI: NETRAL</h3>
                        <p style='margin: 8px 0 0 0; color: #92400E; font-size: 14px; font-weight:500;'>Kalimat bersifat objektif, seimbang, atau memberikan informasi umum seputar pantai.</p>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown(f"<p style='font-size: 14px; margin-top: 10px; text-align: center; font-weight:600;'>AI Confidence Score: <span style='color:#0066FF; font-size:16px; font-weight:800;'>{confidence * 100:.2f}%</span></p>", unsafe_allow_html=True)
            st.progress(float(confidence))
    else:
        st.info("Sistem siap menerima input data teks ulasan untuk diuji.")

# ==============================================================================
# 5. DASHBOARD PARAMETER MODEL EVALUASI (BAGIAN KOTAK GRAFIK TELAH DIHAPUS)
# ==============================================================================
st.write("---")
st.markdown("### Ringkasan Statistik dan Parameter Model")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
        <div class='premium-card' style='text-align: center; height: 220px; display: flex; flex-direction: column; justify-content: center;'>
            <p style='color: #475569; font-size: 13px; font-weight: 700; letter-spacing: 1px; margin-bottom: 5px;'>RASIO AKURASI SISTEM</p>
            <h1 style='font-size: 58px; font-weight: 800; color: #0066FF; margin: 0;'>86.7<span style='font-size: 24px; color: #475569;'>%</span></h1>
            <div style='margin-top: 10px;'>
                <span style='background-color: #DCFCE7; color: #15803D; padding: 6px 16px; border-radius: 50px; font-weight: 700; font-size: 13px;'>Klasifikasi: Valid / Optimal</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class='premium-card' style='height: 220px; display: flex; flex-direction: column; justify-content: center;'>
            <p style='color: #475569; font-size: 13px; font-weight: 700; letter-spacing: 1px; margin-bottom: 10px;'>METADATA PENELITIAN</p>
            <div style='border-bottom: 1px solid #E2E8F0; padding-bottom: 6px; margin-bottom: 6px;'>
                <p style='margin:0; font-size:13px; color:#64748B;'>Algoritma Klasifikasi: <span style='font-weight:700; color:#0F172A;'>Naive Bayes Classifier</span></p>
            </div>
            <div style='border-bottom: 1px solid #E2E8F0; padding-bottom: 6px; margin-bottom: 6px;'>
                <p style='margin:0; font-size:13px; color:#64748B;'>Total Sampel Data Uji: <span style='font-weight:700; color:#0066FF;'>120 Ulasan (Support 20%)</span></p>
            </div>
            <div>
                <p style='margin:0; font-size:13px; color:#64748B;'>Lokasi Objek Wisata: <span style='font-weight:700; color:#0F172A;'>Pantai Bensam, Pesawaran</span></p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 6. WAWASAN AKADEMIS STRATEGIS
# ==============================================================================
st.write("---")
st.markdown("### Interpretasi Data & Kesimpulan Eksekutif")
st.write("Ekstraksi wawasan berdasarkan kecenderungan data ulasan Pantai Benteng Samudra (Bensam):")

c_ins1, c_ins2, c_ins3 = st.columns(3)

with c_ins1:
    st.markdown("""
        <div class='premium-card' style='height: 240px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='font-size: 12px; font-weight: 700; color: #0066FF; letter-spacing:0.5px;'>INDIKATOR POSITIF</span>
                <span style='background-color: #E0F2FE; color: #0369A1; padding: 4px 12px; border-radius: 50px; font-weight: 700; font-size: 11px;'>UTAMA</span>
            </div>
            <h3 style='margin: 15px 0; font-weight:700; color:#0F172A;'>Kondisi Alam & Pantai</h3>
            <p style='color: #334155; font-size: 13px; line-height:1.5; font-weight:500;'>Sebagian besar ulasan positif terfokus pada keindahan garis pantai, kebersihan area pasir, serta kecocokan destinasi sebagai tempat rekreasi keluarga.</p>
        </div>
    """, unsafe_allow_html=True)

with c_ins2:
    st.markdown("""
        <div class='premium-card' style='height: 240px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='font-size: 12px; font-weight: 700; color: #0066FF; letter-spacing:0.5px;'>MATRIKS NETRAL</span>
                <span style='background-color: #E0F2FE; color: #0369A1; padding: 4px 12px; border-radius: 50px; font-weight: 700; font-size: 11px;'>INFORMASI</span>
            </div>
            <h3 style='margin: 15px 0; font-weight:700; color:#0F172A;'>Rute & Fasilitas Umum</h3>
            <p style='color: #334155; font-size: 13px; line-height:1.5; font-weight:500;'>Ulasan netral didominasi oleh deskripsi rute menuju lokasi, tarif parkir standar, serta penyediaan fasilitas pendukung dasar seperti gazebo dan toilet.</p>
        </div>
    """, unsafe_allow_html=True)

with c_ins3:
    st.markdown("""
        <div class='premium-card' style='height: 240px; border-left: 6px solid #DC2626;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='font-size: 12px; font-weight: 700; color: #DC2626; letter-spacing:0.5px;'>REKOMENDASI PERBAIKAN</span>
                <span style='background-color: #FEE2E2; color: #991B1B; padding: 4px 12px; border-radius: 50px; font-weight: 700; font-size: 11px;'>KRITIS</span>
            </div>
            <h3 style='margin: 15px 0; font-weight:700; color:#0F172A;'>Akses Jalan & Layanan</h3>
            <p style='color: #334155; font-size: 13px; line-height:1.5; font-weight:500;'>Aspek yang muncul pada ulasan negatif menyoroti perlunya optimalisasi akses jalan masuk pada hari libur nasional serta peningkatan manajemen kebersihan berkala.</p>
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 7. FOOTER FORMAL PORTAL AKADEMIK
# ==============================================================================
st.markdown("""
    <div style='background-color: #E2E8F0; padding: 30px; margin: 50px -15px -15px -15px; border-radius: 8px 8px 0 0;'>
        <div style='display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: auto;'>
            <div>
                <h4 style='color: #0F172A; margin: 0; font-weight:700;'>Bensam Analytics Engine</h4>
                <p style='color: #475569; font-size: 13px; margin: 4px 0 0 0;'>Eksplorasi Riset Ilmu Komputer dan Sains Data</p>
            </div>
            <div style='text-align: right; color: #475569; font-size: 13px; font-weight:600;'>
                <span>Sumber Data: Google Maps Platform (Review Ekstraksi)</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)