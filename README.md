# Sistem Komputasi Analisis Sentimen Pengunjung Wisata Pantai Bensam Kabupaten Pesawaran

Aplikasi berbasis web untuk melakukan analisis sentimen secara *real-time* terhadap ulasan pengunjung objek wisata **Pantai Benteng Samudra (Bensam), Kabupaten Pesawaran, Lampung**. Proyek ini diimplementasikan menggunakan algoritma *Machine Learning* **Multinomial Naive Bayes** dan dideploy menggunakan framework **Streamlit**.

---

## 📌 Fitur Utama
* **Klasifikasi Sentimen Real-Time:** Memilah teks ulasan baru ke dalam kategori **Positif**, **Netral**, atau **Negatif**.
* **AI Confidence Score:** Menampilkan tingkat keyakinan atau probabilitas akurasi prediksi model terhadap teks input.
* **Metadata Penelitian Terintegrasi:** Ringkasan parameter data uji langsung di halaman utama aplikasi.

---

## 📊 Ringkasan Dataset & Performa Model

Sistem ini dilatih dan diuji menggunakan dataset ulasan Pantai Bensam dengan rincian parameter akademis sebagai berikut:

* **Total Dataset:** 600 ulasan
* **Rasio Proporsi Data:** 80% Data Latih (480 ulasan) & 20% Data Uji (120 ulasan)
* **Akurasi Model Global:** **86.67%**

### Metrik Evaluasi (Classification Report)
| Kelas Sentimen | Precision | Recall | F1-Score | Support (Data Uji) |
| :--- | :---: | :---: | :---: | :---: |
| **Positif** | 0.89 | 0.92 | 0.90 | 94 |
| **Netral** | 0.81 | 0.74 | 0.77 | 22 |
| **Negatif** | 0.85 | 0.78 | 0.81 | 4 |

---

## 🛠️ Alur Pemrosesan Teks (Pipeline)
1. **Data Crawling:** Ekstraksi ulasan dari Google Maps Platform Pantai Bensam.
2. **Text Preprocessing:** Pembersihan data teks (*Case folding, Filtering/Stopword removal, Stemming* bahasa Indonesia).
3. **Feature Extraction:** Transformasi teks bersih menjadi matriks bobot angka menggunakan metode vektorisasi (*Vectorizer*).
4. **Classification:** Perhitungan nilai peluang probabilitas bersyarat menggunakan rumus Teorema Naive Bayes.

---

## 🚀 Cara Menjalankan Aplikasi Secara Lokal

### 1. Clone Repositori
```bash
git clone https://github.com/iniokta/Analisis_Sentimen_Pengunjung_Wisata_PantaiBensam.git
cd NAMA_REPOSITORI
