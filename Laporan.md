# 🔬 Laporan Proyek Machine Learning

# **Analisis Klasifikasi Biner pada Dataset Kanker Payudara**

## Muhammad Alvaro Khikman

---

## 📍 Domain Proyek

### 🔍 Latar Belakang

Kanker payudara adalah salah satu penyebab utama kematian pada wanita di seluruh dunia. Diagnosis dini sangat penting untuk meningkatkan tingkat kelangsungan hidup pasien. Dataset Wisconsin Breast Cancer (Original) dari UCI Machine Learning Repository menyediakan data hasil biopsi sitologi tumor payudara untuk membedakan antara tumor jinak (*Benign*) dan ganas (*Malignant*). Dataset ini terdiri dari 699 sampel dengan 10 fitur morfologis seluler, seperti ketebalan gumpalan (*Clump_thickness*), keseragaman ukuran sel (*Uniformity_of_cell_size*), dan tingkat mitosis (*Mitoses*). Kolom target `Class` memiliki nilai `2` untuk *Benign* dan `4` untuk *Malignant*.

**Tujuan proyek ini** adalah membangun model klasifikasi biner untuk memprediksi apakah tumor payudara bersifat jinak atau ganas berdasarkan fitur-fitur tersebut, sehingga dapat mendukung tenaga medis dalam diagnosis dini.

### ✅ Alasan dan Pentingnya Solusi Otomatis

- **Efisiensi**: Mempercepat proses diagnosis dibandingkan metode manual.
- **Akurasi**: Mengurangi risiko kesalahan manusia dalam interpretasi data biopsi.
- **Skrining Awal**: Memungkinkan deteksi dini sebelum konfirmasi medis lebih lanjut.
- **Integrasi**: Model dapat diintegrasikan ke dalam sistem pendukung keputusan klinis.

### 📚 Referensi

- Wolberg, W.H., & Mangasarian, O.L. (1990). *Multisurface method of pattern separation for medical diagnosis applied to breast cytology*. Proceedings of the National Academy of Sciences, 87(23), 9193–9196.
- UCI Machine Learning Repository: [Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)

---

## 💼 Business Understanding

### 🧩 Problem Statements

1. Bagaimana memprediksi apakah tumor payudara bersifat jinak (*Benign*) atau ganas (*Malignant*) berdasarkan fitur morfologis seluler?
2. Bagaimana membangun model machine learning yang andal dan dapat diinterpretasi untuk mendukung keputusan medis?

### 🎯 Goals

1. Mengembangkan model klasifikasi biner dengan akurasi, presisi, dan recall tinggi (>95%) untuk membedakan tumor *Benign* dan *Malignant*.
2. Memastikan model memiliki recall tinggi untuk kelas *Malignant* guna meminimalkan *false negative* (kasus kanker yang salah diklasifikasikan sebagai jinak), yang krusial dalam konteks medis.
3. Memilih model yang mudah diinterpretasi untuk mendukung penerimaan di lingkungan klinis.

### ⚙️ Solution Statements

1. Menerapkan **DummyClassifier** sebagai baseline untuk mengevaluasi performa minimum.
2. Menggunakan **Logistic Regression** (dengan dan tanpa `class_weight='balanced'`) sebagai model sederhana dan interpretable untuk klasifikasi biner.
3. Menggunakan **KNeighborsClassifier (KNN)** dengan dan tanpa optimasi parameter melalui `GridSearchCV` untuk mengeksplorasi algoritma berbasis jarak.
4. Mengatasi ketidakseimbangan kelas (65.5% *Benign* vs. 34.5% *Malignant*) menggunakan **SMOTE** untuk model KNN.
5. Mengevaluasi model dengan metrik *accuracy*, *precision*, *recall*, dan *F1-score*, dengan fokus pada *recall* untuk kelas *Malignant*.

---

## 📊 Data Understanding

### 🔗 Sumber Dataset

Dataset diambil dari UCI Machine Learning Repository: [Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original).

### 📋 Deskripsi Fitur

Dataset awal memiliki **699 baris** dan **11 kolom** (10 fitur dan 1 target). Berikut adalah deskripsi fitur:

| No | Nama Kolom                  | Tipe Data       | Deskripsi                                                    |
| -- | --------------------------- | --------------- | ------------------------------------------------------------ |
| 1  | Sample_code_number          | int64           | Nomor ID sampel (tidak relevan untuk modeling)               |
| 2  | Clump_thickness             | int64           | Ketebalan gumpalan (1–10)                                   |
| 3  | Uniformity_of_cell_size     | int64           | Keseragaman ukuran sel (1–10)                               |
| 4  | Uniformity_of_cell_shape    | int64           | Keseragaman bentuk sel (1–10)                               |
| 5  | Marginal_adhesion           | int64           | Adhesi marginal (1–10)                                      |
| 6  | Single_epithelial_cell_size | int64           | Ukuran sel epitel tunggal (1–10)                            |
| 7  | Bare_nuclei                 | object → int64 | Inti sel tanpa sitoplasma (1–10)                            |
| 8  | Bland_chromatin             | int64           | Kromatin halus (1–10)                                       |
| 9  | Normal_nucleoli             | int64           | Nukleoli normal (1–10)                                      |
| 10 | Mitoses                     | int64           | Tingkat mitosis (1–10)                                      |
| 11 | Class                       | int64           | Target: 2 = Benign, 4 = Malignant (diencode menjadi 0 dan 1) |

**Catatan**: Kolom `Bare_nuclei` awalnya bertipe `object` karena adanya nilai `"?"` (missing values). Setelah preprocessing, kolom ini dikonversi ke `int64`.

### 📈 Statistik Deskriptif

Statistik deskriptif dari dataset awal (699 baris) menunjukkan distribusi fitur dan target:

| Statistik | Sample_code_number | Clump_thickness | Uniformity_of_cell_size | Uniformity_of_cell_shape | Marginal_adhesion | Single_epithelial_cell_size | Bare_nuclei | Bland_chromatin | Normal_nucleoli | Mitoses | Class |
| --------- | ------------------ | --------------- | ----------------------- | ------------------------ | ----------------- | --------------------------- | ----------- | --------------- | --------------- | ------- | ----- |
| count     | 699                | 699             | 699                     | 699                      | 699               | 699                         | 699         | 699             | 699             | 699     | 699   |
| mean      | 1.07e+06           | 4.42            | 3.13                    | 3.21                     | 2.81              | 3.22                        | 3.54        | 3.44            | 2.87            | 1.59    | 2.69  |
| std       | 6.17e+05           | 2.82            | 3.05                    | 2.97                     | 2.86              | 2.21                        | 3.64        | 2.44            | 3.05            | 1.72    | 0.95  |
| min       | 6.16e+04           | 1               | 1                       | 1                        | 1                 | 1                           | 1           | 1               | 1               | 1       | 2     |
| 25%       | 8.71e+05           | 2               | 1                       | 1                        | 1                 | 2                           | 1           | 2               | 1               | 1       | 2     |
| 50%       | 1.17e+06           | 4               | 1                       | 1                        | 1                 | 2                           | 1           | 3               | 1               | 1       | 2     |
| 75%       | 1.24e+06           | 6               | 5                       | 5                        | 4                 | 4                           | 5           | 5               | 4               | 1       | 4     |
| max       | 1.35e+07           | 10              | 10                      | 10                       | 10                | 10                          | 10          | 10              | 10              | 10      | 4     |

**Analisis**: Semua fitur memiliki rentang nilai 1–10, kecuali `Sample_code_number` yang tidak relevan untuk modeling. Kolom `Class` memiliki nilai 2 (*Benign*) atau 4 (*Malignant*), yang kemudian diencode menjadi 0 dan 1.

### 🥧 Distribusi Kelas

Distribusi kelas dihitung menggunakan `value_counts()` dan divisualisasikan dengan *pie chart*:

![download](https://github.com/user-attachments/assets/f0b68d7d-2959-40fe-8cbd-5e1ab88548dc)

- **Benign (0)**: 65.5% (458 sampel)
- **Malignant (1)**: 34.5% (241 sampel)

**Analisis**: Dataset menunjukkan ketidakseimbangan kelas, dengan kelas *Benign* mendominasi. Hal ini menjadi tantangan untuk model klasifikasi, sehingga diperlukan teknik seperti SMOTE atau `class_weight='balanced'` untuk meningkatkan deteksi kelas *Malignant*.

### 📉 Kondisi Dataset

- **Missing Values**: Kolom `Bare_nuclei` memiliki nilai `"?"` yang dianggap sebagai *missing values*. Setelah konversi ke numerik dan penghapusan baris dengan `NaN`, dataset berkurang dari 699 menjadi 683 baris.
- **Duplikasi**: Terdapat 8 baris duplikat, yang dihapus sehingga dataset menjadi 675 baris.
- **Outlier**: Deteksi outlier menggunakan metode IQR menemukan:
  - `Mitoses`: 119 outlier
  - `Normal_nucleoli`: 28 outlier
  - `Single_epithelial_cell_size`: 52 outlier
  - `Marginal_adhesion`: 59 outlier
    Setelah penghapusan outlier, dataset berkurang menjadi 485 baris.
- **Heatmap Korelasi Fitur**:
Heatmap berikut menampilkan korelasi Pearson antar fitur numerik dan target `Class`. Nilai korelasi berkisar antara -1 hingga 1, di mana:
  
![download (1)](https://github.com/user-attachments/assets/1810fba0-a642-478f-95e5-d3013c43947c)

- Nilai mendekati **1** menunjukkan hubungan linear positif yang kuat.
- Nilai mendekati **0** menunjukkan tidak ada hubungan linear.
- Nilai mendekati **-1** menunjukkan hubungan linear negatif.

---

📌 Korelasi Fitur terhadap Target `Class`

| No | Fitur                         | Korelasi terhadap `Class` | Interpretasi                                                                 |
|----|-------------------------------|----------------------------|------------------------------------------------------------------------------|
| 1  | Uniformity_of_cell_size       | **0.82**                   | Sangat tinggi – sangat relevan dalam membedakan benign vs malignant.        |
| 2  | Uniformity_of_cell_shape      | **0.82**                   | Sangat tinggi – fitur penting untuk klasifikasi kanker.                     |
| 3  | Bare_nuclei                   | **0.82**                   | Sangat tinggi – menunjukkan banyak informasi tentang keganasan sel.         |
| 4  | Bland_chromatin               | 0.76                       | Kuat – tekstur kromatin berperan penting.                                   |
| 5  | Clump_thickness               | 0.72                       | Kuat – ketebalan gumpalan sel berhubungan dengan keganasan.                |
| 6  | Normal_nucleoli               | 0.72                       | Kuat – jumlah dan bentuk nukleoli punya pengaruh dalam klasifikasi.         |
| 7  | Marginal_adhesion             | 0.71                       | Kuat – adhesi antar sel cukup berkontribusi.                                |
| 8  | Single_epithelial_cell_size   | 0.69                       | Sedang – cukup penting untuk model.                                         |
| 9  | Mitoses                       | 0.42                       | Lemah – namun tetap relevan sebagai indikator biologis (laju pembelahan).   |

---

📌 Korelasi Antar-Fitur (Top Pairs)

| Fitur A                    | Fitur B                      | Korelasi | Catatan                                                                 |
|----------------------------|-------------------------------|----------|-------------------------------------------------------------------------|
| Uniformity_of_cell_size    | Uniformity_of_cell_shape      | **0.91** | Sangat tinggi – indikasi multikolinearitas, waspadai saat modeling.    |
| Uniformity_of_cell_size    | Bland_chromatin               | 0.76     | Korelasi kuat, berpotensi memberi informasi serupa.                    |
| Uniformity_of_cell_size    | Single_epithelial_cell_size   | 0.75     | Korelasi tinggi – bisa memberikan sinyal yang redundant.               |
| Uniformity_of_cell_shape   | Bland_chromatin               | 0.74     | Korelasi tinggi – penting untuk diperhitungkan dalam pemilihan fitur.  |

---

✅ Kesimpulan

- Tiga fitur paling berkorelasi dengan target `Class` adalah: `Uniformity_of_cell_size`, `Uniformity_of_cell_shape`, dan `Bare_nuclei`, masing-masing dengan korelasi **0.82**.
- Perlu diwaspadai adanya **multikolinearitas** antara `Uniformity_of_cell_size` dan `Uniformity_of_cell_shape` (**r = 0.91**) jika menggunakan model linear seperti Logistic Regression.
- Fitur `Mitoses` memiliki korelasi paling lemah terhadap `Class`, namun tetap dapat berkontribusi secara kombinatif.

**Tujuan**: Heatmap ini sangat membantu dalam memahami karakteristik dataset untuk memastikan kualitas data sebelum modeling dan mengidentifikasi fitur penting untuk klasifikasi.

---

## 🧹 Data Preparation

### 🔧 Teknik yang Digunakan

1. **Pembersihan Data**:
   - **Konversi `Bare_nuclei`**: Mengubah nilai `"?"` menjadi `NaN` menggunakan `pd.to_numeric(errors='coerce')`, lalu menghapus baris dengan `NaN` (dataset berkurang dari 699 menjadi 683 baris).
   - **Validasi Rentang**: Memastikan nilai `Bare_nuclei` berada dalam rentang 1–10, lalu mengonversi ke `int64`.
   - **Penghapusan Duplikasi**: Menghapus 8 baris duplikat (dataset menjadi 675 baris).
   - **Penghapusan Kolom**: Menghapus `Sample_code_number` karena tidak relevan untuk modeling.
   - **Encoding Target**: Mengubah `Class` dari `2` menjadi `0` (*Benign*) dan `4` menjadi `1` (*Malignant*) menggunakan `map({2: 0, 4: 1})`.
   - **Penghapusan Outlier**: Menggunakan metode IQR untuk menghapus outlier pada kolom seperti `Mitoses`, `Normal_nucleoli`, dan lainnya (dataset menjadi 485 baris).
2. **Pemisahan Data**:
   - Membagi dataset menjadi data pelatihan (80%) dan pengujian (20%) dengan `train_test_split(test_size=0.2, random_state=42, stratify=y)` untuk menjaga distribusi kelas.
3. **Normalisasi Data**:
   - Menggunakan `StandardScaler` untuk menstandarisasi fitur (mean=0, std=1), penting untuk algoritma berbasis jarak seperti KNN.
4. **Penanganan Ketidakseimbangan Kelas**:
   - Menerapkan SMOTE (*Synthetic Minority Over-sampling Technique*) dengan `random_state=42` untuk menyeimbangkan data pelatihan pada model KNN.

### 📌 Alasan

- **Pembersihan Data**: Memastikan data bersih dari *missing values*, duplikasi, dan outlier untuk meningkatkan kualitas input model.
- **Encoding Target**: Mengubah label menjadi format numerik (0 dan 1) agar kompatibel dengan algoritma machine learning.
- **Pemisahan Data**: Memungkinkan evaluasi performa model pada data yang belum dilihat (*unseen data*).
- **Normalisasi**: Menyamakan skala fitur untuk meningkatkan performa algoritma seperti KNN yang sensitif terhadap jarak.
- **SMOTE**: Mengatasi ketidakseimbangan kelas untuk meningkatkan deteksi kelas *Malignant*, yang kritis dalam konteks medis.

**Hasil**: Dataset bersih dengan 485 baris dan 9 fitur numerik siap untuk dilakukan proses modeling.

---

## 🤖 Modeling

### 🔧 Algoritma yang Digunakan

1. **Dummy Classifier**:
   - **Cara Kerja**: Memilih kelas mayoritas (*Benign*) untuk semua prediksi menggunakan strategi `most_frequent`.
   - **Parameter**: `strategy='most_frequent'`.
   - **Alasan**: Digunakan sebagai baseline untuk mengevaluasi performa minimum model yang tidak belajar dari data.
2. **Logistic Regression (Default)**:
   - **Cara Kerja**: Memodelkan probabilitas kelas berdasarkan kombinasi linier fitur, menggunakan fungsi sigmoid untuk klasifikasi biner.
   - **Parameter**: Default (`penalty='l2'`, `C=1.0`, `solver='lbfgs'`).
   - **Alasan**: Sederhana, interpretable, dan efektif untuk dataset dengan hubungan linier seperti dataset ini.
3. **Logistic Regression (Balanced)**:
   - **Cara Kerja**: Sama seperti Logistic Regression default, tetapi dengan bobot kelas seimbang untuk menangani ketidakseimbangan data.
   - **Parameter**: `class_weight='balanced'`, yang memberikan bobot lebih besar pada kelas minoritas (*Malignant*).
   - **Alasan**: Meningkatkan deteksi kelas *Malignant* pada dataset yang tidak seimbang.
4. **KNeighborsClassifier (KNN) Default with SMOTE**:
   - **Cara Kerja**: Mengklasifikasikan data berdasarkan jarak ke *k* tetangga terdekat, menggunakan data pelatihan yang diseimbangkan dengan SMOTE.
   - **Parameter**: Default (`n_neighbors=5`, `metric='minkowski'`, `p=2` untuk jarak Euclidean).
   - **Alasan**: Cocok untuk dataset numerik dan dapat menangkap pola non-linier, dengan SMOTE untuk menangani ketidakseimbangan kelas.
5. **KNeighborsClassifier with GridSearchCV and SMOTE**:
   - **Cara Kerja**: Sama seperti KNN, tetapi dengan optimasi parameter `n_neighbors` (1 hingga 19) menggunakan `GridSearchCV` dengan validasi silang 5-fold.
   - **Parameter**: `n_neighbors=2` (hasil terbaik dari GridSearchCV).
   - **Alasan**: Optimasi parameter untuk meningkatkan performa KNN pada dataset yang telah diseimbangkan dengan SMOTE.

**Tujuan**: Membangun model yang akurat dan mampu mendeteksi kelas *Malignant* dengan recall tinggi, dengan mempertimbangkan ketidakseimbangan kelas.

---

## 📈 Evaluation

### 📊 Metrik Evaluasi

Metrik yang digunakan untuk mengevaluasi model adalah:

- **Accuracy**: Proporsi prediksi yang benar secara keseluruhan.
  $$
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  $$
- **Precision**: Proporsi prediksi positif yang benar.
  $$
  \text{Precision} = \frac{TP}{TP + FP}
  $$
- **Recall (Sensitivitas)**: Proporsi kasus positif yang terdeteksi.
  $$
  \text{Recall} = \frac{TP}{TP + FN}
  $$
- **F1-Score**: Rata-rata harmonis antara *precision* dan *recall*.
  $$
  \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  $$
- **Macro F1-Score**: Rata-rata F1-score untuk kedua kelas, untuk mengevaluasi performa seimbang pada dataset yang tidak seimbang.

**Catatan**: *Recall* untuk kelas *Malignant* sangat penting untuk meminimalkan *false negative*, yang dapat berakibat fatal dalam konteks medis.

### 📊 Perbandingan Performa Model

Berikut adalah hasil evaluasi model berdasarkan data pengujian (97 sampel, 84 *Benign*, 13 *Malignant*):

| Model                          | Accuracy | Precision (Benign) | Recall (Benign) | F1 (Benign) | Precision (Malignant) | Recall (Malignant) | F1 (Malignant) | Macro F1 | Keterangan                                     |
| ------------------------------ | -------- | ------------------ | --------------- | ----------- | --------------------- | ------------------ | -------------- | -------- | ---------------------------------------------- |
| Dummy Classifier               | 0.87     | 0.87               | 1.00            | 0.93        | 0.00                  | 0.00               | 0.00           | 0.46     | Hanya menebak mayoritas                        |
| Logistic Regression            | 0.97     | 0.98               | 0.99            | 0.98        | 0.92                  | 0.85               | 0.88           | 0.93     | Performa sangat stabil                         |
| Logistic Regression (Balanced) | 0.98     | 1.00               | 0.98            | 0.99        | 0.87                  | 1.00               | 0.93           | 0.96     | Sangat baik untuk kelas minoritas              |
| KNN Default with SMOTE         | 0.97     | 1.00               | 0.96            | 0.98        | 0.81                  | 1.00               | 0.90           | 0.94     | Baik, tapi precision Malignant rendah          |
| Best KNN with SMOTE (k=2)      | 0.96     | 0.99               | 0.96            | 0.98        | 0.80                  | 0.92               | 0.86           | 0.92     | Performa seimbang, tapi recall Malignant turun |

### 🏆 Pemilihan Model Terbaik

**Model Terbaik**: Logistic Regression (class_weight='balanced')

**Alasan**:

- **Akurasi Tertinggi (0.98)**: Mengungguli model lain, termasuk Logistic Regression default (0.97), KNN Default with SMOTE (0.97), dan Best KNN with SMOTE (0.96).
- **Recall Malignant Sempurna (1.00)**: Memastikan tidak ada kasus *Malignant* yang terlewat (*false negative*), yang sangat krusial dalam konteks medis.
- **Macro F1-Score Tertinggi (0.96)**: Menunjukkan performa seimbang untuk kedua kelas, meskipun dataset tidak seimbang.
- **F1-Score Malignant (0.93)**: Lebih tinggi dibandingkan model lain (0.90 untuk KNN Default with SMOTE, 0.86 untuk Best KNN with SMOTE, 0.88 untuk Logistic Regression default).
- **Interpretasi Mudah**: Logistic Regression memberikan koefisien fitur yang dapat diinterpretasikan, cocok untuk aplikasi medis.

---

## ✅ Kesimpulan

### 📌 Hubungan dengan Business Understanding

1. **Problem Statements**:
   - **Prediksi Tumor**: Model Logistic Regression (Balanced) berhasil memprediksi tumor *Benign* dan *Malignant* dengan akurasi 0.98 dan recall 1.00 untuk *Malignant*, menjawab kebutuhan untuk klasifikasi akurat berdasarkan fitur morfologis.
   - **Interpretasi Medis**: Model ini sederhana dan interpretable, memungkinkan tenaga medis memahami hubungan antara fitur (misalnya, *Uniformity_of_cell_size*) dan prediksi, mendukung keputusan klinis.
2. **Goals**:
   - **Akurasi dan Recall Tinggi**: Tercapai dengan akurasi 0.98 dan recall *Malignant* 1.00, melebihi target >95%.
   - **Minimalkan False Negative**: Recall 1.00 untuk *Malignant* memastikan tidak ada kasus kanker yang terlewat, sesuai dengan kebutuhan medis.
   - **Interpretasi Klinis**: Logistic Regression (Balanced) dipilih karena transparansi dan kemudahan interpretasi, cocok untuk sistem pendukung keputusan.
3. **Solution Statements**:
   - **DummyClassifier**: Berhasil sebagai baseline dengan akurasi 0.87, menunjukkan perlunya model canggih.
   - **Logistic Regression**: Baik default (akurasi 0.97) maupun balanced (akurasi 0.98) memberikan performa tinggi, dengan balanced lebih unggul untuk kelas *Malignant*.
   - **KNN dengan SMOTE**: KNN Default (akurasi 0.97) dan Best KNN (akurasi 0.96) efektif dengan SMOTE, tetapi kalah dalam recall *Malignant* dan interpretasi dibandingkan Logistic Regression (Balanced).
   - **Evaluasi Metrik**: Semua metrik (*accuracy*, *precision*, *recall*, *F1-score*) dihitung, dengan fokus pada *recall* untuk *Malignant*, menghasilkan model yang andal.

### 📈 Dampak Solusi

- **Efisiensi Diagnosis**: Model dapat memproses data biopsi dengan cepat, mendukung skrining awal.
- **Akurasi Tinggi**: Mengurangi risiko kesalahan diagnosis, terutama untuk kasus *Malignant*.
- **Aplikasi Klinis**: Model Logistic Regression (Balanced) dapat diintegrasikan ke dalam sistem pendukung keputusan untuk membantu dokter dalam diagnosis dini kanker payudara.
- **Skalabilitas**: Model ini dapat diperluas untuk dataset serupa atau dioptimalkan lebih lanjut dengan teknik seperti ensemble learning.

**Kesimpulan Akhir**: Proyek ini berhasil membangun model klasifikasi biner menggunakan dataset Wisconsin Breast Cancer, dengan **Logistic Regression (class_weight='balanced')** sebagai model terbaik. Model ini mencapai akurasi 0.98, recall *Malignant* 1.00, dan Macro F1-score 0.96, menjadikannya solusi yang andal dan interpretable untuk mendeteksi tumor ganas secara akurat. Dengan kemampuan menangani ketidakseimbangan kelas melalui `class_weight='balanced'`, model ini berpotensi mendukung sistem pendeteksian awal kanker payudara secara otomatis, meningkatkan efisiensi dan akurasi diagnosis di lingkungan medis.
