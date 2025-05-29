# Laporan Proyek Machine Learning - Food Recommendation System
## Muhammad Alvaro Khikman

## Project Overview

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi makanan yang dapat memberikan saran hidangan kepada pengguna. Di era digital saat ini, dengan melimpahnya pilihan kuliner yang tersedia melalui berbagai platform online dan aplikasi, pengguna seringkali menghadapi *information overload* dan kesulitan dalam memutuskan makanan apa yang ingin mereka coba atau yang paling sesuai dengan preferensi dan kebutuhan diet mereka (Trattner & Elsweiler, 2017). Sistem rekomendasi makanan dapat memainkan peran krusial dalam membantu pengguna menyaring pilihan tersebut, memperkenalkan mereka pada hidangan baru yang mungkin mereka sukai, dan pada akhirnya meningkatkan pengalaman kuliner serta mendukung pilihan makanan yang lebih sehat.

Masalah ini penting untuk diselesaikan karena sistem rekomendasi yang efektif dapat memberikan nilai tambah yang signifikan bagi berbagai pemangku kepentingan, termasuk platform layanan makanan, aplikasi pengiriman, blog resep, dan bahkan pengguna individu yang mencari inspirasi kuliner. Dengan rekomendasi yang akurat dan personal, platform dapat meningkatkan *engagement* pengguna, loyalitas, dan kepuasan. Berbagai pendekatan telah dieksplorasi dalam literatur, dengan dua yang utama adalah Content-Based Filtering, yang merekomendasikan item berdasarkan atribut item itu sendiri (seperti bahan, jenis masakan), dan Collaborative Filtering, yang merekomendasikan item berdasarkan pola perilaku (misalnya, rating) dari pengguna serupa (Ge, Liu, & Wang, 2019). Proyek ini akan mengimplementasikan dan mengevaluasi kedua pendekatan tersebut.

Referensi:
*   Trattner, C., & Elsweiler, D. (2017). Food Recommender Systems: Important Factors, Current Approaches and Future Challenges. *Frontiers in ICT, 4*, 9. [https://www.frontiersin.org/articles/10.3389/fict.2017.00009/full](https://www.frontiersin.org/articles/10.3389/fict.2017.00009/full) (Open Access)
*   Ge, M., Liu, J., & Wang, H. (2019). A review of food recommendation system. *Journal of Physics: Conference Series, 1237*(2), 022036. [https://iopscience.iop.org/article/10.1088/1742-6596/1237/2/022036/pdf](https://iopscience.iop.org/article/10.1088/1742-6596/1237/2/022036/pdf) (Open Access)

## Business Understanding

### Problem Statements

1.  Bagaimana cara merekomendasikan makanan kepada pengguna berdasarkan kemiripan atribut makanan (seperti jenis masakan, bahan, deskripsi, status vegetarian)?
2.  Bagaimana cara merekomendasikan makanan kepada pengguna berdasarkan preferensi dan pola rating dari pengguna lain yang memiliki selera serupa?
3.  Bagaimana cara mengevaluasi performa dari kedua pendekatan sistem rekomendasi yang dibangun?

### Goals

1.  Mengembangkan sistem rekomendasi makanan menggunakan pendekatan Content-Based Filtering yang memanfaatkan fitur tekstual dari makanan.
2.  Mengembangkan sistem rekomendasi makanan menggunakan pendekatan Collaborative Filtering (menggunakan SVD) yang memanfaatkan histori rating pengguna.
3.  Menyajikan top-N rekomendasi untuk kedua sistem dan mengevaluasinya menggunakan metrik yang sesuai (kualitatif untuk Content-Based, dan kuantitatif seperti RMSE/MAE untuk Collaborative Filtering).

### Solution statements
1.  **Content-Based Filtering:** Menggunakan TfidfVectorizer untuk mengubah fitur teks gabungan (nama, jenis, status vegetarian, deskripsi) menjadi representasi numerik, kemudian menghitung kemiripan antar makanan menggunakan Cosine Similarity.
2.  **Collaborative Filtering:** Menggunakan algoritma Singular Value Decomposition (SVD) dari library Surprise, yang merupakan teknik faktorisasi matriks untuk menemukan pola laten dalam data rating pengguna.

## Data Understanding

Dataset yang digunakan dalam proyek ini bersumber dari Kaggle: Food Recommendation System.
https://www.kaggle.com/datasets/schemersays/food-recommendation-system

Dataset ini terdiri dari dua file utama:
1.  `food_df` (awalnya `1662574418893344.csv`): Berisi informasi detail tentang makanan. Awalnya terdapat 400 makanan dengan 5 kolom.
2.  `ratings_df` (awalnya `ratings.csv`): Berisi data rating yang diberikan pengguna terhadap makanan. Awalnya terdapat 512 rating dengan 3 kolom.

Setelah proses data preparation (penghapusan nilai NaN dan penyesuaian tipe data), diperoleh 511 entri rating yang valid. Rata-rata rating keseluruhan adalah sekitar 5.438356 pada skala 1-10.

Variabel-variabel pada dataset adalah sebagai berikut:

**Pada `food_df`:**
*   `Food_ID`: Identifikasi unik untuk setiap makanan (Integer).
*   `Name`: Nama makanan (Teks/Object).
*   `C_Type`: Jenis masakan (misalnya, 'Healthy Food', 'Snack', 'Indian', 'Mexican') (Teks/Object).
*   `Veg_Non`: Status vegetarian makanan ('veg' atau 'non-veg') (Teks/Object).
*   `Describe`: Deskripsi singkat mengenai makanan (Teks/Object).

**Pada `ratings_df`:**
*   `User_ID`: Identifikasi unik untuk setiap pengguna (Float, kemudian diubah ke Integer).
*   `Food_ID`: Identifikasi unik untuk makanan yang dirating (Float, kemudian diubah ke Integer).
*   `Rating`: Skor rating yang diberikan pengguna untuk makanan (skala 1-10) (Float, kemudian diubah ke Integer).

**Exploratory Data Analysis (EDA) Highlights:**
*   **Dataset Makanan (`food_df`):**
    *   Terdiri dari 400 baris dan 5 kolom.
    *   Tidak ada nilai yang hilang (non-null count = 400 untuk semua kolom).
    *   `Food_ID` bertipe integer, sisanya object (teks).
    *   Distribusi `C_Type`: Jenis masakan 'Indian' dan 'Healthy Food' paling banyak muncul. Terdapat duplikasi minor karena spasi ('Korean' vs ' Korean') yang diperbaiki di tahap Data Preparation.
 
      ![image](https://github.com/user-attachments/assets/df9fb31e-26a9-457d-8dac-f344109452c0)

    *   Distribusi `Veg_Non`: Sekitar 59.5% makanan adalah vegetarian, dan 40.5% non-vegetarian.
 
      ![image](https://github.com/user-attachments/assets/c37bfb27-06cd-415a-a0f6-b82ed92d9f68)

*   **Dataset Rating (`ratings_df`):**
    *   Awalnya 512 baris dan 3 kolom. Terdapat 1 baris dengan nilai NaN pada ketiga kolom, yang kemudian dihapus.
    *   Setelah penghapusan NaN, menjadi 511 baris.
    *   Tipe data awal adalah float64, yang kemudian diubah menjadi int64.
    *   Distribusi Rating Makanan:
      
      ![image](https://github.com/user-attachments/assets/8e415fbb-27d2-4cf6-af2f-275acaf5ede8)

    *   Rating 3.0, 5.0, dan 10.0 adalah yang paling umum diberikan oleh pengguna.
    *   Jumlah User Unik: 100 pengguna.
    *   Jumlah Makanan Unik yang Dirating: 309 makanan (dari total 400 makanan dalam katalog).

## Data Preparation

Tahapan data preparation yang dilakukan adalah sebagai berikut, sesuai urutan dalam notebook:
1.  **Pembersihan Kolom `C_Type` pada `food_df`**:
    *   **Proses**: Menggunakan `str.strip()` untuk menghapus spasi berlebih di awal dan akhir setiap nilai pada kolom `C_Type`.
    *   **Alasan**: Untuk memastikan konsistensi data dan menghindari duplikasi kategori karena perbedaan spasi (misalnya, 'Korean' dan ' Korean' dianggap sebagai kategori yang berbeda sebelum pembersihan). Distribusi C_Type: Jenis masakan 'Indian' dan 'Healthy Food' paling banyak muncul. dan paling sedikit muncul 'Vietnames' dan 'Spanish'.
    
      ![image](https://github.com/user-attachments/assets/85c21e34-1901-47a7-aa2c-9d483e1b2ebd)
  
      
2.  **Penanganan Missing Values pada `ratings_df`**:
    *   **Proses**: Memeriksa nilai NaN menggunakan `isna().sum()`. Ditemukan 1 baris dengan nilai NaN di semua kolom (`User_ID`, `Food_ID`, `Rating`). Baris ini dihapus menggunakan `dropna()`.
    *   **Alasan**: Model machine learning umumnya tidak dapat memproses data dengan nilai yang hilang. Menghapus baris dengan NaN memastikan integritas data untuk pemodelan. Jumlah data rating berkurang dari 512 menjadi 511.
3.  **Konversi Tipe Data pada `ratings_df`**:
    *   **Proses**: Kolom `User_ID`, `Food_ID`, dan `Rating` diubah dari tipe `float64` menjadi `int64` menggunakan `astype('int64')`.
    *   **Alasan**: `User_ID` dan `Food_ID` secara konseptual adalah identifier yang seharusnya integer. `Rating` juga merupakan nilai diskrit. Konversi ini penting agar data dapat diproses dengan benar oleh library seperti Surprise dan untuk efisiensi memori.
4.  **Pembuatan Fitur Gabungan (`content_features`) untuk Content-Based Filtering pada `food_df`**:
    *   **Proses**: Menggabungkan kolom teks `Name`, `C_Type`, `Veg_Non`, dan `Describe` menjadi satu kolom baru bernama `content_features`.
    *   **Alasan**: Untuk membuat representasi konten yang lebih kaya dan komprehensif untuk setiap makanan, yang akan digunakan oleh TfidfVectorizer dalam Content-Based Filtering.
5.  **Penggabungan `food_df` dan `ratings_df` (Opsional, untuk Analisis)**:
    *   **Proses**: Menggabungkan kedua dataframe berdasarkan `Food_ID` menggunakan `pd.merge()`.
    *   **Alasan**: Untuk memudahkan analisis dan visualisasi dimana detail makanan perlu ditampilkan bersama dengan ratingnya, meskipun tidak secara langsung digunakan sebagai input untuk kedua model secara terpisah.

## Modeling

Dua solusi sistem rekomendasi dikembangkan:

### Solusi 1: Content-Based Filtering
Pendekatan ini merekomendasikan makanan berdasarkan kemiripan atribut atau kontennya.
1.  **TF-IDF Vectorization**:
    *   Fitur teks gabungan (`content_features`) dari setiap makanan diubah menjadi vektor numerik menggunakan `TfidfVectorizer`. Parameter `stop_words='english'` digunakan untuk mengabaikan kata-kata umum dalam bahasa Inggris.
    *   Hasilnya adalah matriks TF-IDF dengan bentuk (400 makanan, 1437 kata unik).
    ```python
    # tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    # tfidf_matrix = tfidf_vectorizer.fit_transform(food_df['content_features'])
    # print(f"Bentuk TF-IDF Matrix: {tfidf_matrix.shape}")
    # Output: Bentuk TF-IDF Matrix: (400, 1437)
    ```
2.  **Cosine Similarity**:
    *   Kemiripan antar vektor makanan (dari matriks TF-IDF) dihitung menggunakan `cosine_similarity`.
    *   Hasilnya adalah matriks kemiripan berukuran (400 makanan, 400 makanan).
    ```python
    # cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # print(f"Bentuk Cosine Similarity Matrix: {cosine_sim_matrix.shape}")
    # Output: Bentuk Cosine Similarity Matrix: (400, 400)
    ```
3.  **Top-N Recommendations**:
    *   Fungsi `get_content_based_recommendations` dibuat untuk mengambil Food_ID input, mencari makanan paling mirip berdasarkan skor cosine similarity, dan mengembalikan top-N makanan yang direkomendasikan beserta detailnya.
    *   Contoh output untuk 'shepherds salad (tamatar-kheera salaad)' (Food_ID: 70) merekomendasikan 'summer squash salad' dengan skor kemiripan 0.281735.

    ```
    Top-5 Rekomendasi Content-Based untuk 'shepherds salad (tamatar-kheera salaad)' (Food_ID: 70):
        Food_ID                  Name        C_Type                                           Describe  similarity_score
    0         1  summer squash salad  Healthy Food  white balsamic vinegar, lemon juice, lemon rin...          0.281735
    3         4       tricolour salad  Healthy Food        vinegar, honey/sugar, soy sauce, salt, garlic          0.256170
    70       71    carrot ginger soup  Healthy Food      Carrots, Olive Oil, Salt, Vegetable Stock, Gin...          0.221872
    344     345  Cucumber and Radish Salad Healthy Food              cucumber,raddish,vinegar, coriander,olive, sal...          0.206825
    358     359         Shirazi Salad  Healthy Food              Spring onion, cheese, lemon juice, cucumber o          0.203389
    ```

**Kelebihan Content-Based Filtering:**
*   Tidak memerlukan data dari pengguna lain (mengatasi cold-start untuk pengguna baru).
*   Dapat merekomendasikan item yang spesifik dan niche.
*   Rekomendasi bersifat transparan karena didasarkan pada fitur item yang jelas.

**Kekurangan Content-Based Filtering:**
*   Terbatas pada fitur item yang ada; sulit menemukan rekomendasi yang "serendipitous" atau di luar profil item.
*   Membutuhkan feature engineering yang baik; kualitas fitur sangat mempengaruhi kualitas rekomendasi.
*   Cenderung menghasilkan over-specialization (pengguna hanya direkomendasikan item yang sangat mirip dengan yang sudah disukai).

### Solusi 2: Collaborative Filtering (menggunakan SVD dari library Surprise)
Pendekatan ini merekomendasikan makanan berdasarkan pola rating dari pengguna dengan preferensi serupa.
1.  **Data Preparation untuk Surprise**:
    *   Objek `Reader` dari Surprise didefinisikan dengan `rating_scale` berdasarkan nilai min/max rating di dataset.
    *   Data rating (`User_ID`, `Food_ID`, `Rating`) dimuat menggunakan `Dataset.load_from_df`.
2.  **Train-Test Split**:
    *   Dataset dibagi menjadi 80% data training dan 20% data testing menggunakan `train_test_split` dari Surprise, dengan `random_state=42`.
3.  **SVD Model Training**:
    *   Model SVD (Singular Value Decomposition) diinisialisasi dengan parameter: `n_factors=50`, `n_epochs=20`, `lr_all=0.005`, `reg_all=0.02` (dan `random_state=42` untuk reproduktifitas).
    *   Model dilatih menggunakan `trainset`.
    ```python
    # svd_model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    # svd_model.fit(trainset)
    ```
4.  **Top-N Recommendations**:
    *   Fungsi `get_collaborative_filtering_recommendations` dibuat untuk mengambil `user_id_input`, memprediksi rating untuk makanan yang belum dirating oleh user tersebut, dan mengembalikan top-N makanan dengan prediksi rating tertinggi.
    *   Contoh output untuk User_ID 6:

    ```
    Top-5 Rekomendasi Collaborative Filtering (SVD) untuk User_ID 6:
       Food_ID  Predicted_Rating                                Name        C_Type
    0       44          6.203652             andhra pan fried pomfret        Indian
    1      127          6.029333  cajun spiced turkey wrapped with bacon      Mexican
    2      136          5.917355                       malabari fish curry        Indian
    3       40          5.864231                  corn and raw mango salad Healthy Food
    4      273          5.843811                   corn & jalapeno poppers     Mexican
    ```

**Kelebihan Collaborative Filtering:**
*   Tidak memerlukan pengetahuan domain atau fitur item.
*   Dapat menemukan rekomendasi yang "serendipitous" dan beragam.
*   Model belajar dari interaksi pengguna secara implisit.

**Kekurangan Collaborative Filtering:**
*   Masalah cold-start: sulit memberikan rekomendasi untuk pengguna baru atau item baru yang belum memiliki interaksi.
*   Data sparsity: performa menurun jika matriks interaksi pengguna-item sangat jarang (banyak rating yang hilang).
*   Popularity bias: item populer cenderung lebih sering direkomendasikan.

## Evaluation

### Evaluasi Content-Based Filtering
Evaluasi untuk Content-Based Filtering dalam proyek ini lebih bersifat **kualitatif**. Metrik kuantitatif seringkali sulit dan kurang intuitif untuk sistem berbasis konten tanpa adanya ground truth yang jelas mengenai "kemiripan ideal".
*   **Kualitas Rekomendasi (Qualitative)**: Penilaian dilakukan dengan melihat apakah rekomendasi yang diberikan masuk akal dan relevan dengan item input.
    *   **Contoh**: Untuk input 'shepherds salad (tamatar-kheera salaad)' (sebuah hidangan salad sehat), sistem merekomendasikan 'summer squash salad', 'tricolour salad', 'carrot ginger soup', 'Cucumber and Radish Salad', dan 'Shirazi Salad'. Semua rekomendasi ini juga merupakan hidangan sehat dan beberapa diantaranya adalah salad atau sup sayuran. Ini menunjukkan bahwa model mampu menangkap kemiripan konteks dan deskripsi bahan. Skor kemiripan (Cosine Similarity) juga disertakan, misalnya 'summer squash salad' memiliki skor 0.281735.

### Evaluasi Collaborative Filtering (SVD)
Untuk Collaborative Filtering, metrik evaluasi kuantitatif digunakan untuk mengukur seberapa akurat model memprediksi rating yang akan diberikan pengguna.
*   **Root Mean Squared Error (RMSE)**:
    *   **Formula**: `RMSE = sqrt(mean((actual_rating - predicted_rating)^2))`
    *   **Cara Kerja**: RMSE mengukur standar deviasi dari error prediksi. Semakin kecil nilai RMSE, semakin baik performa model dalam memprediksi rating. RMSE memberikan bobot lebih besar pada error yang besar karena adanya kuadrat.
*   **Mean Absolute Error (MAE)**:
    *   **Formula**: `MAE = mean(|actual_rating - predicted_rating|)`
    *   **Cara Kerja**: MAE mengukur rata-rata selisih absolut antara rating aktual dan prediksi. Seperti RMSE, nilai MAE yang lebih kecil menunjukkan performa model yang lebih baik. MAE kurang sensitif terhadap outlier dibandingkan RMSE.

**Hasil Metrik Evaluasi untuk Model SVD**:
Prediksi dibuat pada `testset` (20% data).
```python
# predictions_svd = svd_model.test(testset)
# rmse_svd = accuracy.rmse(predictions_svd)
# mae_svd = accuracy.mae(predictions_svd)
# Output:
# RMSE: 2.9298
# MAE: 2.5390
```
## Kesimpulan

Bagian ini merangkum bagaimana masing-masing *problem statement* yang dirumuskan di bagian **Business Understanding** telah terjawab melalui pengembangan dan evaluasi sistem rekomendasi makanan.

1. **Bagaimana cara merekomendasikan makanan kepada pengguna berdasarkan kemiripan atribut makanan (seperti jenis masakan, bahan, deskripsi, status vegetarian)?**  
   Tujuan ini telah tercapai melalui pendekatan **Content-Based Filtering**. Dengan menggunakan `TfidfVectorizer`, fitur teks gabungan (`content_features`) dari kolom `Name`, `C_Type`, `Veg_Non`, dan `Describe` diubah menjadi representasi numerik dalam bentuk matriks TF-IDF. Kemudian, kemiripan antar makanan dihitung menggunakan *Cosine Similarity*, menghasilkan matriks kemiripan berukuran (400, 400). Fungsi `get_content_based_recommendations` berhasil menghasilkan rekomendasi top-N yang relevan, seperti yang ditunjukkan pada contoh rekomendasi untuk 'shepherds salad (tamatar-kheera salaad)' (Food_ID: 70), yang merekomendasikan hidangan serupa seperti 'summer squash salad' dan 'tricolour salad' dengan skor kemiripan tertinggi (0.281735 untuk 'summer squash salad'). Evaluasi kualitatif menunjukkan bahwa rekomendasi ini relevan karena mencerminkan kesamaan dalam jenis masakan (seperti 'Healthy Food') dan bahan (seperti sayuran), sehingga menjawab *problem statement* ini dengan baik.

2. **Bagaimana cara merekomendasikan makanan kepada pengguna berdasarkan preferensi dan pola rating dari pengguna lain yang memiliki selera serupa?**  
   Tujuan ini telah tercapai melalui pendekatan **Collaborative Filtering** menggunakan algoritma Singular Value Decomposition (SVD) dari library Surprise. Model SVD dilatih pada data rating (`ratings_df`) dengan parameter seperti `n_factors=50`, `n_epochs=20`, `lr_all=0.005`, dan `reg_all=0.02`, setelah membagi dataset menjadi 80% data latih dan 20% data uji. Fungsi `get_collaborative_filtering_recommendations` berhasil menghasilkan rekomendasi top-N berdasarkan prediksi rating untuk makanan yang belum dirating oleh pengguna. Contohnya, untuk User_ID 6, sistem merekomendasikan makanan seperti 'andhra pan fried pomfret' dengan prediksi rating 6.203652. Pendekatan ini memanfaatkan pola rating dari pengguna lain dengan selera serupa, sehingga secara efektif menjawab *problem statement* ini dengan menghasilkan rekomendasi yang relevan berdasarkan preferensi pengguna.

3. **Bagaimana cara mengevaluasi performa dari kedua pendekatan sistem rekomendasi yang dibangun?**  
   Evaluasi performa kedua pendekatan telah dilakukan dengan metrik yang sesuai. Untuk **Content-Based Filtering**, evaluasi bersifat kualitatif dengan menilai relevansi rekomendasi berdasarkan kesamaan konteks dan fitur makanan. Contohnya, rekomendasi untuk 'shepherds salad' menunjukkan bahwa sistem mampu mengidentifikasi makanan dengan karakteristik serupa (seperti salad atau hidangan sehat lainnya), yang menunjukkan keberhasilan model dalam menangkap kemiripan atribut. Untuk **Collaborative Filtering**, evaluasi kuantitatif dilakukan menggunakan metrik **RMSE** (2.9298) dan **MAE** (2.5390) pada *testset*. Nilai RMSE dan MAE ini mengindikasikan bahwa model SVD memiliki tingkat error yang moderat dalam memprediksi rating, yang wajar mengingat skala rating 1-10 dan distribusi rating yang bervariasi (d dengan puncak pada 3.0, 5.0, dan 10.0). Meskipun nilai RMSE dan MAE menunjukkan adanya ruang untuk perbaikan, hasil ini cukup baik untuk dataset dengan jumlah rating terbatas (511 entri) dan menunjukkan bahwa model dapat memprediksi preferensi pengguna dengan akurasi yang memadai. Dengan demikian, kedua pendekatan telah dievaluasi secara menyeluruh, menjawab *problem statement* ini dengan baik.

**Kesimpulan Umum**:  
Proyek ini berhasil mengembangkan dua sistem rekomendasi makanan yang menjawab kebutuhan yang diidentifikasi dalam *problem statement*. Pendekatan **Content-Based Filtering** efektif untuk merekomendasikan makanan berdasarkan atribut, cocok untuk pengguna baru atau ketika data rating terbatas. Pendekatan **Collaborative Filtering** dengan SVD efektif untuk merekomendasikan makanan berdasarkan preferensi pengguna lain, meskipan performanya dipengaruhi oleh keterbatasan data (sparsity). Evaluasi menunjukkan bahwa kedua pendekatan memiliki kelebihan dan kekurangan masing-masing, dengan Content-Based Filtering unggul dalam transparansi dan Collaborative Filtering unggul dalam menemukan pola preferensi yang lebih beragam. Untuk pengembangan lebih lanjut, hybrid approach yang menggabungkan kedua metode dapat dipertimbangkan untuk meningkatkan akurasi dan relevansi rekomendasi.
