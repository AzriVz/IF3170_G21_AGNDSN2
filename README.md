# IF3170_G21_AGNDSN2

Tugas Besar 2 IF3170 Inteligensi Artifisial

<p align="center">
  <img src="pisa.gif" alt="Victoire Pisa">
</p>

Pada tugas besar ini, diselesaikan persoalan supervised learning berupa klasifikasi biner. Berdasarkan himpunan fitur historis dan detail reservasi tamu, sistem pembelajaran mesin dilatih untuk memetakan setiap data pemesanan ke dalam dua kelas diskret: pemesanan yang dibatalkan (1) dan pemesanan yang tidak dibatalkan (0).

## Pembagian Tugas

| Nama | NIM | Tugas |
| --- | --- | --- |
| I Gusti Ngurah Alit Dharma Y | 13524072 | Logistic Regression, Preprocessing |
| Billy Ontoseno Irawan | 13524121 | Bayesian Network / Naive Bayes, Preprocessing |
| Eduard Daniel Ariajaya | 13524129 | KNN, Preprocessing, Error Analysis |
| Azri Arzaq Pohan | 13524139 | Decision Tree (CART), Preprocessing, Visualization, Error Analysis |
| Rasyad Satyatma | 13524142 | SVM, Preprocessing |


## Struktur Repository

```
data/       train.csv, test.csv, sample_submission.csv
src/        source code dan notebook
  models/   implementasi setiap model
scripts/    skrip tuning hyperparameter (KNN, DTL, SVM)
output/     hasil run: experiments/ (CSV/JSON/log), figures/, models/ (.pkl), submissions/
docs/       laporan (.pdf)
```

## Setup

Membutuhkan versi Python 3.11 atau lebih baru. ffmpeg juga bersifat opsional dan hanya dipakai untuk menyimpan video Logistic

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Cara Menjalankan

Jalankan seluruh notebook dari awal sampai akhir, bisa langsung **Run All** jika menggunakan VS Code.

```bash
jupyter lab src/IF3170_G21_AGNDSN2.ipynb
```

Lalu pilih **Run → Run All Cells**. 

Atau bisa juga dijalankan pada cli seperti ini:

```bash
jupyter nbconvert --to notebook --execute --inplace src/IF3170_G21_AGNDSN2.ipynb
```

Contoh running script tuning:

```bash
python scripts/knn_search.py --metric manhattan --rounds 2 --grid 0,0.5,1,2,3 --out output/experiments/knn_ascent_manhattan.json
python scripts/tree_search.py --out output/experiments/tree_search_gini.json
python scripts/svm_search.py --out output/experiments/svm_search.json
```