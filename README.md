# Network Intrusion Detection using Machine Learning (UNSW-NB15)

## 🇹🇷 Türkçe

### Proje Hakkında

Bu proje, ağ trafiği verilerini analiz ederek siber saldırıları tespit etmek amacıyla geliştirilmiştir. Çalışmada UNSW-NB15 veri seti kullanılmış ve çeşitli makine öğrenmesi algoritmaları ile ağ trafiği sınıflandırması gerçekleştirilmiştir.

Projenin amacı normal ve zararlı ağ aktivitelerini ayırt ederek olası saldırıları erken aşamada tespit etmektir.

### Kullanılan Teknolojiler

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Jupyter Notebook

### Veri Seti

* Dataset: UNSW-NB15
* Ağ trafiği özellikleri
* Bağlantı metrikleri
* Trafik hacimleri
* Zamanlama metrikleri
* Davranışsal özellikler

### Proje Akışı

1. Veri ön işleme
2. Eksik verilerin temizlenmesi
3. Özellik mühendisliği
4. Veri görselleştirme
5. Model eğitimi
6. Model değerlendirme
7. Sonuç analizi

---

## Sonuçlar

### Confusion Matrix


<img width="903" height="423" alt="image" src="https://github.com/user-attachments/assets/eeefce6b-dc56-422a-8bfd-0e120dfe2de4" />


---

### Confusion Matrix


<img width="941" height="365" alt="image" src="https://github.com/user-attachments/assets/8e8a7285-b72a-49ac-a453-ad3d4318247d" />

---

### Model Performance

| Metric    | Score  |
|-----------|---------|
| Accuracy  | 86.52% |
| Precision | 82.12% |
| Recall    | 96.52% |
| F1-Score  | 88.74% |
| AUC-ROC   | 95.95% |

---

## 🇬🇧 English

### About The Project

This project focuses on detecting cyber attacks by analyzing network traffic data. The UNSW-NB15 dataset was used to train and evaluate machine learning models for intrusion detection.

The main objective is to distinguish normal network activities from malicious behaviors and identify potential cyber threats.

### Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Jupyter Notebook

### Dataset

* UNSW-NB15 Dataset
* Network traffic features
* Connection metrics
* Traffic volume metrics
* Timing metrics
* Behavioral attributes

### Workflow

1. Data preprocessing
2. Missing value handling
3. Feature engineering
4. Data visualization
5. Model training
6. Model evaluation
7. Performance analysis

---
## Project Structure

```text
.
├── data/
├── notebooks/
├── models/
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── src/
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/hypervsec/Network-Intrusion-Detection-using-Machine-Learning.git
cd Network-Intrusion-Detection-using-Machine-Learning
pip install -r 
```

Developed for network intrusion detection and cyber security research purposes.
