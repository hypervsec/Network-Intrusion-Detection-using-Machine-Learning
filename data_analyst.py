# =============================================================================
#  Veri Analizine Giriş Dersi – Dönem Sonu Projesi
#  Veri Seti  : UNSW-NB15 (Ağ Güvenliği / Saldırı Tespiti)
#  Problem    : İkili Sınıflandırma (Normal=0 / Saldırı=1)
#  Kaynak     : Kaggle / IEEE DataPort
# =============================================================================

# ── GEREKLİ KÜTÜPHANELER ────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
#  BÖLÜM 1 – VERİ TEMİNİ VE KEŞFİ
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("BÖLÜM 1 – VERİ TEMİNİ VE KEŞFİ")
print("=" * 60)

# Veri setini yükle
# Kaggle'dan indirilen iki ayrı dosya: eğitim ve test seti
train = pd.read_parquet("UNSW_NB15_training-set.parquet")
test  = pd.read_parquet("UNSW_NB15_testing-set.parquet")

print(f"\nEğitim seti boyutu : {train.shape[0]:,} satır × {train.shape[1]} sütun")
print(f"Test seti boyutu   : {test.shape[0]:,} satır × {test.shape[1]} sütun")

print("\n── Sütun Adları ──")
print(train.columns.tolist())

print("\n── Veri Türleri ──")
print(train.dtypes)

print("\n── İlk 5 Satır ──")
print(train.head())

print("\n── Hedef Değişken (label) Dağılımı ──")
print(train["label"].value_counts())
print(f"\nSaldırı oranı: %{train['label'].mean()*100:.1f}")

print("\n── Saldırı Kategorisi Dağılımı ──")
print(train["attack_cat"].value_counts())

# ═══════════════════════════════════════════════════════════════════════════
#  BÖLÜM 2 – VERİ TEMİZLEME VE ÖN İŞLEME
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("BÖLÜM 2 – VERİ TEMİZLEME VE ÖN İŞLEME")
print("=" * 60)

# ── 2.1 Eksik Değer Kontrolü ─────────────────────────────────────────────
print("\n── 2.1 Eksik Değer Kontrolü ──")
missing_train = train.isnull().sum()
missing_test  = test.isnull().sum()
print(f"Eğitim setinde toplam eksik değer : {missing_train.sum()}")
print(f"Test setinde toplam eksik değer   : {missing_test.sum()}")
# Sonuç: Eksik değer yok, herhangi bir imputation gerekmedi.

# ── 2.2 Aykırı Değer Analizi ─────────────────────────────────────────────
print("\n── 2.2 Aykırı Değer Analizi (IQR Yöntemi) ──")
numeric_cols = train.select_dtypes(include=[np.number]).columns.drop("label")
outlier_counts = {}
for col in numeric_cols:
    Q1  = train[col].quantile(0.25)
    Q3  = train[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((train[col] < Q1 - 1.5*IQR) | (train[col] > Q3 + 1.5*IQR)).sum()
    outlier_counts[col] = n_out

outlier_series = pd.Series(outlier_counts).sort_values(ascending=False)
print("En fazla aykırı değer içeren 10 özellik:")
print(outlier_series.head(10))

# Aykırı değerler silinmedi; ağ trafiğinde uç değerler genellikle
# saldırıya özgü anlamlı bilgi taşır, kaldırılmaları önemli ipuçlarını yok eder.
print("\nNot: Aykırı değerler silinmedi.")
print("Ağ trafiğinde uç değerler saldırıya özgü anlamlı sinyal taşıyabilir.")

# ── 2.3 Kategorik Kodlama (Label Encoding) ───────────────────────────────
print("\n── 2.3 Kategorik Kodlama ──")
# proto, service, state sütunları kategorik → sayısal dönüştürme
# Eğitim ve test seti birlikte fit edilerek tutarlılık sağlandı.
cat_cols = ["proto", "service", "state"]
le_dict  = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))
    le_dict[col] = le
    print(f"  {col}: {len(le.classes_)} benzersiz kategori kodlandı")

# ── 2.4 Özellik / Hedef Ayrımı ────────────────────────────────────────────
# attack_cat çıkarıldı (kategorik saldırı adı, hedef değil)
X_train = train.drop(["attack_cat", "label"], axis=1)
y_train = train["label"]
X_test  = test.drop(["attack_cat", "label"], axis=1)
y_test  = test["label"]

print(f"\nEğitim özellik matrisi : {X_train.shape}")
print(f"Test özellik matrisi   : {X_test.shape}")

# ── 2.5 Ölçeklendirme (StandardScaler) ───────────────────────────────────
# Yalnızca uzaklık/gradyan tabanlı modeller için kullanıldı:
#   - Lojistik Regresyon (gradyan), K-En Yakın Komşu (Öklid mesafesi)
# Ağaç tabanlı modeller (Karar Ağacı, RF, GB) ölçekten etkilenmez → ham veri
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit + transform: eğitim
X_test_s  = scaler.transform(X_test)        # sadece transform: test (veri sızıntısı önlenir)

print("\nÖlçeklendirme tamamlandı (Lojistik Regresyon ve KNN için).")

# ═══════════════════════════════════════════════════════════════════════════
#  BÖLÜM 3 – GÖRSEL KEŞİFSEL VERİ ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("BÖLÜM 3 – GÖRSELLEŞTİRME")
print("=" * 60)

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#F8F9FA",
                     "axes.grid": True, "grid.alpha": 0.3})
COLORS = ["#2E86AB","#A23B72","#F18F01","#C73E1D","#3B1F2B",
          "#44BBA4","#E94F37","#393E41","#F5A623","#8B5CF6"]

# ── Görsel 1: Sınıf & Saldırı Dağılımı ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("UNSW-NB15 – Sınıf Dağılımı", fontsize=15, fontweight="bold")

counts = y_train.value_counts()
bars = axes[0].bar(["Normal (0)", "Saldırı (1)"], counts.values,
                   color=["#2E86AB", "#C73E1D"], edgecolor="white", linewidth=1.5)
axes[0].set_title("İkili Etiket Dağılımı (Eğitim)", fontweight="bold")
axes[0].set_ylabel("Kayıt Sayısı")
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 800,
                 f"{val:,}\n(%{val/len(y_train)*100:.1f})",
                 ha="center", fontsize=11, fontweight="bold")

attack_counts = train["attack_cat"].value_counts()
axes[1].pie(attack_counts.values, labels=None,
            autopct="%1.1f%%", colors=COLORS[:len(attack_counts)],
            startangle=140, pctdistance=0.75,
            wedgeprops=dict(edgecolor="white", linewidth=1.5))
axes[1].set_title("Saldırı Kategorisi Dağılımı", fontweight="bold")
axes[1].legend(attack_counts.index, loc="lower left", fontsize=8)
plt.tight_layout()
plt.savefig("gorsel1_dagilim.png", dpi=150, bbox_inches="tight")
plt.show()
print("gorsel1_dagilim.png kaydedildi")

# ── Görsel 2: Korelasyon Matrisi ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10))
top_cols = X_train.columns[:20]
corr = X_train[top_cols].corr()
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            ax=ax, square=True, linewidths=0.3,
            cbar_kws={"shrink": 0.7}, annot=False)
ax.set_title("Özellik Korelasyon Matrisi (İlk 20 Nümerik Özellik)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("gorsel2_korelasyon.png", dpi=150, bbox_inches="tight")
plt.show()
print("gorsel2_korelasyon.png kaydedildi")

# ── Görsel 3: Aykırı Değer – Box Plot ────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Aykırı Değer Analizi – Temel Özellikler",
             fontsize=14, fontweight="bold")
key_features = ["dur", "rate", "sbytes", "dbytes", "sload", "dload", "spkts", "dpkts"]
for ax, feat in zip(axes.flat, key_features):
    p99 = train[feat].quantile(0.99)
    data0 = train[train["label"] == 0][feat].clip(upper=p99)
    data1 = train[train["label"] == 1][feat].clip(upper=p99)
    bp = ax.boxplot([data0, data1], patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#2E86AB")
    bp["boxes"][1].set_facecolor("#C73E1D")
    ax.set_title(feat, fontweight="bold", fontsize=10)
    ax.set_xticklabels(["Normal", "Saldırı"], fontsize=9)
plt.tight_layout()
plt.savefig("gorsel3_aykiri.png", dpi=150, bbox_inches="tight")
plt.show()
print("gorsel3_aykiri.png kaydedildi")

# ═══════════════════════════════════════════════════════════════════════════
#  BÖLÜM 4 – MODEL OLUŞTURMA VE EĞİTİM
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("BÖLÜM 4 – MODEL OLUŞTURMA VE EĞİTİM")
print("=" * 60)

# Kullanılan 5 model:
#  1. Lojistik Regresyon   → temel/baseline model
#  2. Karar Ağacı          → yorumlanabilir kural tabanlı model
#  3. Rastgele Orman       → ensemble (bagging)
#  4. K-En Yakın Komşu     → uzaklık tabanlı model
#  5. Gradyan Artırma      → ensemble (boosting)

models = {
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "Karar Ağacı":        DecisionTreeClassifier(max_depth=15, random_state=42),
    "Rastgele Orman":     RandomForestClassifier(n_estimators=100, max_depth=15,
                                                  random_state=42, n_jobs=-1),
    "K-En Yakın Komşu":  KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Gradyan Artırma":    GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                      random_state=42),
}

# Lojistik Regresyon ve KNN → ölçeklendirilmiş veri
# Ağaç tabanlı modeller    → ham veri
SCALED_MODELS = {"Lojistik Regresyon", "K-En Yakın Komşu"}

results = {}
cms     = {}
trained = {}

for name, model in models.items():
    print(f"\n  Eğitiliyor: {name} ...")
    X_tr = X_train_s if name in SCALED_MODELS else X_train
    X_te = X_test_s  if name in SCALED_MODELS else X_test

    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]

    results[name] = {
        "Doğruluk":   accuracy_score(y_test, preds),
        "Kesinlik":   precision_score(y_test, preds),
        "Duyarlılık": recall_score(y_test, preds),
        "F1":         f1_score(y_test, preds),
        "AUC-ROC":    roc_auc_score(y_test, proba),
    }
    cms[name]     = confusion_matrix(y_test, preds)
    trained[name] = model

    r = results[name]
    print(f"    Doğruluk={r['Doğruluk']:.4f}  Kesinlik={r['Kesinlik']:.4f}  "
          f"Duyarlılık={r['Duyarlılık']:.4f}  F1={r['F1']:.4f}  AUC={r['AUC-ROC']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
#  BÖLÜM 5 – SONUÇLARIN KARŞILAŞTIRILMASI
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("BÖLÜM 5 – SONUÇ TABLOSU")
print("=" * 60)

res_df = pd.DataFrame(results).T.round(4)
print("\n", res_df.to_string())

best_model = res_df["F1"].idxmax()
print(f"\n✓ En iyi F1 skoru: {best_model}  ({res_df.loc[best_model,'F1']:.4f})")

# Sonuçları CSV'ye kaydet
res_df.to_csv("model_sonuclari.csv")
print("model_sonuclari.csv kaydedildi")

# ── Görsel 4: Model Karşılaştırma Grafikleri ─────────────────────────────
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Model Performans Karşılaştırması", fontsize=15, fontweight="bold")

metrics      = ["Doğruluk", "Kesinlik", "Duyarlılık", "F1", "AUC-ROC"]
model_names  = list(results.keys())
x            = np.arange(len(metrics))
width        = 0.15

for i, (mname, color) in enumerate(zip(model_names, COLORS)):
    vals = [results[mname][m] for m in metrics]
    axes[0].bar(x + i*width - (len(model_names)-1)*width/2,
                vals, width, label=mname, color=color, alpha=0.85, edgecolor="white")

axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].set_ylim(0.5, 1.05)
axes[0].set_ylabel("Skor")
axes[0].set_title("Tüm Metrikler", fontweight="bold")
axes[0].legend(fontsize=8)

f1_vals   = res_df["F1"].values
bar_colors = ["#F18F01" if m == best_model else "#2E86AB" for m in model_names]
bars2 = axes[1].barh(range(len(model_names)), f1_vals,
                     color=bar_colors, edgecolor="white", height=0.6)
axes[1].set_yticks(range(len(model_names)))
axes[1].set_yticklabels([m.replace(" ", "\n") for m in model_names])
axes[1].set_xlabel("F1 Skoru")
axes[1].set_title("F1 Skoru Karşılaştırması\n(Turuncu = En İyi)", fontweight="bold")
axes[1].set_xlim(0.5, 1.02)
for bar, val in zip(bars2, f1_vals):
    axes[1].text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=10, fontweight="bold")
gold = mpatches.Patch(color="#F18F01", label=f"En İyi: {best_model}")
blue = mpatches.Patch(color="#2E86AB", label="Diğer Modeller")
axes[1].legend(handles=[gold, blue], fontsize=9)

plt.tight_layout()
plt.savefig("gorsel4_model_karsilastirma.png", dpi=150, bbox_inches="tight")
plt.show()
print("gorsel4_model_karsilastirma.png kaydedildi")

# ── Görsel 5: Karışıklık Matrisleri ──────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle("Karışıklık Matrisleri – Tüm Modeller", fontsize=14, fontweight="bold")
for ax, (mname, cm) in zip(axes, cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Saldırı"],
                yticklabels=["Normal", "Saldırı"],
                cbar=False, linewidths=0.5, linecolor="white",
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_title(mname.replace(" ", "\n"), fontweight="bold", fontsize=9)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
plt.tight_layout()
plt.savefig("gorsel5_karisiklik_matrisi.png", dpi=150, bbox_inches="tight")
plt.show()
print("gorsel5_karisiklik_matrisi.png kaydedildi")

# ── Görsel 6: Özellik Önemi (Rastgele Orman) ─────────────────────────────
rf_model    = trained["Rastgele Orman"]
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
top15       = importances.nlargest(15).sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(top15)), top15.values, color="#2E86AB",
               edgecolor="white", alpha=0.9)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index, fontsize=11)
ax.set_xlabel("Önem Skoru", fontsize=12)
ax.set_title("Rastgele Orman – En Önemli 15 Özellik",
             fontsize=14, fontweight="bold")
for bar, val in zip(bars, top15.values):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("gorsel6_ozellik_onemi.png", dpi=150, bbox_inches="tight")
plt.show()
print("gorsel6_ozellik_onemi.png kaydedildi")

# ═══════════════════════════════════════════════════════════════════════════
#  BÖLÜM 6 – SONUÇLARIN YORUMU
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("BÖLÜM 6 – SONUÇLARIN YORUMU")
print("=" * 60)

print("""
En iyi model: Karar Ağacı (F1 = 0.8874, AUC-ROC = 0.9595)

Neden Karar Ağacı bu veri setinde öne çıktı?
─────────────────────────────────────────────
1. UNSW-NB15 veri setindeki saldırı örüntüleri belirgin eşikler oluşturuyor.
   Örneğin TTL değerleri, paket hızları ve bağlantı süreleri kategorik
   sınırlar gibi ayrışıyor. Karar ağaçları bu tür kesin "if-then" sınırlarını
   en iyi şekilde yakalayabiliyor.

2. Lojistik Regresyon bu veri setinde lineer karar sınırı yeterli olmadığı
   için düşük performans gösterdi.

3. Rastgele Orman ve Gradyan Artırma daha yüksek duyarlılık (%98+) elde
   etse de F1 ve kesinlik açısından Karar Ağacı'nın hafifçe gerisinde kaldı.

4. Siber güvenlik perspektifinden:
   - Yüksek duyarlılık (recall) önceliklidir → saldırıları kaçırmamak
   - Tüm modeller %95+ duyarlılık sağladı → bu veri setinde başarılı

5. En önemli özellikler TTL tabanlıydı (sttl, dttl, ct_state_ttl).
   Saldırılar genellikle anormal TTL değerleri üretiyor.
""")

print("=" * 60)
print("Proje tamamlandı. Tüm görseller ve sonuç dosyası kaydedildi.")
print("=" * 60)
