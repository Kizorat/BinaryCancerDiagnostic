import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_PATH = "dataset/clean_data.csv"
FIG_DIR = "dataset/plot_analisys"

#elimino le immagini se riavvio il codice per riempire di nuovo la cartella
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".svg", ".pdf")

if os.path.isdir(FIG_DIR):
    for fname in os.listdir(FIG_DIR):
        if fname.lower().endswith(IMAGE_EXTS):
            os.remove(os.path.join(FIG_DIR, fname))
else:
    os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=200)
    plt.close()

df = pd.read_csv(DATA_PATH)

df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")
df = df.dropna(axis=1, how="all")

print(df.head(5))
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1}) 
print(df.head(5))

df_no_id = df.drop(columns=["id"], errors="ignore") #ignoro la colonna id

counts = df["diagnosis"].value_counts()

plt.figure()
plt.bar(["Benigno", "Maligno"], counts.values)
plt.title("Distribuzione delle diagnosi") #maligno e benigno
plt.ylabel("Numero di campioni")
savefig("diagnostic_distribution.png")

#ISTOGRAMMA, BOXPLOT E SCATTERPLOT 
#classi: perimetro, area e concavità

def hist_by_class(feature, fname):
    plt.figure()
    plt.hist(df[df["diagnosis"] == 0][feature], bins=30, alpha=0.6, label="Benigno")
    plt.hist(df[df["diagnosis"] == 1][feature], bins=30, alpha=0.6, label="Maligno")
    plt.legend()
    plt.title(f"Confronto {feature}: tra benigno e maligno")
    plt.xlabel(feature)
    plt.ylabel("Frequenza")
    savefig(fname)

hist_by_class("radius_mean", "hist_radius_mean_overlap.png")
hist_by_class("perimeter_mean", "hist_perimeter_mean_overlap.png")
hist_by_class("area_mean", "hist_area_mean_overlap.png")

def box_by_class(feature, fname):
    plt.figure()
    plt.boxplot(
        [df[df["diagnosis"] == 0][feature], df[df["diagnosis"] == 1][feature]],
        tick_labels=["Benigno", "Maligno"]
    )
    plt.title(f"{feature} per classe")
    plt.ylabel(feature)
    savefig(fname)

box_by_class("radius_mean", "box_radius_mean.png")
box_by_class("perimeter_mean", "box_perimeter_mean.png")
box_by_class("area_mean", "box_area_mean.png")
box_by_class("concavity_mean", "box_concavity_mean.png")
box_by_class("concave points_mean", "box_concave_points_mean.png")

plt.figure()
plt.scatter(
    df[df["diagnosis"] == 0]["radius_mean"],
    df[df["diagnosis"] == 0]["concavity_mean"],
    alpha=0.6,
    label="Benigno"
)
plt.scatter(
    df[df["diagnosis"] == 1]["radius_mean"],
    df[df["diagnosis"] == 1]["concavity_mean"],
    alpha=0.6,
    label="Maligno"
)
plt.xlabel("radius_mean")
plt.ylabel("concavity_mean")
plt.legend()
plt.title("Confronto radius_mean e concavity_mean")
savefig("scatter_radius_concavity.png")

mean_benign = df_no_id[df_no_id["diagnosis"] == 0].mean(numeric_only=True)
mean_malign = df_no_id[df_no_id["diagnosis"] == 1].mean(numeric_only=True)

mean_df = pd.DataFrame({"Benigno": mean_benign, "Maligno": mean_malign}).drop(index="diagnosis", errors="ignore")

fig, ax = plt.subplots(figsize=(18, 6))
mean_df.plot(kind="bar", ax=ax)
ax.set_title("Confronto delle medie – tutte le feature")
ax.set_ylabel("Valore medio")
ax.tick_params(axis="x", labelrotation=90, labelsize=7)
savefig("mean_comparison_all.png")

main_features = [
    "radius_mean",
    "perimeter_mean",
    "area_mean",
    "concavity_mean",
    "concave points_mean"
]

fig, ax = plt.subplots(figsize=(10, 5))
mean_df.loc[main_features].plot(kind="bar", ax=ax)
ax.set_title("Confronto delle medie – feature principali")
ax.set_ylabel("Valore medio")
ax.tick_params(axis="x", labelrotation=45, labelsize=10)
savefig("mean_comparison_main.png")

#per spiegare meglio i risultati dell'istogramma
#viene riportata la media singola di M/B per evidenziare valori più piccoli
for f in main_features:
    benign_vals = df[df["diagnosis"] == 0][f].dropna()
    malign_vals = df[df["diagnosis"] == 1][f].dropna()

    mean_b = benign_vals.mean()
    mean_m = malign_vals.mean()

    xmin = df[f].min()
    xmax = df[f].max()

    plt.figure()
    plt.hist(benign_vals, bins=30, alpha=0.8, color="tab:blue", label="Benigno")
    plt.axvline(mean_b, color="tab:gray", linestyle="--", linewidth=2, label=f"Media Benigno = {mean_b:.3f}")
    plt.title(f"Istogramma {f} – Benigno")
    plt.xlabel(f)
    plt.ylabel("Frequenza")
    plt.xlim(xmin, xmax)
    plt.legend()
    savefig(f"hist_{f}_benigno.png")

    plt.figure()
    plt.hist(malign_vals, bins=30, alpha=0.8, color="tab:orange", label="Maligno")
    plt.axvline(mean_m, color="tab:gray", linestyle="--", linewidth=2, label=f"Media Maligno = {mean_m:.3f}")
    plt.title(f"Istogramma {f} – Maligno")
    plt.xlabel(f)
    plt.ylabel("Frequenza")
    plt.xlim(xmin, xmax)
    plt.legend()
    savefig(f"hist_{f}_maligno.png")

X = df_no_id.drop(columns=["diagnosis"], errors="ignore")
y = df_no_id["diagnosis"].values

mask = ~np.isnan(X.to_numpy()).any(axis=1)
X = X.loc[mask]
y = y[mask]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#ANALISI DELLA PCA/PCA2D
pca = PCA()
pca.fit(X_scaled)
explained_var = np.cumsum(pca.explained_variance_ratio_)

plt.figure()
plt.plot(explained_var, marker="o")
plt.axhline(y=0.90, linestyle="--")
plt.axhline(y=0.95, linestyle="--")
plt.xlabel("Numero di componenti principali")
plt.ylabel("Varianza cumulativa")
plt.title("PCA")
savefig("pca_explained_variance.png")

pca_2 = PCA(n_components=2)
X_pca = pca_2.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], alpha=0.6, label="Benigno")
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], alpha=0.6, label="Maligno")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.title("PCA 2D")
savefig("pca_2d.png")

print(f"\nPlot salvati in: {FIG_DIR}/")
