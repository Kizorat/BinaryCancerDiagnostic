import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc
)

from sklearn.inspection import permutation_importance

# OUTPUT risultati
plot_dir = "dataset/plot_SVM"
os.makedirs(plot_dir, exist_ok=True)

df = pd.read_csv("dataset/clean_data.csv")
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop("diagnosis", axis=1)
if "id" in X.columns:
    X = X.drop("id", axis=1)

y = df["diagnosis"]

# Train e split test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# MODELLO SVM
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    ))
])

# tuning di C e gamma
param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": [0.001, 0.01, 0.1, 1]
}

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("Migliori parametri:", grid.best_params_)

# PREDIZIONE E METRICHE
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Benigno (B)", "Maligno (M)"])

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# FEATURE IMPORTANCE (Permutation Importance su test)
perm = permutation_importance(
    best_model,
    X_test, y_test,
    n_repeats=30,
    random_state=42,
    scoring="f1"
)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance_Mean": perm.importances_mean,
    "Importance_Std": perm.importances_std,
    "Abs_Importance": np.abs(perm.importances_mean)
}).sort_values("Abs_Importance", ascending=False)

print("\nTop 10 Feature Importance (Permutation Importance):")
print(feature_importance.head(10).to_string(index=False))

# 
feature_csv_path = f"{plot_dir}/feature_importance.csv"
feature_importance.to_csv(feature_csv_path, index=False)
print(f"\nFeature importance salvata in: {feature_csv_path}")

# DIREZIONE (verde/rosso) per le feature più importanti (15)
# stimata come correlazione feature vs prob maligno
direction = {}
for col in X.columns:
    corr = np.corrcoef(X_test[col].values, y_pred_proba)[0, 1]
    if np.isnan(corr):
        corr = 0.0
    direction[col] = corr

# ROC curve values
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc_plot = auc(fpr, tpr)


# combined PLOT (2x2) 
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Confusion matrix
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Benigno (B)", "Maligno (M)"],
    yticklabels=["Benigno (B)", "Maligno (M)"],
    ax=axes[0, 0]
)
axes[0, 0].set_title("Matrice di Confusione - SVM")
axes[0, 0].set_ylabel("Etichetta Vera")
axes[0, 0].set_xlabel("Etichetta Predetta")

# ROC curve
axes[0, 1].plot(fpr, tpr, lw=2, label=f"Curva ROC (AUC = {roc_auc_plot:.3f})")
axes[0, 1].plot([0, 1], [0, 1], lw=2, linestyle="--")
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel("Tasso di Falsi Positivi")
axes[0, 1].set_ylabel("Tasso di Veri Positivi")
axes[0, 1].set_title("Curva ROC - SVM")
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(True, alpha=0.3)

# Feature importance (top 10) CON COLORI + LEGENDA
top_features = feature_importance.head(10).copy()
top_features["Signed_Direction"] = top_features["Feature"].map(direction)
colors_top10 = ["red" if v > 0 else "green" for v in top_features["Signed_Direction"]]

axes[1, 0].barh(range(len(top_features)), top_features["Abs_Importance"], color=colors_top10)
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features["Feature"])
axes[1, 0].invert_yaxis()
axes[1, 0].set_xlabel("Importanza (Permutation, valore assoluto)")
axes[1, 0].set_title("Top 10 Feature Importanti - SVM")
axes[1, 0].grid(True, alpha=0.3, axis="x")

legend_elements = [
    Patch(facecolor="red", label="Rosso: associata a maggiore probabilità di malignità"),
    Patch(facecolor="green", label="Verde: associata a maggiore probabilità di benignità")
]
axes[1, 0].legend(handles=legend_elements, loc="lower right", frameon=True)

# Distribuzione probabilità predette
axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7,
                label="Benigno (B)", density=True, color="green")
axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7,
                label="Maligno (M)", density=True, color="red")
axes[1, 1].set_xlabel("Probabilità Predetta (Maligno)")
axes[1, 1].set_ylabel("Densità")
axes[1, 1].set_title("Distribuzione delle Probabilità Predette - SVM")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

main_plot_path = f"{plot_dir}/svm_analysis.png"
plt.savefig(main_plot_path, dpi=300, bbox_inches="tight")
print(f"\nPlot principale salvato in: {main_plot_path}")
plt.show()

#plot singoli
# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Benigno (B)", "Maligno (M)"],
    yticklabels=["Benigno (B)", "Maligno (M)"]
)
plt.title("Matrice di Confusione - SVM")
plt.ylabel("Etichetta Vera")
plt.xlabel("Etichetta Predetta")
plt.tight_layout()
plt.savefig(f"{plot_dir}/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f"Curva ROC (AUC = {roc_auc_plot:.3f})")
plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Tasso di Falsi Positivi")
plt.ylabel("Tasso di Veri Positivi")
plt.title("Curva ROC - SVM")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{plot_dir}/roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# Feature più rilevanti (TOP 15) con colori verde/rosso + LEGENDA
plt.figure(figsize=(10, 8))
top15 = feature_importance.head(15).copy()
top15["Signed_Direction"] = top15["Feature"].map(direction)
colors_top15 = ["red" if v > 0 else "green" for v in top15["Signed_Direction"]]

plt.barh(range(len(top15)), top15["Abs_Importance"], color=colors_top15)
plt.yticks(range(len(top15)), top15["Feature"])
plt.gca().invert_yaxis()
plt.xlabel("Importanza (Permutation, valore assoluto)")
plt.title("Top 15 Feature Più Importanti - SVM (Verde=Benigno, Rosso=Maligno)")
plt.grid(True, alpha=0.3, axis="x")

plt.legend(
    handles=[
        Patch(facecolor="red", label="Maligno (prob. maligno)"),
        Patch(facecolor="green", label="Benigno (prob. maligno)")
    ],
    loc="lower right",
    frameon=True
)

plt.tight_layout()
plt.savefig(f"{plot_dir}/feature_importance_top15.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot individuali salvati in: {plot_dir}/")

# Predizione su un campione
sample_idx = 0
sample_features = X_test.iloc[sample_idx:sample_idx+1]
true_label = y_test.iloc[sample_idx]
pred_label = best_model.predict(sample_features)[0]
pred_proba = best_model.predict_proba(sample_features)[0]

print(f"\nCampionamento dal test set (indice {sample_idx}):")
print(f"Diagnosi vera: {'Maligno (M)' if true_label == 1 else 'Benigno (B)'}")
print(f"Diagnosi predetta: {'Maligno (M)' if pred_label == 1 else 'Benigno (B)'}")
print(f"Probabilità di predizione: [Benigno: {pred_proba[0]:.4f}, Maligno: {pred_proba[1]:.4f}]")

# CROSS-VALIDATION
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy")
print(f"\nPunteggi cross-validation (5-fold): {cv_scores}")
print(f"Accuratezza media CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# TXT (stile Logistic) salvato in plot_dir
results_path = f"{plot_dir}/model_results.txt"

# Tabella "simile a coefficienti": direzione stimata * importanza
top10_txt = feature_importance.head(10).copy()
top10_txt["Coefficient"] = top10_txt["Feature"].map(direction) * top10_txt["Abs_Importance"]
top10_txt["Abs_Coefficient"] = np.abs(top10_txt["Coefficient"])
top10_txt = top10_txt[["Feature", "Coefficient", "Abs_Coefficient"]]

with open(results_path, "w") as f:
    f.write("RISULTATI MODELLO SVM\n")
    f.write(f"Accuratezza: {accuracy:.4f}\n")
    f.write(f"AUC-ROC: {roc_auc:.4f}\n\n")

    f.write("Matrice di Confusione:\n")
    f.write(str(cm) + "\n\n")

    f.write("Report di Classificazione:\n")
    f.write(report + "\n\n")

    f.write("Top 10 Feature Importance:\n")
    f.write(top10_txt.to_string(index=False) + "\n")

    f.write(f"Punteggi cross-validation (5-fold): {cv_scores}\n")
    f.write(f"Accuratezza media CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})\n")

print(f"\nRisultati completi salvati in: {results_path}")
print(f"Tutti i file sono stati salvati in: '{plot_dir}'")
