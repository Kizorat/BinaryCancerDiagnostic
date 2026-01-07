import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
import os
import warnings
warnings.filterwarnings('ignore')

# Creazione directory per i plot
plot_dir = "dataset/plot_logistic_regression"
os.makedirs(plot_dir, exist_ok=True)

# Caricamento dei dati
df = pd.read_csv('dataset/clean_data.csv')

# 1. Preprocessing dei dati
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})


X = df.drop('diagnosis', axis=1)
# Rimozione colonna 'id' 
if 'id' in X.columns:
    X = X.drop('id', axis=1)
y = df['diagnosis']

# Divisione in training set e test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardizzione delle feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Addestramento del modello di regressione logistica
log_reg = LogisticRegression(
    max_iter=1000,  
    random_state=42,
    solver='lbfgs',  
    class_weight='balanced'
)

log_reg.fit(X_train_scaled, y_train)

# Predizione sul test set
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]


# Accuratezza
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# AUC-ROC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC: {roc_auc:.4f}")

# Matrice di confusione
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Report di classificazione
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benigno (B)', 'Maligno (M)']))

# 5. Analisi dei coefficienti
# Creazione DataFrame per importanza delle feature
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_reg.coef_[0],
    'Abs_Coefficient': np.abs(log_reg.coef_[0])
})

# Ordinamento per importanza assoluta
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
print("\nTop 10 Feature Importance:")
print(feature_importance.head(10).to_string(index=False))

# feature importance to CSV
feature_importance.to_csv(f'{plot_dir}/feature_importance.csv', index=False)
print(f"\nFeature importance salvata in: {plot_dir}/feature_importance.csv")



fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Matrice di confusione
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benigno (B)', 'Maligno (M)'],
            yticklabels=['Benigno (B)', 'Maligno (M)'], ax=axes[0, 0])
axes[0, 0].set_title('Matrice di Confusione')
axes[0, 0].set_ylabel('Etichetta Vera')
axes[0, 0].set_xlabel('Etichetta Predetta')

#  Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'Curva ROC (AUC = {roc_auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('Tasso di Falsi Positivi')
axes[0, 1].set_ylabel('Tasso di Veri Positivi')
axes[0, 1].set_title('Curva ROC (Receiver Operating Characteristic)')
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(True, alpha=0.3)

# Prime 10 feature più importanti
top_features = feature_importance.head(10)
colors = ['red' if coef < 0 else 'green' for coef in top_features['Coefficient']]
bars = axes[1, 0].barh(range(len(top_features)), top_features['Abs_Coefficient'], color=colors)
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['Feature'])
axes[1, 0].invert_yaxis()
axes[1, 0].set_xlabel('Valore Assoluto del Coefficiente')
axes[1, 0].set_title('Top 10 Feature Più Importanti')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Legenda per i colori
axes[1, 0].text(0.95, 0.05, 'Rosso: aumenta probabilità malignità\nVerde: aumenta probabilità benignità',
                transform=axes[1, 0].transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# Distribuzione delle probabilità predette
axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, 
                label='Benigno (B)', color='green', density=True)
axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, 
                label='Maligno (M)', color='red', density=True)
axes[1, 1].set_xlabel('Probabilità Predetta (Maligno)')
axes[1, 1].set_ylabel('Densità')
axes[1, 1].set_title('Distribuzione delle Probabilità Predette')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Salvare il plot principale
plot_path = f'{plot_dir}/logistic_regression_analysis.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot principale salvato in: {plot_path}")

# Salvare plot individuali separati
# Matrice di confusione individuale
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benigno (B)', 'Maligno (M)'],
            yticklabels=['Benigno (B)', 'Maligno (M)'])
plt.title('Matrice di Confusione - Regressione Logistica')
plt.ylabel('Etichetta Vera')
plt.xlabel('Etichetta Predetta')
plt.tight_layout()
plt.savefig(f'{plot_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Curva ROC individuale
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'Curva ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasso di Falsi Positivi')
plt.ylabel('Tasso di Veri Positivi')
plt.title('Curva ROC - Regressione Logistica')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plot_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature importance individuale
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
colors = ['red' if coef < 0 else 'green' for coef in top_features['Coefficient']]
plt.barh(range(len(top_features)), top_features['Abs_Coefficient'], color=colors)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.gca().invert_yaxis()
plt.xlabel('Valore Assoluto del Coefficiente')
plt.title('Top 15 Feature Più Importanti - Regressione Logistica')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{plot_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot individuali salvati in: {plot_dir}/")

# plot finale
plt.show()

# Riepilogo interpretativo

print("1. Le feature più importanti per distinguere tra tumori maligni e benigni sono:")
print("   - Coefficienti positivi (rosso): aumentano la probabilità di malignità")
print("   - Coefficienti negativi (verde): aumentano la probabilità di benignità")
print(f"\n2. Il modello ha un'accuratezza del {accuracy*100:.2f}% sul test set.")
print(f"3. L'area sotto la curva ROC è {roc_auc:.3f} (1.0 = perfetto, 0.5 = casuale).")

# predizione su un campione del test set

sample_idx = 0
sample_features = X_test_scaled[sample_idx].reshape(1, -1)
true_label = y_test.iloc[sample_idx]
pred_label = log_reg.predict(sample_features)[0]
pred_proba = log_reg.predict_proba(sample_features)[0]

print(f"Campionamento dal test set (indice {sample_idx}):")
print(f"Diagnosi vera: {'Maligno (M)' if true_label == 1 else 'Benigno (B)'}")
print(f"Diagnosi predetta: {'Maligno (M)' if pred_label == 1 else 'Benigno (B)'}")
print(f"Probabilità di predizione: [Benigno: {pred_proba[0]:.4f}, Maligno: {pred_proba[1]:.4f}]")

# Cross-validation


cv_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Punteggi cross-validation (5-fold): {cv_scores}")
print(f"Accuratezza media CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Salvataggio risultati completi su file di testo
results_path = f'{plot_dir}/model_results.txt'
with open(results_path, 'w') as f:
    f.write("RISULTATI MODELLO REGRESSIONE LOGISTICA\n")
    f.write(f"Accuratezza: {accuracy:.4f}\n")
    f.write(f"AUC-ROC: {roc_auc:.4f}\n\n")
    f.write("Matrice di Confusione:\n")
    f.write(str(cm) + "\n\n")
    f.write("Report di Classificazione:\n")
    f.write(classification_report(y_test, y_pred, target_names=['Benigno (B)', 'Maligno (M)']))
    f.write("\nTop 10 Feature Importance:\n")
    f.write(feature_importance.head(10).to_string(index=False) + "\n")
    f.write(f"Punteggi cross-validation (5-fold): {cv_scores}\n")
    f.write(f"Accuratezza media CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})\n")

print(f"\nRisultati completi salvati in: {results_path}")
print(f"\nTutti i file sono stati salvati nella cartella: '{plot_dir}'")