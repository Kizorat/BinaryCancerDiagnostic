import pandas as pd
import numpy as np

# Caricamento dati
data = pd.read_csv('dataset/data.csv')


print(f"\nShape: {data.shape}")
print(f"\nPrime righe:\n{data.head()}")
print(f"\nInfo:")
data.info()
print(f"\nValori mancanti per colonna:\n{data.isnull().sum()}")
print(f"\nNumero totale di valori mancanti: {data.isnull().sum().sum()}")
print(f"\nStatistiche descrittive:\n{data.describe()}")
print(f"\nDistribuzione diagnosis:\n{data['diagnosis'].value_counts()}")

# Rimozione colonne vuote

empty_cols = data.columns[data.isnull().all()].tolist()
print(f"\nColonne completamente vuote: {empty_cols}")

if len(empty_cols) > 0:
    data_clean = data.drop(columns=empty_cols)
    print(f"Shape dopo rimozione colonne vuote: {data_clean.shape}")
    print(f"Colonne rimosse: {len(empty_cols)}")
else:
    data_clean = data.copy()
    print("\nNessuna colonna vuota trovata!")

# Rimozione duplicati

print(f"\nDuplicati presenti: {data_clean.duplicated().sum()}")
data_clean = data_clean.drop_duplicates()
print(f"Shape dopo rimozione duplicati: {data_clean.shape}")
print(f"Righe rimosse: {data.shape[0] - data_clean.shape[0]}")

# Gestione valori mancanti
print(f"\nValori mancanti per colonna:\n{data_clean.isnull().sum()}")
print(f"\nNumero totale di valori mancanti: {data_clean.isnull().sum().sum()}")

if data_clean.isnull().sum().sum() > 0:
    print("\nRimozione righe con valori mancanti...")
    before_shape = data_clean.shape
    data_clean = data_clean.dropna()
    print(f"Shape dopo rimozione NaN: {data_clean.shape}")
    print(f"Righe rimosse: {before_shape[0] - data_clean.shape[0]}")
else:
    print("\nNessun valore mancante trovato!")

# Verifica outliers con metodo IQR


numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col != 'id']

outliers_summary = {}
for col in numeric_cols:
    Q1 = data_clean[col].quantile(0.25)
    Q3 = data_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = ((data_clean[col] < lower_bound) | (data_clean[col] > upper_bound)).sum()
    
    if outliers > 0:
        outliers_summary[col] = outliers

print(f"\nColonne con outliers: {len(outliers_summary)}")
print(f"Totale valori outlier: {sum(outliers_summary.values())}")

if len(outliers_summary) > 0:
    print("\nTop 5 colonne con più outliers:")
    sorted_outliers = sorted(outliers_summary.items(), key=lambda x: x[1], reverse=True)[:5]
    for col, count in sorted_outliers:
        print(f"  {col}: {count} outliers")


# Verifica colonne costanti


constant_cols = []
for col in numeric_cols:
    if data_clean[col].nunique() == 1:
        constant_cols.append(col)
        print(f"\nColonna costante: {col}")
        print(f"  Valore unico: {data_clean[col].unique()[0]}")

if len(constant_cols) > 0:
    data_clean = data_clean.drop(columns=constant_cols)
    print(f"\nShape dopo rimozione colonne costanti: {data_clean.shape}")
else:
    print("\nNessuna colonna costante trovata!")

# Verifica correlazioni perfette


# Aggiorna numeric_cols dopo eventuali rimozioni
numeric_cols = [col for col in numeric_cols if col in data_clean.columns]
numeric_data = data_clean[numeric_cols]
corr_matrix = numeric_data.corr().abs()

upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_pairs = []
for column in upper_triangle.columns:
    for index in upper_triangle.index:
        if upper_triangle.loc[index, column] == 1.0:
            high_corr_pairs.append((index, column))

if len(high_corr_pairs) > 0:
    print(f"\nTrovate {len(high_corr_pairs)} coppie con correlazione perfetta:")
    for pair in high_corr_pairs:
        print(f"  {pair[0]} <-> {pair[1]}")
else:
    print("\nNessuna correlazione perfetta trovata!")

# Reset index

data_clean = data_clean.reset_index(drop=True)
print(f"Index resettato. Range: 0 - {len(data_clean)-1}")



print(f"\nShape originale: {data.shape}")
print(f"Shape finale: {data_clean.shape}")
print(f"Righe rimosse: {data.shape[0] - data_clean.shape[0]}")
print(f"Colonne rimosse: {data.shape[1] - data_clean.shape[1]}")

if len(data_clean) > 0:
    print(f"\nDistribuzione diagnosis finale:\n{data_clean['diagnosis'].value_counts()}")
    print(f"\nPercentuali:")
    print(data_clean['diagnosis'].value_counts(normalize=True) * 100)
    print(f"\nColonne finali: {list(data_clean.columns)}")
else:
    print("\nDataset vuoto dopo la pulizia!")

# Salvataggio dataset pulito
output_file = 'dataset/clean_data.csv'
data_clean.to_csv(output_file, index=False)
print(f"\n\nDataset pulito salvato in: {output_file}")
