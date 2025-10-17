import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import gc

df = pd.read_csv("cac40.csv", low_memory=False)

# correction des types des colones
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
for col in ["Open", "High", "Low", "Close", "Volume"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Optimisation des types pour la memoire
for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")
for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = df[col].astype("int32")

gc.collect()

# Suppression des doublons (par dates)
avant = len(df)
df = df.drop_duplicates(subset=["Date"])
apres = len(df)
print(f"Doublons supprimés : {avant - apres}")

# Vérification de la continuité des dates
df = df.sort_values("Date").reset_index(drop=True)
df["date_diff"] = df["Date"].diff().dt.days
ecarts_uniques = df["date_diff"].dropna().unique()
grands_ecarts = df[df["date_diff"] > 3]
if grands_ecarts.empty:
    print("Les dates se suivent correctement (écarts normaux de 1 à 3 jours).")
else:
    print(f"{len(grands_ecarts)} écarts inhabituels détectés :")
    print(grands_ecarts[["Date", "date_diff"]].head())

# Gestion des valeurs manquantes
print("\nValeurs manquantes avant traitement :")
print(df.isna().sum())

# Remplissage des valeurs manquantes
for col in ["Open", "High", "Low", "Close"]:
    if col in df.columns:
        df[col] = df[col].ffill().bfill()

if "Volume" in df.columns:
    df["Volume"] = df["Volume"].fillna(0)

df['Return'] = df['Close'].pct_change().astype('float32')
df['Volatility'] = ((df['High'] - df['Low']) / df['Low']).astype('float32')

# Évolution temporelle du CAC40 (close)
plt.figure(figsize=(12,5))
plt.plot(df['Date'], df['Close'], color='tab:blue', label='Close')
plt.title('Évolution du CAC40 (Close)')
plt.xlabel('Date')
plt.ylabel('Cours de clôture')
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()

# Corrélation entre variables clés : Prix, Volume, Volatilité
cols_corr = ['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility']
corr_matrix = df[cols_corr].corr()
plt.figure(figsize=(8,6))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Corrélation')
plt.xticks(range(len(cols_corr)), cols_corr, rotation=45)
plt.yticks(range(len(cols_corr)), cols_corr)
plt.title('Corrélation entre variables clés : Prix, Volume, Volatilité')
plt.tight_layout()
# plt.show()

# Histogramme des rendements journaliers
plt.figure(figsize=(10,4))
plt.hist(df['Return'].dropna(), bins=100, color='skyblue', edgecolor='black')
plt.title('Distribution des rendements journaliers du CAC40')
plt.xlabel('Rendement')
plt.ylabel('Fréquence')
plt.tight_layout()
# plt.show()

# Anomalies
df['z_return'] = zscore(df['Return'].fillna(0))
df['z_volatility'] = zscore(df['Volatility'].fillna(0))
threshold = 3

df['Anomaly'] = ((df['z_return'].abs() > threshold) | (df['z_volatility'].abs() > threshold))
print(f"Nombre d'anomalies détectées : {df['Anomaly'].sum()}")

plt.figure(figsize=(12,5))
plt.plot(df['Date'], df['Close'], color='tab:blue', label='Close')
plt.scatter(df.loc[df['Anomaly'], 'Date'], df.loc[df['Anomaly'], 'Close'],
            color='red', label='Anomalie', marker='o')
plt.title('Évolution du CAC40 avec anomalies détectées')
plt.xlabel('Date')
plt.ylabel('Cours de clôture')
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()

plt.figure(figsize=(10,4))
plt.hist(df['Return'].dropna(), bins=100, color='skyblue', edgecolor='black')
plt.hist(df.loc[df['Anomaly'], 'Return'], bins=20, color='red', alpha=0.6, label='Anomalies')
plt.title('Rendements journaliers et anomalies')
plt.xlabel('Rendement')
plt.ylabel('Fréquence')
plt.legend()
plt.tight_layout()
# plt.show()

df.to_csv("cac40_clean.csv", index=False)

# Comparaison avec le fichier parquet
sto = pd.read_parquet("sto.parquet")
sto['Date'] = pd.to_datetime(sto['Date'], unit='D')
for col in ["Open", "High", "Low", "Close", "Volume"]:
    if col in sto.columns:
        sto[col] = pd.to_numeric(sto[col], errors="coerce")

sto.rename(columns={
    'Open':'Open_STO', 'High':'High_STO', 'Low':'Low_STO', 'Close':'Close_STO', 'Volume':'Volume_STO'
}, inplace=True)

sto['Return_STO'] = sto['Close_STO'].pct_change().astype('float32')
sto['Volatility_STO'] = ((sto['High_STO'] - sto['Low_STO']) / sto['Low_STO']).astype('float32')
df_merged = pd.merge(df, sto, on="Date", how="inner")

plt.figure(figsize=(12,5))
plt.plot(df_merged['Date'], df_merged['Close'], label='CAC40 Close', color='tab:blue')
plt.plot(df_merged['Date'], df_merged['Close_STO'], label='STO Close', color='tab:orange', alpha=0.8)
plt.title('Comparaison CAC40 et STO')
plt.xlabel('Date')
plt.ylabel('Cours de clôture')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()

# Pediction avec ML
df['MA_10'] = df['Close'].rolling(10).mean()
df['MA_20'] = df['Close'].rolling(20).mean()
df['Momentum_20'] = df['Close'] - df['Close'].shift(20)
df['Prediction'] = np.where((df['MA_10'] > df['MA_20']) & (df['Momentum_20'] > 0), 1, 0)
df['Actual'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
accuracy = (df['Prediction'] == df['Actual']).mean()
print(f"Précision du modèle simple basé sur les moyennes mobiles et momentum : {accuracy:.2%}")

plt.figure(figsize=(12,5))
plt.plot(df['Date'], df['Close'], color='tab:blue', label='Close')
plt.scatter(df.loc[df['Prediction']==1, 'Date'], df.loc[df['Prediction']==1, 'Close'],
            color='green', label='Prédiction hausse', marker='^', alpha=0.6)
plt.scatter(df.loc[df['Prediction']==0, 'Date'], df.loc[df['Prediction']==0, 'Close'],
            color='red', label='Prédiction baisse', marker='v', alpha=0.6)
plt.title('CAC40 : prédictions basées sur MA et Momentum')
plt.xlabel('Date')
plt.ylabel('Cours de clôture')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()

df.to_parquet("cac40_stat_analysis.parquet", index=False)
print("Données enrichies sauvegardées : cac40_stat_analysis.parquet")

# # Prédictions
df['MA_5'] = df['Close'].rolling(5).mean()
df['MA_20'] = df['Close'].rolling(20).mean()
df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
df['ROC_5'] = df['Close'].pct_change(5)
df['Volatility_5'] = df['Close'].pct_change().rolling(5).std()
df['High_Low_Range'] = df['High'] - df['Low']
df['Volume_SMA_5'] = df['Volume'].rolling(5).mean()
df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()

df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
features = ['MA_5','MA_20','EMA_5','Momentum_5','ROC_5','Volatility_5',
            'High_Low_Range','Volume_SMA_5','Volume_SMA_10']
df_ml = df.dropna(subset=features + ['Target'])

X = df_ml[features]
y = df_ml['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle Random Forest : {accuracy*100:.2f}%")

plt.figure(figsize=(12,5))
plt.plot(df_ml['Date'].iloc[-len(y_test):], df_ml['Close'].iloc[-len(y_test):], 
         label='Cours réel', color='tab:blue')
plt.scatter(df_ml['Date'].iloc[-len(y_test):][y_pred==1],
            df_ml['Close'].iloc[-len(y_test):][y_pred==1],
            label='Prédiction hausse', color='green', marker='^', alpha=0.6)
plt.scatter(df_ml['Date'].iloc[-len(y_test):][y_pred==0],
            df_ml['Close'].iloc[-len(y_test):][y_pred==0],
            label='Prédiction baisse', color='red', marker='v', alpha=0.6)
plt.title("Prédictions Random Forest : Hausse vs Baisse")
plt.xlabel("Date")
plt.ylabel("Cours de clôture")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()