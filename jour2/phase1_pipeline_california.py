import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Charger le dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
# StandardScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
# normalise chaque feature : (X - mean) / std.
# Résultat : mean = 0, std = 1 sur le train set.
# Pourquoi fitter sur X_train uniquement : si on fitte sur X entier,
# les stats du scaler "voient" le test set avant l'évaluation (data leakage).
# TODO : faire un premier split train/test avec test_size=0.2 et random_state=42


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO : faire un second split train/val sur le résultat précédent (val_size=0.2 du train)

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, 
    y_train, 
    test_size=0.2, 
    random_state=42
)
# TODO : instancier un StandardScaler et le fitter sur X_train UNIQUEMENT
scaler = StandardScaler()
X_train_final_scaled = scaler.fit_transform(X_train_final)
# TODO : transformer X_train, X_val, X_test avec le scaler fitté
X_train = scaler.transform(X_train)
X_train_final_scaled = scaler.transform(X_train_final_scaled)



X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
# TODO : afficher les shapes de X_train, X_val, X_test
print(f"X_train_final : {X_train_final.shape}")
print(f"X_val : {X_val.shape}")
print(f"X_test : {X_test.shape}")
# TODO : afficher les stats descriptives de X_train_norm (mean et std par feature)
moyennes = np.mean(X_train, axis=0)
ecarts_types = np.std(X_train, axis=0)
# TODO : afficher les feature_names du dataset ET vérifier qu'il y en a bien 8
print(f"Moyennes par feature     : {np.round(moyennes, 2)}")
print(f"Écarts-types par feature : {np.round(ecarts_types, 2)}\n")

X_extreme = np.array([[99999, -99999, 0, 0, 0, 0, 37.0, -120.0]])

X_extreme = scaler.transform(X_extreme)
print(f"X_extreme : {X_extreme.shape}")
