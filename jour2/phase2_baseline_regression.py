import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras import layers

# ==========================================
# PARTIE 1 : PRÉPARATION DES DONNÉES
# ==========================================

housing = fetch_california_housing()
X, y = housing.data, housing.target

# Premier split train/test
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Second split train/val
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.2, random_state=42)

# Instancier et appliquer le StandardScaler
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train) # Fit uniquement sur le Train final
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# Vérifications
print(f"X_train_norm : {X_train_norm.shape}")
print(f"X_val_norm : {X_val_norm.shape}")
print(f"X_test_norm : {X_test_norm.shape}")

moyennes = np.mean(X_train_norm, axis=0)
ecarts_types = np.std(X_train_norm, axis=0)
print(f"Moyennes par feature     : {np.round(moyennes, 2)}")
print(f"Écarts-types par feature : {np.round(ecarts_types, 2)}\n")

feature_names = housing.feature_names
assert len(feature_names) == 8, "Erreur: pas 8 features"
print(f"Features (8) : {feature_names}\n")

# ==========================================
# PARTIE 2 : MODÈLE DEEP LEARNING
# ==========================================

def build_regression_model(input_dim):
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_regression_model(input_dim=8)
model.summary()

history = model.fit(
    X_train_norm, 
    y_train, 
    epochs=100, 
    batch_size=32,
    validation_data=(X_val_norm, y_val), 
    verbose=1
)

test_loss, test_mae = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"\nMAE test : {test_mae:.4f} (en centaines de milliers de $)")