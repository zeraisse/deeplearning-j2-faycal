import os
os.environ["KERAS_BACKEND"] = "torch"
import datetime
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras import layers
from torch.utils.tensorboard import SummaryWriter

# ==========================================
# PARTIE 1 : PRÉPARATION DES DONNÉES
# ==========================================
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# ==========================================
# PARTIE 2 : CALLBACK CUSTOM PYTORCH
# ==========================================
class TorchTensorBoardCallback(keras.callbacks.Callback):
    """Écrit les logs TensorBoard en utilisant PyTorch au lieu de TensorFlow."""
    def __init__(self, log_dir):
        super().__init__()
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for name, value in logs.items():
            self.writer.add_scalar(name, value, epoch)

    def on_train_end(self, logs=None):
        self.writer.close()

# ==========================================
# PARTIE 3 : MODÈLE ET ENTRAÎNEMENT
# ==========================================
def build_regression_model(input_dim):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,))) # Correction du UserWarning
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_with_tensorboard(X_t, y_t, X_v, y_v, run_name, epochs=100):
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    log_dir = os.path.join("logs", "fit", f"{run_name}_{timestamp}")
    
    # Utilisation du callback PyTorch
    tb_callback = TorchTensorBoardCallback(log_dir=log_dir)
    
    model = build_regression_model(input_dim=8)
    
    print(f"Entraînement de '{run_name}' en cours...")
    history = model.fit(
        X_t, 
        y_t,
        validation_data=(X_v, y_v),
        epochs=epochs,
        batch_size=32,
        callbacks=[tb_callback],
        verbose=0
    )
    
    print(f"Run '{run_name}' terminé. Logs dans {log_dir}")
    return model, history

# ==========================================
# EXÉCUTION
# ==========================================
model_norm, history_norm = train_with_tensorboard(
    X_train_norm, y_train, 
    X_val_norm, y_val,
    run_name="california_norm"
)

model_raw, history_raw = train_with_tensorboard(
    X_train, y_train, 
    X_val, y_val,
    run_name="california_raw"
)