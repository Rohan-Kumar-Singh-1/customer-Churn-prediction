import os

# CPU optimization
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"

import tensorflow as tf
import multiprocessing
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import numpy as np
from src.preprocessing import preprocess_data
from src.model import build_model

# Thread tuning
num_cores = multiprocessing.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)

# XLA acceleration
tf.config.optimizer.set_jit(True)


def train():
    # 📥 Load data
    df = pd.read_csv("churn.csv")

    # 🔄 Preprocess
    X, y = preprocess_data(df, target_col="Churn")

    # ✅ Train/Test Split (IMPORTANT)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # ⚖️ Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(enumerate(class_weights))

    print("\nClass Weights:", class_weights)

    # 🧠 Build model
    model = build_model(X.shape[1])

    # ⏹ Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # 🚀 Train
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=64,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1
    )

    # 💾 Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/model.keras")

    # 🔮 Predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.6).astype(int)

    # 📊 Classification Report (Precision, Recall, F1)
    print("\n================= CLASSIFICATION REPORT =================\n")
    print(classification_report(y_test, y_pred))

    # 📊 Confusion Matrix
    print("\n================= CONFUSION MATRIX =================\n")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    train()