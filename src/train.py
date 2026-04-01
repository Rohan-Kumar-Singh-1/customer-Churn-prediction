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
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils import class_weight

import matplotlib.pyplot as plt
import seaborn as sns

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

    # ✅ Train/Test Split
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

    # 🚀 Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=64,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1
    )

    # 📁 Create plots directory
    os.makedirs("plots", exist_ok=True)

    # ============================
    # 📊 1. LOSS CURVE
    # ============================
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.savefig("plots/loss_curve.png")
    plt.close()

    # ============================
    # 📊 2. ACCURACY CURVE
    # ============================
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.savefig("plots/accuracy_curve.png")
    plt.close()

    # 💾 Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/model.keras")

    # 🔮 Predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.6).astype(int)

    # ============================
    # 📊 3. CONFUSION MATRIX
    # ============================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

    # ============================
    # 📊 4. ROC CURVE
    # ============================
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("plots/roc_curve.png")
    plt.close()

    # 📊 Classification Report
    print("\n================= CLASSIFICATION REPORT =================\n")
    print(classification_report(y_test, y_pred))

    print("\n================= CONFUSION MATRIX =================\n")
    print(cm)


if __name__ == "__main__":
    train()