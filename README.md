# 🚀 Customer Churn Prediction (End-to-End MLOps Project)

## 📌 Overview

This project builds a **production-style Machine Learning system** to predict customer churn using structured tabular data.

It combines:

* 🧠 **Deep Learning (ANN)**
* ⚙️ **Data Engineering Pipelines**
* 🌐 **API Deployment (FastAPI)**
* 🧪 **Automated Testing (pytest)**
* 🔁 **CI/CD (GitHub Actions)**

👉 Designed to demonstrate **real-world MLOps practices**, not just model training.

---

## 🎯 Objectives

* Predict customer churn (binary classification)
* Build a robust ANN model for tabular data
* Design reusable preprocessing pipelines
* Deploy model via REST API
* Ensure reliability with unit testing
* Automate workflows using CI/CD

---

## 🧠 Model Architecture

* **Type:** Feedforward Artificial Neural Network (ANN)
* **Layers:** Dense + Dropout (regularization)
* **Output Layer:** Sigmoid (probability output)
* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam
* **Regularization:** Dropout + L2

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas / NumPy
* FastAPI
* Pytest
* GitHub Actions (CI/CD)

---

## 📂 Project Structure

```
customer-churn-prediction/
│
├── src/
│   ├── preprocessing.py      # Data pipelines
│   ├── model.py              # ANN architecture
│   ├── train.py              # Training script
│
├── api/
│   └── main.py               # FastAPI app
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_api.py
│
├── models/
│   ├── model.keras
│   ├── preprocessor.pkl
│
├── .github/workflows/main.yml   # CI pipeline
├── requirements.txt
└── README.md
```

---

## 🔄 Workflow

1. 📥 Load dataset (CSV)
2. 🧹 Preprocess data:

   * Missing value imputation
   * One-hot encoding
   * Feature scaling
3. 🧠 Train ANN with:

   * Dropout
   * Early stopping
   * Class weights (imbalance handling)
4. 📊 Evaluate model (F1-score, confusion matrix)
5. 💾 Save model + preprocessor
6. 🌐 Serve via FastAPI
7. 🧪 Run unit tests
8. 🔁 Automate using CI/CD

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/customer-Churn-prediction.git
cd customer-Churn-prediction
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model

```bash
python -m src.train
```

### 4️⃣ Run the API

```bash
uvicorn api.main:app --reload
```

### 5️⃣ Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Running Tests

```bash
PYTHONPATH=. pytest
```

✔ Tests include:

* Data preprocessing validation
* Model output shape
* API response check

---

## 🌐 API Endpoint

### 🔹 POST `/predict`

#### Sample Input:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.5,
  "TotalCharges": 350.5
}
```

#### Sample Output:

```json
{
  "probability": 0.59,
  "prediction": 0,
  "label": "No Churn",
  "risk_level": "High Risk"
}
```

---

## 🔁 CI/CD Pipeline

* Triggered on every push & pull request
* Installs dependencies
* Runs all unit tests
* Ensures code reliability

👉 Final pipeline shows **successful execution (green status)**

---

## 📂 Dataset

* **Source:** Telco Customer Churn Dataset
* **Rows:** ~7000
* **Features:** 20+
* **Target:** Churn (Yes/No)

---

## 🤔 Why ANN?

Although tree-based models are common for tabular data, ANN was used to:

* Capture complex non-linear relationships
* Demonstrate deep learning capabilities
* Apply regularization techniques (Dropout, L2)
* Showcase scalable ML system design

---

## 🚀 Future Improvements

* 🐳 Add Docker for containerization
* 📦 Implement model versioning
* 📈 Hyperparameter tuning
* 📊 Add monitoring & logging
* ☁️ Deploy on cloud (AWS / Render)

---

## 📚 Key Learnings

* Building ANN models for tabular data
* Handling class imbalance using class weights
* Preventing overfitting with dropout
* Designing modular ML pipelines
* Deploying ML models as APIs
* Implementing CI/CD for ML systems

---

## 👨‍💻 Author

**Rohan Kumar**

---

⭐ If you found this project useful, consider giving it a star!
