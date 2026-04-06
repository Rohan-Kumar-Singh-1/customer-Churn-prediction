# 🚀 Customer Churn Prediction (Full-Stack MLOps System)

## 📌 Overview

This project is a **production-style end-to-end Machine Learning system** for predicting customer churn using structured tabular data.

It combines:

* 🧠 **Deep Learning (ANN)**
* ⚙️ **Data Engineering Pipelines**
* 🌐 **API Deployment (FastAPI)**
* 🎨 **React Frontend Dashboard (Visualization)**
* 🐳 **Dockerized Deployment (Full-stack)**
* 🧪 **Automated Testing (pytest)**
* 🔁 **CI/CD (GitHub Actions)**

👉 Designed to demonstrate **real-world MLOps practices**, not just model training.

---

## 🎯 Objectives

* Predict customer churn (binary classification)
* Build a robust ANN model for tabular data
* Design reusable preprocessing pipelines
* Deploy model via REST API with a React frontend
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

### Backend:
* Python
* FastAPI
* TensorFlow / Keras
* Scikit-learn
* Pandas / NumPy

### Frontend:
* React.js
* Vite
* SVG (custom gauge visualization)

### DevOps / MLOps:
* Docker & Docker Compose
* GitHub Actions (CI/CD)
* Pytest

---

## 📂 Project Structure

```
customer-churn-prediction/
│
├── src/
│   ├── preprocessing.py      # Data pipelines
│   ├── model.py              # ANN architecture
│   └── train.py              # Training script
│
├── api/
│   ├── main.py               # FastAPI app
│   ├── model.py
│   ├── preprocessing.py
│   └── requirements.txt
│
├── frontend/                 # React frontend
│   ├── src/
│   └── public/
│
├── models/                   # Saved ML artifacts
│   ├── model.keras
│   └── preprocessor.pkl
│
├── plots/                    # Training plots & visualizations
│
├── tests/                    # Unit tests
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
│
├── .github/workflows/        # CI/CD pipeline
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── threshold.pkl
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
5. 💾 Save model + preprocessor + optimal threshold
6. 🌐 Serve via FastAPI
7. 🎨 Visualize via React dashboard
8. 🧪 Run unit tests
9. 🔁 Automate using CI/CD

---

## 🚀 Run with Docker (Recommended)

```bash
docker compose up --build
```

### 🌍 Access:
* Frontend → http://localhost:3000
* Backend → http://localhost:8000

---

## ⚙️ Run Locally (Without Docker)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Rohan-Kumar-Singh-1/customer-Churn-prediction.git
cd customer-Churn-prediction
```

### 2️⃣ Install dependencies

```bash
pip install -r api/requirements.txt
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

### 6️⃣ Run frontend

```bash
cd frontend
npm install
npm run dev
```

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

## 🧪 Running Tests

```bash
PYTHONPATH=. pytest
```

✔ Tests include:
* Data preprocessing validation
* Model output shape
* API response check

---

## 🔁 CI/CD Pipeline

* Triggered on every push & pull request
* Installs dependencies
* Runs all unit tests
* Ensures code reliability

👉 Pipeline status visible in GitHub Actions (green ✅)

---

## 📊 Dataset

* **Source:** Telco Customer Churn Dataset
* **Rows:** ~7000
* **Features:** 20+
* **Target:** Churn (Yes / No)

---

## 🤔 Why ANN?

Although tree-based models are common for tabular data, ANN was chosen to:

* Capture complex non-linear relationships
* Demonstrate deep learning capabilities
* Apply regularization techniques (Dropout, L2)
* Showcase scalable ML system design

---

## 🚀 Future Improvements

* 📈 Model explainability (SHAP)
* 📦 Model versioning (MLflow)
* 📊 Monitoring & logging
* ☁️ Cloud deployment (AWS / Render)
* 🔐 Authentication layer
* 🔁 Hyperparameter tuning

---

## 📚 Key Learnings

* Building ANN models for tabular data
* Handling class imbalance using class weights
* Preventing overfitting with Dropout + L2
* Designing modular ML pipelines
* Deploying ML models as REST APIs
* Frontend-backend integration with React
* Docker-based containerization
* Implementing CI/CD for ML systems

---

## 👨‍💻 Author

**Rohan Kumar**

---

⭐ If you found this project useful, consider giving it a star!
