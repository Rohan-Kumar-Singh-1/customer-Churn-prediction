#  Customer Churn Prediction (MLOps Project)

##  Overview

This project builds an end-to-end **Machine Learning system** to predict whether a customer is likely to churn.
It uses a **Deep Artificial Neural Network (ANN)** trained on structured tabular data and is deployed using **FastAPI** with full **MLOps practices** including testing and CI/CD.

---

##  Objectives

* Predict customer churn (binary classification)
* Build a deep learning model for tabular data
* Apply proper preprocessing pipelines
* Deploy model using FastAPI
* Implement unit testing with pytest
* Automate testing using CI/CD (GitHub Actions)

---

##  Model Details

* Architecture: Feedforward Artificial Neural Network (ANN)
* Layers: Dense + Dropout (for regularization)
* Output: Sigmoid (binary classification)
* Loss: Binary Crossentropy
* Optimizer: Adam

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The model is optimized to balance **precision and recall**, ensuring effective detection of churn customers.

---

## Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas / NumPy
* FastAPI
* Pytest
* GitHub Actions (CI/CD)

---

##  Project Structure

```
churn-ann/
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│
├── api/
│   └── main.py
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
├── .github/workflows/main.yml
├── requirements.txt
└── README.md
```

---

##  Workflow

1. Data preprocessing using pipelines (imputation + encoding + scaling)
2. Model training with dropout and early stopping
3. Evaluation using F1-score and confusion matrix
4. Model saving and loading
5. API deployment using FastAPI
6. Automated testing using pytest
7. CI/CD pipeline using GitHub Actions

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository

```
git clone https://github.com/<your-username>/customer-Churn-prediction.git
cd customer-Churn-prediction
```

---

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Train the model

```
python -m src.train
```

---

### 4️⃣ Run the API

```
uvicorn api.main:app --reload
```

---

### 5️⃣ Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

##  Running Tests

```
PYTHONPATH=. pytest
```

---

##  API Endpoint

### POST `/predict`

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

##  CI/CD Pipeline

* Automated testing using GitHub Actions
* Runs pytest on every push and pull request
* Ensures code quality and reliability

---

## Key Learnings

* Building ANN models for tabular data
* Handling class imbalance using class weights
* Preventing overfitting with dropout and regularization
* Designing scalable ML pipelines
* Deploying ML models as APIs
* Implementing CI/CD for ML systems
