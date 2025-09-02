# Credit Card Fraud Detection System

A machine learning project to detect fraudulent credit card transactions from highly imbalanced real-world datasets.  
The project covers the full pipeline from **data preprocessing → handling class imbalance → model training → evaluation → deployment** as an API.

---

## Project Overview
Fraud detection is a critical problem in finance due to the **highly imbalanced nature** of datasets, where fraudulent cases are <1% of total transactions.  
This project demonstrates how to:
- Preprocess and explore transaction data.
- Handle class imbalance using advanced techniques like **SMOTE**.
- Train, tune, and compare multiple ML models.
- Evaluate models using metrics suitable for imbalanced classification.
- Deploy the best model as a **FastAPI REST API** for real-time fraud detection.

---

## Tech Stack
- **Programming Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, Imbalanced-learn, LightGBM, XGBoost, Matplotlib, SHAP  
- **Deployment**: FastAPI, Uvicorn  
- **Dataset**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## Project Workflow

### 1. Data Preprocessing & EDA
- Loaded anonymized transaction data (`V1–V28` + Amount, Time).
- Conducted **EDA** to study class imbalance, correlations, and feature distributions.
- Scaled numerical features with **RobustScaler** to handle outliers.

### 2. Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance minority fraud cases.
- Used **stratified sampling** to ensure representative train-test splits.

### 3. Model Training & Hyperparameter Tuning
- Trained multiple models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- Performed **RandomizedSearchCV** for hyperparameter optimization.

### 4. Model Evaluation
- Metrics used:
  - **Confusion Matrix**
  - **Precision–Recall Curve**
  - **AUPRC (Area Under Precision-Recall Curve)**
- Selected LightGBM as the best-performing model.

### 5. Model Interpretation
- Used **SHAP values** to explain model predictions and feature importance.

### 6. API Deployment
- Built a **FastAPI endpoint** for real-time fraud detection.
- Input: Transaction features (preprocessed).
- Output: Probability of fraud + prediction label.

---

## Results
- Achieved strong recall and precision trade-off on fraud detection.
- Significantly reduced **false negatives** compared to baseline logistic regression.
- Deployed API enables real-time fraud checks in production pipelines.

---

## Installation & Usage

```bash
# Clone repo
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run FastAPI app
uvicorn app:app --reload
