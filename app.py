# app.py

# --- 1. Imports ---
# No changes to imports
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# --- 2. Load ML Assets ---
# This section remains unchanged
try:
    print("--- Loading ML Assets ---")
    model = joblib.load("final_fraud_detection_model.joblib")
    scaler = joblib.load("robust_scaler.joblib")
    print("--- ML Assets Loaded Successfully ---")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    model = None
    scaler = None

# --- 3. Define the input data model using Pydantic ---
# This class remains unchanged
class TransactionFeatures(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# --- 4. Create App Object ---
# This part remains the same
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="An API to predict fraudulent credit card transactions using a trained LightGBM model.",
    version="1.0.0"
)

# --- 5. API Endpoints ---
@app.get("/")
def read_root():
    """
    Health check endpoint to confirm the API is running.
    """
    return {"message": "Welcome to the Credit Card Fraud Detection API!"}

@app.post("/predict")
def predict_fraud(transaction: TransactionFeatures):
    """
    Receives transaction data, preprocesses it, and returns a fraud prediction.
    """
    if not model or not scaler:
        return {"error": "Model or scaler not loaded. Check server logs."}

    # 1. Parse incoming data into a DataFrame
    transaction_dict = transaction.dict()
    input_df = pd.DataFrame([transaction_dict])

    # 2. Preprocess the DataFrame using the loaded scaler
    input_df[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])

    # 3. Use the loaded model to make a prediction
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)
    fraud_probability = probabilities[0][1]
    
    # --- This is the new logic for the current task ---
    # 4. Format the final response
    
    # Use a simple if-else statement to determine the human-readable label.
    if prediction[0] == 1:
        prediction_label = "Fraud"
        is_fraud_flag = True
    else:
        prediction_label = "Not Fraud"
        is_fraud_flag = False

    # Construct the final JSON response with clear, descriptive keys.
    # We explicitly cast the numpy float to a native Python float for safety.
    return {
        "prediction": prediction_label,
        "is_fraud": is_fraud_flag,
        "confidence_score": float(fraud_probability)
    }
