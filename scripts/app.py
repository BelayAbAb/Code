from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from pydantic import BaseModel
import numpy as np

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Initialize FastAPI
app = FastAPI()

# Define the request body model
class CustomerFeatures(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add all required features here according to your dataset

# Define the prediction response model
class PredictionResponse(BaseModel):
    risk_probability: float
    credit_score: int  # Assume you have a credit score calculation
    recommended_loan_params: dict  # Customize as needed

@app.post('/predict', response_model=PredictionResponse)
async def predict(customer_features: CustomerFeatures):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([customer_features.dict()])

    
    # Make prediction
    risk_probability = model.predict_proba(input_data)[:, 1][0]
    prediction = model.predict(input_data)[0]

    # Simulate a credit score and recommended loan parameters
    credit_score = int(risk_probability * 100)  # Example conversion
    recommended_loan_params = {
        "amount": 10000 * (1 - risk_probability),  # Example calculation
        "interest_rate": 5.0 + (risk_probability * 10)  # Example calculation
    }

    return PredictionResponse(
        risk_probability=risk_probability,
        credit_score=credit_score,
        recommended_loan_params=recommended_loan_params
    )

# Run the API
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
