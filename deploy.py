from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load('model_weights.pkl')
scaler = joblib.load('scaler_weights.pkl')

app = FastAPI(title="House Price Prediction API")

class HouseData(BaseModel):
    features: list  

@app.post("/predict")
def predict_price(data: HouseData):
    input_data = np.array(data.features).reshape(1, -1)
    
    scaled_data = scaler.transform(input_data)
    
    prediction = model.predict(scaled_data)
    
    final_price = float(prediction[0][0])
    
    return {
        "predicted_price": round(final_price, 2),
        "currency": "USD" 
    }

@app.get("/")
def home():
    return {"message": "Regression API is online. Go to /docs to test house price prediction."}