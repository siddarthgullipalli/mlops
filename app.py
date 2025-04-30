# mlops/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model.pkl")
app = FastAPI()

class IrisInput(BaseModel):
    data: list

@app.post("/predict")
def predict(input: IrisInput):
    prediction = model.predict(np.array(input.data).reshape(1, -1))
    return {"prediction": prediction.tolist()}
