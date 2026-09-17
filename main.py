from fastapi import FastAPI
from backend.predictor import predict_mastitis_risk


app = FastAPI(
    title="Mastitis Early Warning API",
    description="AI-based predictive modelling for early mastitis risk forecasting",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "message": "Mastitis AI API is running"
    }


@app.post("/predict")
def predict(cow_data: dict):

    result = predict_mastitis_risk(cow_data)

    return result