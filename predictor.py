import joblib
import pandas as pd
from pathlib import Path

print("PREDICTOR FILE STARTED")

# Find the project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load our trained model package
MODEL_PATH = BASE_DIR / "models" / "mastitis_model_v21.pkl"

print("Loading model...")
model_package = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

model = model_package["model"]
scaler = model_package["scaler"]
features = model_package["features"]


def predict_mastitis_risk(cow_data):
    """
    Takes cow data and returns mastitis risk.
    """

    # Convert input dictionary into a DataFrame
    input_df = pd.DataFrame([cow_data])

    # Make sure features are in exactly the order used during training
    input_df = input_df[features]

    # Scale using the scaler saved during training
    input_scaled = scaler.transform(input_df)

    # Get probability of mastitis risk
    risk_probability = model.predict_proba(input_scaled)[0][1]

    # Convert probability to percentage
    risk_percentage = risk_probability * 100

    # Simple prototype risk categories
    if risk_percentage >= 50:
        risk_level = "HIGH"
    elif risk_percentage >= 20:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_probability": round(risk_probability, 4),
        "risk_percentage": round(risk_percentage, 2),
        "risk_level": risk_level
    }

if __name__ == "__main__":
    print("Starting prediction test...")

    test_cow = {
        "age": 5,
        "lactation_num": 3,
        "days_since_calving": 100,
        "prior_mastitis": 0,
        "milk_yield": 15,
        "body_temp": 39.0,
        "milk_conductivity": 8.5,
        "activity_score": 70,

        "milk_yield_7d_mean": 17,
        "milk_yield_change_7d": -2,

        "body_temp_7d_mean": 38.5,
        "body_temp_change_7d": 0.5,

        "milk_conductivity_7d_mean": 7.5,
        "milk_conductivity_change_7d": 1.0,

        "activity_score_7d_mean": 80,
        "activity_score_change_7d": -10,

        "milk_yield_slope_7d": -0.3,
        "body_temp_slope_7d": 0.05,
        "milk_conductivity_slope_7d": 0.15,
        "activity_score_slope_7d": -1.5
    }

    print("Cow data created.")

    result = predict_mastitis_risk(test_cow)

    print("Prediction completed!")
    print(result)