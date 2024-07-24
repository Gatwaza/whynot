import joblib
import numpy as np

def load_model(filepath):
    return joblib.load(filepath)

def predict(model, scaler, input_data):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction

if __name__ == "__main__":
    model = load_model('../models/mlp_model.pkl')
    scaler = load_model('../models/scaler.pkl')
    input_data = np.array([[70, 1, 4, 130, 322, 0, 2.4, 2, 3, 3]]).reshape(1, -1)  # Example input data
    prediction = predict(model, scaler, input_data)
    print(f"Prediction: {prediction}")
