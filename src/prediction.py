import joblib
import numpy as np
import pandas as pd

def load_model(filepath):
    return joblib.load(filepath)

def predict(model, scaler, input_data):
    # Ensure input_data is a DataFrame with the correct feature names
    columns = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
               'FBS over 120', 'EKG results', 'Max HR', 
               'Exercise angina', 'ST depression', 'Slope of ST', 
               'Number of vessels fluro', 'Thallium']
    input_data_df = pd.DataFrame(input_data, columns=columns)
    input_data_scaled = scaler.transform(input_data_df)
    prediction = model.predict(input_data_scaled)
    return prediction

if __name__ == "__main__":
    model = load_model('../models/mlp_model.pkl')
    scaler = load_model('../models/scaler.pkl')
    
    # Example input data with the correct number of features
    input_data = np.array([[70, 1, 4, 130, 322, 1, 0, 150, 0, 2.4, 2, 3, 3]])  # Adjust to include all required features
    prediction = predict(model, scaler, input_data)
    print(f"Prediction: {prediction}")
