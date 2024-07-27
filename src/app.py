from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Determine the base path for models
base_path = '/app/src/models' if os.path.exists('/app') else 'models'

# Load the model, scaler, and encoders
model = joblib.load(os.path.join(base_path, 'mlp_model.pkl'))
scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))

encoders = {
    'Sex': joblib.load(os.path.join(base_path, 'encoder_Sex.pkl')),
    'Chest pain type': joblib.load(os.path.join(base_path, 'encoder_Chest pain type.pkl')),
    'Exercise angina': joblib.load(os.path.join(base_path, 'encoder_Exercise angina.pkl')),
    'Slope of ST': joblib.load(os.path.join(base_path, 'encoder_Slope of ST.pkl'))
}

def preprocess_input(data):
    data['Sex'] = encoders['Sex'].transform([data['Sex']])[0]
    data['Chest pain type'] = encoders['Chest pain type'].transform([data['Chest pain type']])[0]
    data['Exercise angina'] = encoders['Exercise angina'].transform([data['Exercise angina']])[0]
    data['Slope of ST'] = encoders['Slope of ST'].transform([data['Slope of ST']])[0]
    return data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        data = preprocess_input(data)
        input_data = np.array([[
            data['Age'], data['Sex'], data['Chest pain type'], data['BP'], 
            data['Cholesterol'], data['FBS over 120'], data['EKG results'], 
            data['Max HR'], data['Exercise angina'], data['ST depression'], 
            data['Slope of ST'], data['Number of vessels fluro'], data['Thallium']
        ]])

        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)

        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)
