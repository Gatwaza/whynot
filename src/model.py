import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from preprocessing import preprocess_data

# Load data
train_data = pd.read_csv('/Users/kigali/Desktop/whynot/whynot/data/Heart_Disease_Prediction.csv')

# Preprocess data
X_train, y_train, X_test, y_test = preprocess_data(train_data)

# Features to use
columns = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
           'FBS over 120', 'EKG results', 'Max HR', 
           'Exercise angina', 'ST depression', 'Slope of ST', 
           'Number of vessels fluro', 'Thallium']

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[columns])

# Save scaler
joblib.dump(scaler, '../models/scaler.pkl')

# Train model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, '../models/mlp_model.pkl')

# Evaluate model (optional)
X_test_scaled = scaler.transform(X_test[columns])
accuracy = model.score(X_test_scaled, y_test)
print(f'Model Accuracy: {accuracy}')
