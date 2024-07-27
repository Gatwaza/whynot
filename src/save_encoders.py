import joblib
from sklearn.preprocessing import LabelEncoder

# Create and fit encoders (this is an example, modify according to your data)
encoders = {
    'Sex': LabelEncoder().fit(['male', 'female']),
    'Chest pain type': LabelEncoder().fit(['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic']),
    'Exercise angina': LabelEncoder().fit(['yes', 'no']),
    'Slope of ST': LabelEncoder().fit(['upsloping', 'flat', 'downsloping'])
}

# Save encoders
for feature, encoder in encoders.items():
    joblib.dump(encoder, f'models/encoder_{feature}.pkl')

print("Encoders saved successfully.")
