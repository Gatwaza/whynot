import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Dropping the 'index' column and encoding 'Heart Disease' column
    df = df.drop(columns=['index'])
    df['Heart Disease'] = df['Heart Disease'].apply(lambda x: 1 if x == 'Presence' else 0)
    
    # Assuming 'Heart Disease' is the target column
    X = df.drop(columns=['Heart Disease'])
    y = df['Heart Disease']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    # Change the path to your dataset location
    df = load_data('/Users/kigali/Desktop/whynot/whynot/data/Heart_Disease_Prediction.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Ensure the 'models' directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(scaler, 'models/scaler.pkl')
