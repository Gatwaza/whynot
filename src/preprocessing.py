import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Assuming 'Heart Disease' is the target column and other columns are features
    target_column = 'Heart Disease'
    feature_columns = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
                       'FBS over 120', 'EKG results', 'Max HR', 
                       'Exercise angina', 'ST depression', 'Slope of ST', 
                       'Number of vessels fluro', 'Thallium']

    # Encode target variable
    df[target_column] = df[target_column].map({'Presence': 1, 'Absence': 0})

    X = df[feature_columns]
    y = df[target_column]

    # Split data into training and testing sets (optional)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test
