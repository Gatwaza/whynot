from sklearn.neural_network import MLPClassifier
import joblib

def train_model(X_train, y_train):
    # Increase max_iter and adjust learning_rate_init
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, learning_rate_init=0.001, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    joblib.dump(model, filepath)

if __name__ == "__main__":
    from preprocessing import load_data, preprocess_data
    # Change the path to your dataset location
    df = load_data('/Users/kigali/Desktop/whynot/whynot/data/Heart_Disease_Prediction.csv')
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    model = train_model(X_train, y_train)
    
    # Ensure the 'models' directory exists
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
    
    save_model(model, 'models/mlp_model.pkl')
