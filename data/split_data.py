import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('Heart_Disease_Prediction.csv')

# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the datasets to the respective directories
train_data.to_csv('../data/train.csv', index=False)
test_data.to_csv('../data/test.csv', index=False)
