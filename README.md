# Heart Disease Prediction

## Project Description
This project predicts whether a patient has heart disease using a multi-layer perceptron (MLP) model. The model is trained on heart disease data and deployed using Flask.

## Project Structure
whynot/
│
├── README.md
│
├── notebook/
│   ├── whynot.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
│
├── data/
│   ├── train/
│   └── test/
│
└── models/
    ├── mlp_model.pkl
    ├── scaler.pkl

## Setup Instructions

1. Clone the repository:
   ```sh
   git clone https://github.com/Gatwaza/whynot.git
   cd whynot

python3 -m venv whynot-env

source whynot-env/bin/activate

pip install -r requirements.txt

python3 src/preprocessing.py

python3 src/model.py

python3 app.py

## Testing purpose
<img width="928" alt="Screenshot 2024-07-29 at 09 25 12" src="https://github.com/user-attachments/assets/4762b6dd-9d99-4caf-a7e1-bd53944d628d">
