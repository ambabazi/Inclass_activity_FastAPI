from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle

app = FastAPI()

# Load the trained model once when the app starts
with open('treemodel.pkl', 'rb') as file:
    tmodel = pickle.load(file)


@app.get("/")
def home():
    return {"message": "Prediction API is running. Go to /predict to get a prediction."}


@app.get("/predict")
def predict(row: int = 0):
    """
    Returns a prediction from the trained model.
    Optional query param 'row' picks which row of x_test.csv to use (default is 0).
    Example: /predict?row=3
    """
    data = pd.read_csv('x_test.csv', index_col=False)
    data = data.iloc[:, 1:]
    x = np.array(data)

    # simple input validation
    if row < 0 or row >= len(x):
        return {"error": f"row must be between 0 and {len(x) - 1}"}

    prediction = tmodel.predict(x[row].reshape(1, -1))[0]
    return {"row_used": row, "prediction": prediction}