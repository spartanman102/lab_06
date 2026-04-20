import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.datasets import load_iris
from model import train_and_predict, get_accuracy

app = FastAPI()

iris = load_iris()


class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def read_root():
    owner = os.getenv("USER", "Student")
    return {
        "message": f"API działa! Użytkownikiem jest: {owner}",
        "target_names": ["setosa", "versicolor", "virginica"],
    }


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        features = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
        ]])
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X, y = iris.data, iris.target
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0].tolist()

        return {
            "prediction": prediction,
            "class_name": iris.target_names[prediction],
            "probabilities": probabilities,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd predykcji: {str(e)}")


@app.get("/info")
def info():
    return {
        "model_type": "LogisticRegression",
        "dataset": "Iris",
        "number_of_features": iris.data.shape[1],
        "classes": iris.target_names.tolist(),
        "accuracy": round(get_accuracy(), 4),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
