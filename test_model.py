import numpy as np
from model import train_and_predict, get_accuracy


def test_predictions_not_none():
    """
    Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję.
    """
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."


def test_predictions_length():
    """
    Test 2: Sprawdza, czy długość listy predykcji jest większa od 0
    i czy odpowiada przewidywanej liczbie próbek testowych.
    """
    preds, y_test = train_and_predict()
    assert len(preds) > 0, "Predictions list should not be empty."
    assert len(preds) == len(y_test), "Predictions length must match test set size."


def test_predictions_value_range():
    """
    Test 3: Sprawdza, czy wartości w predykcjach mieszczą się w spodziewanym
    zakresie: dla zbioru Iris mamy 3 klasy (0, 1, 2).
    """
    preds, _ = train_and_predict()
    for p in preds:
        assert p in [0, 1, 2], f"Prediction {p} out of expected range [0, 1, 2]."


def test_model_accuracy():
    """
    Test 4: Sprawdza, czy model osiąga co najmniej 70% dokładności.
    """
    acc = get_accuracy()
    assert acc >= 0.70, f"Model accuracy {acc:.2f} is below the required 70%."
