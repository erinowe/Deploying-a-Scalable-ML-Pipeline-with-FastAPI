import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, fbeta_score

# TODO: implement the first test. Change the function name and input as needed
def test_model_type():
    """
    Test that the model being used is AdaBoostClassifier
    """
    model = AdaBoostClassifier(random_state=0)
    assert isinstance(model, AdaBoostClassifier)


def test_model_prediction_shape():
    """
    Test that model predictions return the expected shape
    """
    # Simple dummy dataset
    X = pd.DataFrame({
        "feature1": [0, 1, 0, 1],
        "feature2": [1, 1, 0, 0]
    })
    y = np.array([0, 1, 0, 1])

    model = AdaBoostClassifier(random_state=0)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)


# TODO: implement the third test. Change the function name and input as needed

def test_compute_metrics_range():
    """
    Test that accuracy and fbeta score are within valid bounds
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    acc = accuracy_score(y_true, y_pred)
    fscore = fbeta_score(y_true, y_pred, beta=0.5)

    assert 0.0 <= acc <= 1.0
    assert 0.0 <= fscore <= 1.0
