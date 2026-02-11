import pytest
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd
from ml.data import process_data, apply_label
from ml.model import train_model, inference, compute_model_metrics

Cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
@pytest.fixture(scope="module")
def data():
    data_path = os.path.join("data", "census.csv")
    df = pd.read_csv(data_path)
    return df.sample(n=500, random_state=42)

@pytest.fixture(scope="module")
def processed(data):
    X, y, encoder, lb = process_data(
        data,
        catergorical_features = Cat_features,
        label = "salary",
        training = True
    )
return X,y, encoder, lb

def test_apply_label():
    #apply_label should return one of the expected salary labels
    assert apply_label (0) in ["<=50K", "<=50k."]
    assert apply_label (1) in [">50k", ">50k."]

def test_train_model():
    #model should train and inference should return predictions of correct length 
    X, y, encoder, lb = processed
    model = train_model(X,y)
    assert isinstance(model,LogisticRegression)

def test_inference_output():
    #precision/recall/F1 should be between 0 and 1 
    X, y, _, _ = processed
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) ==len(y)

def test_compute_model
    X,y, _, _ = processed
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall,fbeta = compute_model_metrics(y, preds)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <=recall <= 1.0
    assert 0.0 <=fbeta <= 1.0

