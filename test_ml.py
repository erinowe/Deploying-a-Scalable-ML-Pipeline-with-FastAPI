import pytest
import os
import pandas as pandas
from ml.data import process_data, apply_label
from ml.model import train_model, inference, compute_model_metrics

project_path = "."
data_path = os.path.join(project_path, "data","densus.csv")

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
    df = pd.read_csv(data_path)
    return df

@pytest.fixture(scope="module")
def processed(data):
    x, y, encoder, lb = process_data(
        data,
        catergorical_features = Cat_features,
        label = "salary",
        training = True
    )
return x,y, encoder, lb

def test_one():
    #apply_label should return one of the expected salary labels
    assert apply_label (0) in ["<=50K", "<=50k."]
    assert apply_label (1) in [">50k", ">50k."]



def test_two():
    #model should train and inference should return predictions of correct length 
    x, y, encoder, lb = processed
    model = train_model(x,y)
    preds = inference (model, x)
    assert len(preds) == len(y)


def test_three():
    #precision/recall/F1 should be between 0 and 1 
    x, y, _, _ = processed
    model = train_model(x, y)
    preds = inference(model, x)
    precision, recall,fbeta = compute_model_metrics(y, preds)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <=recall <= 1.0
    assert 0.0 <=fbeta <= 1.0