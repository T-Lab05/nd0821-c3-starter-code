import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from starter.main import app

client = TestClient(app)

def test_welcome():
    """ Test welcome function """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to Salary Prediction API"}


def test_get_prediction():
    """ Test get_prediction function """
    sample = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    r = client.post("/predict/")
    assert r.status_code == 200, "The status code of response is not 200"
    # assert r.json() == {"predicted_class": 0}, ""
    
