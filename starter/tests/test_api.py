import json
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_welcome():
    """ Test welcome function """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to Salary Prediction API"}


def test01_get_prediction():
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
    sample = json.dumps(sample)
    r = client.post("/predict/", data=sample)
    assert r.status_code == 200, "The status code of response is not 200"
    assert r.json() == {"predicted_class": 0}, "Inference is not correct"


def test02_get_prediction():
    """ Test get_prediction function """
    sample = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 280464,
        "education": "Masters",
        "education-num": 15,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 10,
        "capital-loss": 0,
        "hours-per-week": 80,
        "native-country": "United-States"
    }
    sample = json.dumps(sample)
    r = client.post("/predict/", data=sample)
    assert r.status_code == 200, "The status code of response is not 200"
    assert r.json() == {"predicted_class": 1}, "Inference is not correct"
