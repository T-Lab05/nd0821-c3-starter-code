"""
Send a sample request to deployed model on Heroku
"""

import requests
import json


def main():

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

    url = "https://udacity-mldevop-part3-api.herokuapp.com/predict"
    r = requests.post(url, sample)
    print(f"Response status code: {r.status_code}")
    print(f"Response JSON: {r.json()}")


if __name__ == "__main__":
    main()
