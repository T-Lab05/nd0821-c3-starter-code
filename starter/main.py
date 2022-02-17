import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from starter.ml.model import inference

# Instanciate FastAPI
app = FastAPI()

# Load a seriarized model
with open("model/model.joblib","rb") as f:
    model = joblib.load(f)

# Declare Data Model for features
class FeatureModel(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education-num: int
    marital-status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital-gain: int
    capital-loss: int
    hours-per-week: int
    native-country: str


# Get method to Route. Show welcome a message.
@app.get("/")
async def welcome():
    return {"message": "Welcome to Salary Prediction API"}

# Post method to predict path. Return a prediction.
@app.post("/predict/")
async def get_prediction(features: FeatureModel):
    # Prepare features data as numpy.ndarray
    # Be careful for the order of features so that it matched
    # that of the training dataset.
    features = features.dict()
    
    age = features["age"]
    workclass = features["workclass"]
    fnlgt = features["fnlgt"]
    education = features["education"]
    education-num = features["education-num"]
    marital-status = features["marital-status"]
    occupation = features["occupation"]
    relationship = features["relationship"]
    race = features["race"]
    sex = features["sex"]
    capital-gain = features["capital-gain"]
    capital-loss = features["capital-loss"]
    hours-per-week = features["hours-per-week"]
    native-country = features["native-country"]
    
    features = np.array([age, workclass, fnlgt, education,education-num,
        marital_status, occupation, relationship, race, sex, capital-gain,
        capicatl-loss, hours-per-week, native-country
    ])

    predicted_class = inference(model, features)    
    return {"predicted_class": predicted_class}