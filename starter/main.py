import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.model import inference

# Instanciate FastAPI
app = FastAPI()

# Load a seriarized model
with open("model/model.joblib", "rb") as f:
    model = joblib.load(f)


# Declare Data Model for features
class FeatureModel(BaseModel):
    age: int = Field(..., example=25)
    workclass: str = Field(..., example="Never-married")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(...,
                                alias="marital-status",
                                example="Divorced")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=0)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(...,
                                alias="native-country",
                                example="United-States")


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
    education_num = features["education_num"]
    marital_status = features["marital_status"]
    occupation = features["occupation"]
    relationship = features["relationship"]
    race = features["race"]
    sex = features["sex"]
    capital_gain = features["capital_gain"]
    capital_loss = features["capital_loss"]
    hours_per_week = features["hours_per_week"]
    native_country = features["native_country"]

    features = np.array([
        age, workclass, fnlgt, education, education_num,
        marital_status, occupation, relationship, race, sex, capital_gain,
        capital_loss, hours_per_week, native_country
    ])

    predicted_class = inference(model, features)
    return {"predicted_class": predicted_class}
