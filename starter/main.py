import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.model import inference

# Instanciate FastAPI
app = FastAPI()

# Load a seriarized model
with open("starter/model/model.joblib", "rb") as f:
    model = joblib.load(f)

# Load a seriarized onehot encoder
with open("starter/model/onehot_encoder.joblib", "rb") as f:
    onehot_encoder = joblib.load(f)


# Declare Data Model for features
class FeatureModel(BaseModel):
    age: int = Field(..., example=25)
    workclass: str = Field(..., example="State-gov")
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
    # Load features from Request body
    features = features.dict()
    df = pd.DataFrame(features, index=[0])

    # Onehot encode categorical features
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country"
    ]
    df_cat = df[cat_features]
    ary_cat_onehot = onehot_encoder.transform(df_cat)
    ary_non_cat = df[df.columns.difference(cat_features)].to_numpy()
    ary = np.hstack([ary_non_cat, ary_cat_onehot])

    # Make a prediction
    predicted_class = inference(model, ary)
    predicted_class = int(predicted_class[0])
    return {"predicted_class": predicted_class}
