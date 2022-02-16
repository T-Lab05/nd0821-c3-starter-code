import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from starter.starter.ml.data import (
    process_data
)
from starter.starter.ml.model import (
    train_model,
    compute_model_metrics,
    inference
)


@pytest.fixture()
def prepare_data():
    """ Fixture to split data """
    # Load data
    data = pd.read_csv("starter/data/census_cleaned.csv")
    target_label = "salary"
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
        
    # # Split feature and label
    # y = df.pop(target_label)
    # X = df

    # Split train and test set
    train, test = train_test_split(
        data, test_size=0.3, random_state=42, shuffle=True, 
        stratify=data[target_label]
    )

    # Encoding features and a label on the train dataset
    # Also, obtain transformer of Onehot encoder and label encoder
    X_train, y_train, encoder, lb = process_data(
        train, cat_features, target_label, training=True
    )
    
    # Split feature and label on the test dataset
    y_test = test.pop(target_label)
    y_test = lb.transform(y_test).reshape(-1,)

    test_cat_onehot = encoder.transform(test[cat_features])
    test.drop(cat_features, axis=1, inplace=True)
    test = test.to_numpy()
    X_test = np.hstack([test, test_cat_onehot])

    return X_train, y_train, X_test, y_test


def test_train_model(prepare_data):
    """ 
    Test if the function returns the subclass of 
    scikit-learn's Base Estimator.
    """
    X_train, y_train, _, _ = prepare_data
    model = train_model(X_train, y_train)
    assert isinstance(model, BaseEstimator), (
        "The output of train_model is not a subclass of " \
        "scikit-learn's BaseEstimator."
    )


def test_compute_model_metrics(prepare_data):
    """ 
    Test if the function returns proper metircs of precision, recall, fbeta 
    in terms of data types and ranges.
    """
    X_train, y_train, X_test, y_test = prepare_data
    model = train_model(X_train, y_train)
    preds = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float), (
        "Precision (first return value) is not float"
    )
    assert isinstance(recall, float), (
        "Recall (second return value) is not float"
    )
    assert isinstance(fbeta, float), (
        "Fbeta (third return value) is not float"
    )
    assert 0 <= precision and precision <= 1, (
        "Precision is not between 0 and 1 "
    )
    assert 0 <= recall and recall <= 1, (
        "Recall is not between 0 and 1 "
    )
    assert 0 <= fbeta and fbeta <= 1, (
        "Fbeta is not between 0 and 1 "
    )


def test_inference(prepare_data):
    """ 
    Test if the function returns a proper value 
    in terms of data type and array length
    """
    X_train, y_train, X_test, _ = prepare_data
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray), (
        "Return value is not an instance of numpy.ndarray"
    )
    assert preds.shape[0] == X_test.shape[0], (
        "Length of returned array doesn't match" \
        "that of the input feature array"
    )   