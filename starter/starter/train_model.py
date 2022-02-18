"""
Script to train machine learning model.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    performance_on_wholedata,
    performance_on_dataslice
)


def main():
    # Load preprocessed data
    data = pd.read_csv("data/census_cleaned.csv")

    # Split into training and test dataset
    train, test = train_test_split(data, test_size=0.20)

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

    # Process the training data. Create Onehot-encoder and Label Encoder.
    X_train, y_train, encoder, lb, _ = process_data(
        train, categorical_features=cat_features,
        label="salary", training=True
    )

    # Save training dataset and transformers
    pd.DataFrame(X_train).to_csv("data/X_train.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/y_train.csv", index=False)
    with open("model/onehot_encoder.joblib", "wb") as f:
        joblib.dump(encoder, f)

    with open("model/label_encoder.joblib", "wb") as f:
        joblib.dump(lb, f)

    # Proces the test data with the process_data function
    X_test, y_test, _, _, X_test_before = process_data(
        test, categorical_features=cat_features,
        label="salary", training=False,
        encoder=encoder, lb=lb
    )

    # Save test dataset
    pd.DataFrame(X_test).to_csv("data/X_test.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)

    # Train and save a model
    model = train_model(X_train, y_train)
    with open("model/model.joblib", "wb") as f:
        joblib.dump(model, f)

    # Confirm metrics on whole dataset and dataslices
    output_dir = "model_performance"
    performance_on_wholedata(model, X_test, y_test, output_dir=output_dir)
    performance_on_dataslice(model, X_test, y_test, X_test_before,
                             cat_features, output_dir)


if __name__ == "__main__":
    main()
