import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Each hyper parameter takes a default value of sklearn
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklean.emsemble.RandomForest
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def performance_on_wholedata(model, X_test, y_test, output_dir):
    """ Check prediction performance on a whole test dataset.

    Inputs
    ------
    model:
        Trained machine learning model.
    X_test: numpy.ndarray
        Feature data used for prediction.
    y_test: numpy.ndarray
        Label data used for evaluation.
    output_dir: str
        Path of directory to output a text file.

    Returns
    -------
        None
        (Write a csv file on disk)

    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    basefile = "performance_on_wholedata.csv"
    filepath = os.path.join(output_dir, basefile)

    # Make predictions
    preds = inference(model, X_test)
    preds = pd.Series(preds, name="pred")
    # Get metrics as pandas Series
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    ser = pd.Series({
        "precision": precision,
        "recall": recall,
        "fbeta": fbeta
    }, name="Metrics")

    # Output as CSV file
    ser.to_csv(filepath)

    return None


def performance_on_dataslice(
        model, X_test, y_test, X_test_raw, slice_columns,
        output_dir):
    """ Check prediction performance on arbitrary groups.

    Inputs
    ------
    model:
        Trained machine learning model.
    X_test: numpy.ndarray
        Feature data used for prediction.
    y_test: numpy.ndarray
        Label data used for evaluation.
    X_test_raw: pandas.DataFrame
        Feature data before preprocess, which contains original
        categorical columns.
    slice_columns : list[str]
        Column names on which data is sliced into groups.
    output_dir: str
        Path of directory to output a text file.

    Returns
    -------
        None
        (Write a csv and png file on disk)

    """


    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    basefile = "performance_on_dataslice.csv"
    filepath = os.path.join(output_dir, basefile)

    # Make predictions
    preds = inference(model, X_test)
    preds = pd.Series(preds, name="pred")
    X_test_raw = X_test_raw.reset_index() # Reset indexes before concat
    X = pd.concat([X_test_raw, preds], axis=1)

    df = pd.DataFrame(
        columns=["columns", "value", "precision", "recall", "fbeta"]
    )

    report_fpath = os.path.join(output_dir, "slice_output.txt")

    # Loop slice columns and their values
    with open(report_fpath, "w+") as f:
        for c in slice_columns:
            for value in X[c].unique():
                # Make a sliced group
                is_target = X[c] == value
                y_slice = y_test[is_target]
                preds_slice = preds[is_target]
                # Get metrics on the group
                precision, recall, fbeta = compute_model_metrics(
                    y_slice, preds_slice
                )
                # Insert data into DataFrame
                ser = pd.Series({
                    "columns": c,
                    "value": value,
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta
                })
                df = pd.concat([df, ser.to_frame().T], ignore_index=True)
                # Write to a text file
                report = f"{c} - {value} \n\t precision: {precision}" + \
                         f"\n\t recall   : {recall}" + \
                         f"\n\t fbeta    : {fbeta} \n"
                f.writelines(report)

    # Output as CSV file
    df.to_csv(filepath)

    # Output plots as PNG file
    for c in df["columns"].unique():
        _tmp = df[df["columns"] == c]
        fig, ax = plt.subplots()
        for met in ["precision", "recall", "fbeta"]:
            sns.lineplot(x="value", y=met, data=_tmp, ax=ax, label=met)
        ax.set_title(f"Metrics on the column: {c} ")
        plt.draw()
        xticklabel = ax.get_xticklabels()
        ax.set_xticklabels(xticklabel, rotation=85, ha="right")
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.legend()
        filename = os.path.join(output_dir, f"{c}.png")
        plt.savefig(filename)

    return None
