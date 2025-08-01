"""Module to make prediction from given file."""

import sys
import os

import pandas as pd
import joblib

from src.exception import CustomException
from src.logger import logging


def run_prediction(df_pred: pd.DataFrame) -> None:
    """Function to predict survival of a person.

    Args:
        df_pred (pd.DataFrame): Data to make prediction.
    """

    logging.info("Function to make prediction has started.")

    ROOT_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    try:
        model_path = os.path.join(ARTIFACTS_DIR, "svc_pca_cv.pkl")
        model = joblib.load(model_path)

        y_pred = model.predict(df_pred.drop(["Id"], axis=1))
        submission = pd.DataFrame(
            {"PassengerId": df_pred["Id"], "Survived": y_pred.astype(int)}
        )
        submission_path = os.path.join(ARTIFACTS_DIR, "submission.csv")
        submission.to_csv(submission_path, index=False)

    except Exception as e:
        logging.info("Function to make predicition has encountered a problem.")
        raise CustomException(e, sys) from e
