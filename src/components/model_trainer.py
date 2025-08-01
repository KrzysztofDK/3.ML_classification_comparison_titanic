"""Module to create and train models."""

import os
import sys
from typing import Union
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, cross_validate
from xgboost import XGBClassifier

from src.logger import logging
from src.exception import CustomException
from src.components.data_evaluation import evaluate_model


def create_base_models() -> dict[str, BaseEstimator]:
    """
    Project specific function to create base models dictionary.
    """

    logging.info("Function to create base models has started.")

    try:
        return {
            "Logistic Regression": make_pipeline(
                StandardScaler(), LogisticRegression(random_state=42, max_iter=1000)
            ),
            "Random Forest": RandomForestClassifier(random_state=42),
            "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
            "XGB": XGBClassifier(random_state=42),
            "SVC": make_pipeline(
                StandardScaler(), SVC(probability=True, random_state=42)
            ),
        }

    except Exception as e:
        logging.info("Function to create base models has encountered a problem.")
        raise CustomException(e, sys) from e


def create_feature_selected_models(
    n_features_to_select: int = 8,
) -> dict[str, BaseEstimator]:
    """
    Project specific function to create models with feature selection/extraction dictionary.
    """

    logging.info("Function to create feature selected models has started.")

    try:
        lr_rfe = RFE(
            estimator=LogisticRegression(), n_features_to_select=n_features_to_select
        )
        rf_sfm = SelectFromModel(
            estimator=RandomForestClassifier(random_state=42),
            max_features=n_features_to_select,
        )
        knn_pca = PCA(n_components=n_features_to_select)
        xgb_sfm = SelectFromModel(
            estimator=XGBClassifier(random_state=42), max_features=n_features_to_select
        )
        svc_pca = PCA(n_components=n_features_to_select)

        return {
            "Logistic Regression RFE": make_pipeline(
                StandardScaler(),
                lr_rfe,
                LogisticRegression(random_state=42, max_iter=1000),
            ),
            "Random Forest SFM": make_pipeline(
                rf_sfm, RandomForestClassifier(random_state=42)
            ),
            "KNN PCA": make_pipeline(StandardScaler(), knn_pca, KNeighborsClassifier()),
            "XGB SFM": make_pipeline(xgb_sfm, XGBClassifier(random_state=42)),
            "SVC PCA": make_pipeline(
                StandardScaler(), svc_pca, SVC(probability=True, random_state=42)
            ),
        }

    except Exception as e:
        logging.info(
            "Function to create feature selected models has encountered a problem."
        )
        raise CustomException(e, sys) from e


def get_gridsearch_models_and_params(
    n_features_to_select: int = 8,
) -> dict[str, tuple[BaseEstimator, dict]]:
    """
    Function to create models and parameters to use in hyperparameter tuning in ML_comparison project.
    """

    logging.info("Function to create gridsearch models and params has started.")

    try:
        pipe_lr = make_pipeline(
            StandardScaler(),
            RFE(estimator=LogisticRegression()),
            LogisticRegression(random_state=42, max_iter=1000),
        )
        pipe_rf = make_pipeline(
            SelectFromModel(estimator=RandomForestClassifier(random_state=42)),
            RandomForestClassifier(random_state=42),
        )
        pipe_knn = make_pipeline(StandardScaler(), PCA(), KNeighborsClassifier())
        pipe_xgb = make_pipeline(
            SelectFromModel(estimator=XGBClassifier(random_state=42)),
            XGBClassifier(random_state=42),
        )
        pipe_svc = make_pipeline(
            StandardScaler(), PCA(), SVC(probability=True, random_state=42)
        )

        param_grid_lr = {
            "rfe__n_features_to_select": [6, 8, 11, 16, 19],
            "logisticregression__C": [0.1, 1.0, 10.0],
        }

        param_grid_rf = {
            "selectfrommodel__max_features": [6, 8, 11, 16, 19],
            "randomforestclassifier__n_estimators": [50, 100, 200],
            "randomforestclassifier__max_depth": [None, 5, 10, 20],
        }

        param_grid_knn = {
            "pca__n_components": [6, 8, 11, 16, 19],
            "kneighborsclassifier__n_neighbors": [3, 5, 7],
            "kneighborsclassifier__weights": ["uniform", "distance"],
        }

        param_grid_xgb = {
            "selectfrommodel__max_features": [6, 8, 11, 16, 19],
            "xgbclassifier__n_estimators": [50, 100],
            "xgbclassifier__max_depth": [3, 6],
            "xgbclassifier__learning_rate": [0.01, 0.1, 0.3],
        }

        param_grid_svc = {
            "pca__n_components": [6, 8, 11, 16, 19],
            "svc__C": [0.1, 1.0, 10.0],
            "svc__gamma": ["scale", "auto"],
            "svc__kernel": ["rbf", "linear"],
        }

        models_params = {
            "Logistic Regression RFE CV": (pipe_lr, param_grid_lr),
            "Random Forest SFM CV": (pipe_rf, param_grid_rf),
            "KNN PCA CV": (pipe_knn, param_grid_knn),
            "XGB SFM CV": (pipe_xgb, param_grid_xgb),
            "SVC PCA CV": (pipe_svc, param_grid_svc),
        }
        return models_params

    except Exception as e:
        logging.info(
            "Function to create gridsearch models and params has encountered a problem."
        )
        raise CustomException(e, sys) from e


def model_training_saving_evaluating(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models: dict[str, Union[BaseEstimator, tuple[BaseEstimator, dict]]],
    use_cv: bool = False,
    scoring_metric: str = "recall",
) -> None:
    """
    Trains, optionally cross-validates, saves, and evaluates models.

    models dict:
        - 'Model Name': model
        - 'Model Name': (pipeline, param_grid)

    If use_cv=True:
        Performs cross_validate (cv=5) and picks best estimator based on scoring_metric.
    """

    logging.info("Function to train models has started.")

    ROOT_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    try:
        for name, model_info in models.items():
            print(f"Training model: {name}")

            if isinstance(model_info, tuple):
                model, param_grid = model_info
                grid = GridSearchCV(
                    model,
                    param_grid=param_grid,
                    cv=5,
                    n_jobs=-1,
                    verbose=0,
                    scoring=scoring_metric,
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_

            elif use_cv:
                model = model_info
                cv_results = cross_validate(
                    model,
                    X_train,
                    y_train,
                    cv=5,
                    scoring=["accuracy", "precision", "recall", "roc_auc"],
                    return_train_score=False,
                    return_estimator=True,
                )
                recalls = cv_results["test_recall"]
                estimators = cv_results["estimator"]

                best_idx = np.argmax(recalls)
                best_recall = recalls[best_idx]
                best_model = estimators[best_idx]
            else:
                best_model = model_info
                best_model.fit(X_train, y_train)

            model_filename = os.path.join(
                ARTIFACTS_DIR, f"{name.lower().replace(' ', '_')}.pkl"
            )

            joblib.dump(best_model, model_filename)

            evaluate_model(best_model, X_test, y_test, name)

    except Exception as e:
        logging.info("Function to train models has encountered a problem.")
        raise CustomException(e, sys) from e
