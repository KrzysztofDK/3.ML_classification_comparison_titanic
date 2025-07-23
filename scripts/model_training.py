import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from typing import Union
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, cross_validate
from .evaluation import evaluate_model

def model_training_saving_evaluating(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models: dict[str, Union[BaseEstimator, tuple[BaseEstimator, dict]]],
    use_cv: bool = False,
    scoring_metric: str = 'recall'
) -> None:
    """
    Trains, optionally cross-validates, saves, and evaluates models.
    
    models dict:
        - 'Model Name': model
        - 'Model Name': (pipeline, param_grid)

    If use_cv=True:
        Performs cross_validate (cv=5) and picks best estimator based on scoring_metric.
    """
    model_dir = Path(__file__).resolve().parent / ".." / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model_info in models.items():
        print(f'Training model: {name}')
        
        if isinstance(model_info, tuple):
            model, param_grid = model_info
            grid = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring=scoring_metric)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        elif use_cv:
            model = model_info
            cv_results = cross_validate(
                model,
                X_train,
                y_train,
                cv=5,
                scoring=['accuracy', 'precision', 'recall', 'roc_auc'],
                return_train_score=False,
                return_estimator=True
            )
            recalls = cv_results['test_recall']
            estimators = cv_results['estimator']
            
            best_idx = np.argmax(recalls)
            best_recall = recalls[best_idx]
            best_model = estimators[best_idx]
        else:
            best_model = model_info
            best_model.fit(X_train, y_train)

        model_filename = os.path.join(model_dir, f"{name.lower().replace(' ', '_')}.pkl")
        try:
            joblib.dump(best_model, model_filename)
            print(f'Model saved to {model_filename}.')
        except Exception as e:
            print(f'Error during savingmodel {name}: {e}.')
            
        evaluate_model(best_model, X_test, y_test, name)