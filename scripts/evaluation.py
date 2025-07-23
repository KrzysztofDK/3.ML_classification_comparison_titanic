import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def evaluate_model(model: BaseEstimator, X_test: pd.DataFrame, y_test: Union[pd.Series, np.ndarray], name: str) -> None:
    """
    Function to evaluate Accuracy, Precision, Recall and Roc Auc in given model.

    Args:
        model: The model argument should be an object of a class that inherits from sklearn models.
        X_test: DataFrame.
        y_test: DataFrame as Series or ndarray.
        name: String object. Name of the model.

    Raises:
        FileNotFoundError: Invalid model.

    Returns:
        None: The function returns nothing.
    """
    try:
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        ra = roc_auc_score(y_test, y_pred)
        
        row = pd.DataFrame([{
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'Roc Auc': ra
        }])

        excel_path = Path("results") / "metrics.xlsx"
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        
        if excel_path.exists():
            existing = pd.read_excel(excel_path)
            df_all = pd.concat([existing, row], ignore_index=True)
        else:
            df_all = row
            
        df_all.to_excel(excel_path, index=False)
        
    except FileNotFoundError:
        print('Invalid model.')