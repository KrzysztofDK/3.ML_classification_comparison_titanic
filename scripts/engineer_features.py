import pandas as pd
from typing import Tuple

def agumentation_with_columns(X_train: pd.DataFrame, X_test: pd.DataFrame, df_pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to agument DataFrames with new columns in ML_comparison project.
    """
    for df in [X_test, X_train, df_pred]:
        df['FamilySize'] = 1 + df['Siblings_Spouses'] + df['Parents_Childs']
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    return X_train, X_test, df_pred