import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

def data_split(df: pd.DataFrame, column: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the input DataFrame into training and testing sets in ML comparison project.

    Args:
        df (pd.DataFrame): Full dataset including features and target.
        column (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop([column], axis=1)
    y = df[column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test