"""Modul to transform data."""

import sys
import os
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def handling_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Function to fill and visualize nulls in 2.ML_regression_comparison_housing project.

    Args:
        df (pd.DataFrame): DataFrame to check and fill nulls.

    Returns:
        pd.DataFrame: Fixed DataFrame.
    """

    logging.info("Function to check and fill nulls has started.")

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("Nulls in dataframe")
    plt.savefig(os.path.join("images", "isnull.png"), dpi=300, bbox_inches="tight")
    plt.close()

    if (df.isnull().sum() == 0).all():
        print("There are no nulls.")
        logging.info("Function to check and fill nulls has ended.")
        return df
    else:
        print("There are some null values.")
        return df


def clean_split_and_save_as_csv_data(
    df: pd.DataFrame, df_predict: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function to clean, split and save as csv data in 3.ML_classification_comparison_titanic project.

    Args:
        df (pd.DataFrame): DataFrame to clean.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Splited, cleaned DataFrame for X/y and train/test.
    """

    logging.info("Function to clean and split data has started.")

    df = df.copy()
    df_predict = df_predict.copy()

    num_duplicates = df.duplicated(keep="first").sum()
    print(f"Number of duplicated data: {num_duplicates}")

    df = handling_nulls(df)

    try:
        X = df.drop(["Survived"], axis=1)
        y = df["Survived"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        for dataframe in [X_test, X_train, df_predict]:
            dataframe.rename(
                columns={
                    "PassengerId": "Id",
                    "Pclass": "Class",
                    "SibSp": "Siblings_Spouses",
                    "Parch": "Parents_Childs",
                },
                inplace=True,
            )

        names = ["X_train", "X_test", "y_train", "y_test", "df_predict"]
        dataframes = [X_train, X_test, y_train, y_test, df_predict]

        for name, dataframe in zip(names, dataframes):
            data_path = os.path.join(ARTIFACTS_DIR, f"{name}_cleaned.csv")
            dataframe.to_csv(data_path, header=True, index=False)

        return X_train, X_test, y_train, y_test, df_predict

    except Exception as e:
        logging.info("Function to clean and split data has encountered a problem.")
        raise CustomException(e, sys) from e


def agumentation_with_columns(
    X_train: pd.DataFrame, X_test: pd.DataFrame, df_pred: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function to agument DataFrames with new columns in ML_comparison project.

    Args:
        X_train (pd.DataFrame): Train dataset.
        X_test (pd.DataFrame): Test dataset.
        df_pred (pd.DataFrame): Dataset to predict.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames with new columns.
    """
    for df in [X_train, X_test, df_pred]:
        df["FamilySize"] = 1 + df["Siblings_Spouses"] + df["Parents_Childs"]
        df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

    return X_train, X_test, df_pred


def transform_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame, df_pred: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function to transform data speciffivly for 3.ML_classification_comparison_titanic project.

    Args:
        X_train (pd.DataFrame): Train dataset.
        X_test (pd.DataFrame): Test dataset.
        df_pred (pd.DataFrame): Dataset to predict.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Transformed data.
    """

    logging.info("Function to transform data has started.")

    try:
        age_imputer = SimpleImputer(strategy="median")
        X_train["Age"] = age_imputer.fit_transform(X_train[["Age"]])
        X_test["Age"] = age_imputer.transform(X_test[["Age"]])
        df_pred["Age"] = age_imputer.transform(df_pred[["Age"]])

        X_train["HasCabin"] = X_train["Cabin"].notna().astype(int)
        X_test["HasCabin"] = X_test["Cabin"].notna().astype(int)
        df_pred["HasCabin"] = df_pred["Cabin"].notna().astype(int)

        X_train, X_test, df_pred = agumentation_with_columns(X_train, X_test, df_pred)

        for df in [X_test, X_train]:
            df.drop(columns=["Id", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

        for df in [df_pred]:
            df.drop(columns=["Name", "Ticket", "Cabin"], axis=1, inplace=True)

        for df in [X_test, X_train, df_pred]:
            columns = ["Age", "HasCabin"]
            for col in columns:
                df[col] = df[col].astype("int64")

        for df in [X_test, X_train, df_pred]:
            df["Embarked"] = df["Embarked"].fillna(value="S")

        for df in [X_test, X_train, df_pred]:
            df["Fare"] = df.groupby("Class")["Fare"].transform(
                lambda x: x.fillna(x.median())
            )

        for df in [X_test, X_train, df_pred]:
            df["Title"] = df["Title"].replace(
                [
                    "Master",
                    "Don",
                    "Rev",
                    "Dr",
                    "Major",
                    "Col",
                    "Sir",
                    "Capt",
                    "Countess",
                    "Jonkheer",
                    "Dona",
                ],
                "Rare",
            )
            df["Title"] = df["Title"].replace("Mlle", "Miss")
            df["Title"] = df["Title"].replace("Lady", "Ms")
            df["Title"] = df["Title"].replace("Mme", "Mrs")

        columns = ["Title", "Sex", "Embarked"]
        encoders = {}
        for column in columns:
            encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore", dtype=int
            )
            encoded = encoder.fit_transform(X_train[[column]])
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out([column]),
                index=X_train.index,
            )
            X_train = pd.concat([X_train.drop(column, axis=1), encoded_df], axis=1)
            encoders[column] = encoder

        for dataset in [X_test, df_pred]:
            for column in columns:
                encoder = encoders[column]
                encoded = encoder.transform(dataset[[column]])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=encoder.get_feature_names_out([column]),
                    index=dataset.index,
                )
                dataset.drop(column, axis=1, inplace=True)
                dataset[encoded_df.columns] = encoded_df

        X_train["Fare_Bins"], bins = pd.qcut(
            X_train["Fare"], q=3, labels=[0, 1, 2], retbins=True
        )
        X_train.drop(columns=["Fare"], inplace=True)

        for df in [X_test, df_pred]:
            df["Fare_Bins"] = pd.cut(
                df["Fare"], bins=bins, labels=[0, 1, 2], include_lowest=True
            )
            df.drop(columns=["Fare"], inplace=True)

        X_train["Fare_Bins"] = X_train["Fare_Bins"].astype(int)
        for df in [X_test, df_pred]:
            df["Fare_Bins"] = df["Fare_Bins"].astype(int)

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=int)
        encoded = encoder.fit_transform(X_train[["Fare_Bins"]])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(["Fare_Bins"]),
            index=X_train.index,
        )
        X_train = pd.concat([X_train.drop("Fare_Bins", axis=1), encoded_df], axis=1)

        for dataset in [X_test, df_pred]:
            encoded = encoder.transform(dataset[["Fare_Bins"]])
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out(["Fare_Bins"]),
                index=dataset.index,
            )
            dataset.drop("Fare_Bins", axis=1, inplace=True)
            dataset[encoded_df.columns] = encoded_df

        return X_train, X_test, df_pred

    except Exception as e:
        logging.info("Function to transform data has encountered a problem.")
        raise CustomException(e, sys) from e
