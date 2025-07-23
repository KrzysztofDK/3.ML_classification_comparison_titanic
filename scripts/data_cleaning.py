import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from sklearn.impute import SimpleImputer
from .engineer_features import agumentation_with_columns
from sklearn.preprocessing import OneHotEncoder

def handling_nulls(X_train: pd.DataFrame, X_test: pd.DataFrame, df_pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to fill and visualize nulls in ML_comparison project.
    """    
    df_full = pd.concat([X_train, X_test, df_pred], axis=0, ignore_index=True)
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_full.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Nulls in dataframe')
    plt.savefig(os.path.join('images', 'isnull.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    if (df_full.isnull().sum() == 0).all():
        print('There are no nulls.')
        return X_train, X_test, df_pred
    else:
        print('There are some null values!')
        
        age_imputer = SimpleImputer(strategy='median')
        X_train['Age'] = age_imputer.fit_transform(X_train[['Age']])
        X_test['Age'] = age_imputer.transform(X_test[['Age']])
        df_pred['Age'] = age_imputer.transform(df_pred[['Age']])

        X_train['HasCabin'] = X_train['Cabin'].notna().astype(int)
        X_test['HasCabin'] = X_test['Cabin'].notna().astype(int)
        df_pred['HasCabin'] = df_pred['Cabin'].notna().astype(int)
    
        df_full = pd.concat([X_train, X_test, df_pred], axis=0, ignore_index=True)
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_full.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title('Fixed nulls in dataframe')
        plt.savefig(os.path.join('images', 'isnull_fixed.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print('Missing age values imputed with median. Cabin converted to 0 and 1.')
        return X_train, X_test, df_pred

def clean_data(train: pd.DataFrame, test: pd.DataFrame, pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to clean data in ML_comparison project.
    """
    X_train = train.copy()
    X_test = test.copy()
    df_pred = pred.copy()
    
    df_full = pd.concat([X_train, X_test, df_pred], axis=0, ignore_index=True)
    num_duplicates = df_full.duplicated(keep='first').sum()
    if num_duplicates == 0:
        print("There aren't any duplicates.")
    else:
        print(f'Number of duplicates: {num_duplicates}')
    
    X_train, X_test, df_pred = handling_nulls(X_train, X_test, df_pred)

    for df in [X_test, X_train, df_pred]:
        df.rename(columns={'PassengerId': 'Id', 'Pclass': 'Class', 'SibSp': 'Siblings_Spouses', 'Parch': 'Parents_Childs'}, inplace=True)

    X_train, X_test, df_pred = agumentation_with_columns(X_train, X_test, df_pred)

    for df in [X_test, X_train]:
        df.drop(columns=['Id', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    for df in [df_pred]:
        df.drop(columns=['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    for df in [X_test, X_train, df_pred]:
        columns = ['Age', 'HasCabin']
        for col in columns:
            df[col] = df[col].astype('int64')
            
    for df in [X_test, X_train, df_pred]:
        df['Embarked'] = df['Embarked'].fillna(value='S')
        
    for df in [X_test, X_train, df_pred]:
        df['Fare'] = df.groupby('Class')['Fare'].transform(lambda x: x.fillna(x.median()))

    for df in [X_test, X_train, df_pred]:
        df['Title'] = df['Title'].replace(['Master', 'Don', 'Rev', 'Dr', 'Major', 'Col', 'Sir', 'Capt', 'Countess', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Lady', 'Ms')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

    columns = ['Title', 'Sex', 'Embarked']
    encoders = {}
    for column in columns:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)
        encoded = encoder.fit_transform(X_train[[column]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]), index=X_train.index)
        X_train = pd.concat([X_train.drop(column, axis=1), encoded_df], axis=1)
        encoders[column] = encoder
        
    for dataset in [X_test, df_pred]:
        for column in columns:
            encoder = encoders[column]
            encoded = encoder.transform(dataset[[column]])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]), index=dataset.index)
            dataset.drop(column, axis=1, inplace=True)
            dataset[encoded_df.columns] = encoded_df

    return X_train, X_test, df_pred

def clean_fare_after_plots(X_train: pd.DataFrame, X_test: pd.DataFrame, df_pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Function to clean 'Fare' column i Titanic dataset after plot generation.
        """
        X_train['Fare_Bins'], bins = pd.qcut(X_train['Fare'], q=3, labels=[0, 1, 2], retbins=True)
        X_train.drop(columns=['Fare'], inplace=True)

        for df in [X_test, df_pred]:
            df['Fare_Bins'] = pd.cut(df['Fare'], bins=bins, labels=[0, 1, 2], include_lowest=True)
            df.drop(columns=['Fare'], inplace=True)
            
        X_train['Fare_Bins'] = X_train['Fare_Bins'].astype(int)
        for df in [X_test, df_pred]:
            df['Fare_Bins'] = df['Fare_Bins'].astype(int)
            
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)
        encoded = encoder.fit_transform(X_train[['Fare_Bins']])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Fare_Bins']), index=X_train.index)
        X_train = pd.concat([X_train.drop('Fare_Bins', axis=1), encoded_df], axis=1)
            
        for dataset in [X_test, df_pred]:
            encoded = encoder.transform(dataset[['Fare_Bins']])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Fare_Bins']), index=dataset.index)
            dataset.drop('Fare_Bins', axis=1, inplace=True)
            dataset[encoded_df.columns] = encoded_df

        return X_train, X_test, df_pred