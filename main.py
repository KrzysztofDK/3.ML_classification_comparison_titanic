import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.components.data_loader import load_csv_with_detected_encoding
from src.plot_generator import run_all_visualizations
from src.components.data_prediction import run_prediction
from src.components.data_transformation import (
    clean_split_and_save_as_csv_data,
    transform_data,
)
from src.components.model_trainer import (
    create_base_models,
    create_feature_selected_models,
    get_gridsearch_models_and_params,
    model_training_saving_evaluating,
)


def main():
    logging.info("Main program has started.")

    excel_path = os.path.join("artifacts", "metrics.xlsx")
    if os.path.exists(excel_path):
        os.remove(excel_path)

    BASE_DIR = os.path.dirname(__file__)
    data_to_predict_path = os.path.join(BASE_DIR, "notebook", "data", "test.csv")
    train_data_path = os.path.join(BASE_DIR, "notebook", "data", "train.csv")

    df = load_csv_with_detected_encoding(train_data_path)
    df_predict = load_csv_with_detected_encoding(data_to_predict_path)

    X_train, X_test, y_train, y_test, df_predict = clean_split_and_save_as_csv_data(
        df, df_predict
    )

    run_all_visualizations(X_train)

    X_train, X_test, df_predict = transform_data(X_train, X_test, df_predict)

    base_models = create_base_models()
    feature_selected_models = create_feature_selected_models()
    gridsearch_models = get_gridsearch_models_and_params()

    model_training_saving_evaluating(X_train, X_test, y_train, y_test, base_models)
    model_training_saving_evaluating(
        X_train, X_test, y_train, y_test, feature_selected_models
    )
    model_training_saving_evaluating(
        X_train,
        X_test,
        y_train,
        y_test,
        gridsearch_models,
        use_cv=5,
        scoring_metric="recall",
    )

    run_prediction(df_predict)

    logging.info("Main program ended.")


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print("Critical error.")
        logging.info("Critical error.")
        raise CustomException(e, sys) from e
