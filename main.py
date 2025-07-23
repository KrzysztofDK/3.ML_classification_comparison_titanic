import os
import logging
from scripts import load_csv_with_detected_encoding
from scripts import data_split
from scripts import clean_data
from scripts import clean_fare_after_plots
from scripts import run_all_visualizations
from scripts import create_base_models
from scripts import create_feature_selected_models
from scripts import get_gridsearch_models_and_params
from scripts import model_training_saving_evaluating
from scripts import run_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filename=('main.log'),
    filemode='w'
)
logger = logging.getLogger(__name__)


def main():
    if os.path.exists("metrics.xlsx"):
        os.remove("metrics.xlsx")

    logger.info("START")
    
    BASE_DIR = os.path.dirname(__file__)

    logger.info("Reading CSV file.")
    data_path = os.path.join(BASE_DIR, "data", "train.csv")
    df = load_csv_with_detected_encoding(data_path)
    data_path = os.path.join(BASE_DIR, "data", "test.csv")
    df_pred = load_csv_with_detected_encoding(data_path)
    
    logger.info("Spliting data.")
    X_train, X_test, y_train, y_test = data_split(df=df, column='Survived', test_size=0.2, random_state=42)
    
    logger.info("Data cleaning.")
    X_train, X_test, df_pred = clean_data(X_train, X_test, df_pred)
    
    logger.info("Running visualizations.")
    run_all_visualizations(X_train)
    
    logger.info("'Fare' column cleaning.")
    X_train, X_test, df_pred = clean_fare_after_plots(X_train, X_test, df_pred)
    
    logger.info("Creating base models.")
    models = create_base_models()
    logger.info("Creating models with feature selection/extraction.")
    models_fs = create_feature_selected_models(n_features_to_select=8)
    logger.info("Creating models with parameters for hyperparameter tuning.")
    models_fs_cv = get_gridsearch_models_and_params()

    logger.info("Models evaluation.")
    model_training_saving_evaluating(X_train, X_test, y_train, y_test, models)
    model_training_saving_evaluating(X_train, X_test, y_train, y_test, models_fs, use_cv=True)
    model_training_saving_evaluating(X_train, X_test, y_train, y_test, models_fs_cv)

    logger.info("Prediction from file.")
    run_prediction(df_pred)

    logger.info("END")
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception('Critical error.')