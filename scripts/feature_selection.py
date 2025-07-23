from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator

def get_gridsearch_models_and_params(n_features_to_select: int = 8) -> dict[str, tuple[BaseEstimator, dict]]:
    """
    Function to create models and parameters to use in hyperparameter tuning in ML_comparison project.
    """
    pipe_lr = make_pipeline(StandardScaler(), RFE(estimator=LogisticRegression()), LogisticRegression(random_state=42, max_iter=1000))
    pipe_rf = make_pipeline(SelectFromModel(estimator=RandomForestClassifier(random_state=42)), RandomForestClassifier(random_state=42))
    pipe_knn = make_pipeline(StandardScaler(), PCA(), KNeighborsClassifier())
    pipe_xgb = make_pipeline(SelectFromModel(estimator=XGBClassifier(random_state=42)), XGBClassifier(random_state=42))
    pipe_svc = make_pipeline(StandardScaler(), PCA(), SVC(probability=True, random_state=42))

    param_grid_lr = {
        'rfe__n_features_to_select': [6, 8, 11, 16, 19],
        'logisticregression__C': [0.1, 1.0, 10.0]
    }

    param_grid_rf = {
        'selectfrommodel__max_features': [6, 8, 11, 16, 19],
        'randomforestclassifier__n_estimators': [50, 100, 200],
        'randomforestclassifier__max_depth': [None, 5, 10, 20]
    }

    param_grid_knn = {
        'pca__n_components': [6, 8, 11, 16, 19],
        'kneighborsclassifier__n_neighbors': [3, 5, 7],
        'kneighborsclassifier__weights': ['uniform', 'distance']
    }

    param_grid_xgb = {
        'selectfrommodel__max_features': [6, 8, 11, 16, 19],
        'xgbclassifier__n_estimators': [50, 100],
        'xgbclassifier__max_depth': [3, 6],
        'xgbclassifier__learning_rate': [0.01, 0.1, 0.3]
    }

    param_grid_svc = {
        'pca__n_components': [6, 8, 11, 16, 19],
        'svc__C': [0.1, 1.0, 10.0],
        'svc__gamma': ['scale', 'auto'],
        'svc__kernel': ['rbf', 'linear']
    }

    models_params = {
        'Logistic Regression RFE CV': (pipe_lr, param_grid_lr),
        'Random Forest SFM CV': (pipe_rf, param_grid_rf),
        'KNN PCA CV': (pipe_knn, param_grid_knn),
        'XGBoost SFM CV': (pipe_xgb, param_grid_xgb),
        'SVC PCA CV': (pipe_svc, param_grid_svc)
    }
    return models_params
