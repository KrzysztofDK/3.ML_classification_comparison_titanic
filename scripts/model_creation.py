from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE,SelectFromModel
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator

def create_base_models() -> dict[str, BaseEstimator]:
    """
    Project ML_comparison-specific function to create base models dictionary.
    """
    return {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000)),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier()),
    'XGB': XGBClassifier(random_state=42),
    'SVC': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
}
    
def create_feature_selected_models(n_features_to_select: int = 8) -> dict[str, BaseEstimator]:
    """
    Project ML_comparison-specific function to create models with feature selection/extraction dictionary.
    """
    lr_rfe = RFE(estimator=LogisticRegression(), n_features_to_select=n_features_to_select)
    rf_sfm = SelectFromModel(estimator=RandomForestClassifier(random_state=42), max_features=n_features_to_select)
    knn_pca = PCA(n_components=n_features_to_select)
    xgb_sfm = SelectFromModel(estimator=XGBClassifier(random_state=42), max_features=n_features_to_select)
    svc_pca = PCA(n_components=n_features_to_select)
    
    return {
        'Logistic Regression RFE': make_pipeline(StandardScaler(), lr_rfe, LogisticRegression(random_state=42, max_iter=1000)),
        'Random Forest SFM': make_pipeline(rf_sfm, RandomForestClassifier(random_state=42)),
        'KNN PCA': make_pipeline(StandardScaler(), knn_pca, KNeighborsClassifier()),
        'XGB SFM': make_pipeline(xgb_sfm, XGBClassifier(random_state=42)),
        'SVC PCA': make_pipeline(StandardScaler(), svc_pca, SVC(probability=True, random_state=42))
    }