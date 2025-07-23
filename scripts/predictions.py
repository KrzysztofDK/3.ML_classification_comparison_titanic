import joblib
import pandas as pd
from pathlib import Path

def run_prediction(df_pred: pd.DataFrame) -> None:
    """
    Function to predict survival of a person.
    """
    try:
        root = Path(__file__).resolve().parent.parent
        model_path = root / "models" / "svc_pca_cv.pkl"
        model = joblib.load(model_path)
        print('Model loaded correctly.')   
        try:
            y_pred = model.predict(df_pred.drop(['Id'], axis=1))
            submission = pd.DataFrame({ 
                'PassengerId': df_pred['Id'],
                'Survived': y_pred.astype(int)
            })
            path = root / "results" / "submission.csv"
            submission.to_csv(path, index=False)
        except Exception as e:
            raise RuntimeError(f'Prediction failed: {e}')
        
    except Exception as e:
        print(f'Error occured during model loading: {e}')