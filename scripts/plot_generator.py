import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

def run_all_visualizations(X_train: pd.DataFrame) -> None:
    """
    Function to visualize data in ML_comparison project.
    """
    try:
        cols = ['Age', 'Siblings_Spouses', 'Parents_Childs', 'HasCabin', 'FamilySize']
        for col in cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(X_train[col], bins=30)
            plt.title(f"Distribution of {col}")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(os.path.join('images', f'{col}_histogram.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        plt.figure(figsize=(10,6))
        sns.histplot(np.log1p(X_train['Fare']), bins=30)
        plt.savefig(os.path.join('images', 'log1p_fare_hist.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12,8))
        sns.heatmap(X_train.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
        plt.savefig(os.path.join('images', 'corr_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        cols = ['Siblings_Spouses', 'Parents_Childs', 'HasCabin', 'FamilySize']
        for col in cols:
            plt.figure(figsize=(10,6))
            sns.boxplot(x=X_train[col], y=X_train['Age'])
            plt.title(f"Age dependence on {col}")
            plt.savefig(os.path.join('images', f'{col}_box.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        print(f'Error while generating chart: {e}')