# 📊 Comparison of Machine Learning Classification Models on a Titanic Dataset

## 🧠 About the Project
A project to predict whether a person survived a disaster by comparing machine learning models, based on certain factors like personID, class, sex, age, etc.

## 📁 Dataset
Dataset taken from Kaggle -> https://www.kaggle.com/competitions/titanic/overview

Contains: personID, class, sex, age, etc.
It consists of two datasets divided into train and test.

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

## ⚙️ Technologies Used
- Python 3.10
- Jupyter Notebook
- VS Code
- Pandas
- Matplotlib
- Seaborn
- Chardet
- scikit-learn
- XGBoost
- joblib
- openpyxl

## 🔧 Installation:
pip install -r requirements.txt
## 🔧 Run by:
python main.py

All visualizations will be saved in 'images' folder.
All ML models will be saved in 'models' folder.
All metrics will be saved in 'results' folder.

## 🧪 Steps Performed
1. **Data Cleaning**
   - Removed duplicates,
   - Filled Nans,
   - Changed columns names,
   - Removed unnecessary columns,
   - Fixed columns data types (and OneHotEncoding, quantiles),
   - Checked for zero intigers/floats.
2. **Feature Engineering**
   - Added a columns with whole family count and extracted name title.
3. **Exploratory Data Analysis**
   - Basic understanding of dataset,
   - Charts were created like histograms, boxplot, pairplot, correlation heatmap,
4. **Models building, feature selection/extraction, hyperparameter tuning, training and metrics**
   - Selected Models:
      + Logistic Regression,
      + Random Forest Classifier,
      + KNN,
      + XGBoost,
      + SVC.
5. **Simple imput prediction for selected model from file**

### Step-by-step analysis with notes and summaries is available in 'notebooks/ML_comparison.ipynb'.

🧑‍💼 Author: Krzysztof Kopytowski
📎 LinkedIn: https://www.linkedin.com/in/krzysztof-kopytowski-74964516a/
📎 GitHub: https://github.com/KrzysztofDK