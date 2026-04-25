# comp-334-assignment-2
 Titanic Survival Prediction 🚢

Predicting passenger survival on the Titanic using data cleaning, feature engineering, and feature selection techniques.

 Dataset

train.csv — training data (includes **Survived**)
  titanic_test.csv/ `test.csv` — test data

**Target:** `Survived` (1 = survived, 0 = not survived)


Workflow

1. Data Cleaning

* Handled missing values (Age, Embarked)
* Extracted **Deck**, created **CabinMissing**
* Capped Fare outliers
* Normalized text & removed duplicates

**Output:** `clean.train.csv`

2. Feature Engineering

* Created features: **FamilySize, IsAlone, Title, AgeGroup, FarePerPerson**
* One-hot encoding for categorical variables
* Log transformations: **LogFare, LogAge**
* Feature scaling applied

**Output:** `engineer.train.csv`

3. Feature Selection

* Correlation analysis
* Random Forest importance
* Recursive Feature Elimination (RFE)

**Output:** `selected_features.csv`
Key Insights

* Family-related features strongly influence survival
* Socioeconomic factors (**Pclass, Fare, Title**) are important
* Log transformations improve model stability
* Feature selection methods agree on key predictors
 Project Structure


titanic_assignment/
├── data/
├── notebooks/
├── scripts/
├── requirements.txt
└── README.md

 How to Run
Notebook

bash
jupyter notebook


Scripts

bash
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/feature_selection.py

 Tech Stack

* Python
* Pandas
* Scikit-learn
* NumPy


