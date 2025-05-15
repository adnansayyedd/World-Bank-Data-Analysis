# Import Data manipulation Libraries

import pandas as pd
import numpy as np

# Import Data Visualization Libraries

import seaborn as sns
import matplotlib.pyplot as plt

# import filter warnings library

import warnings
warnings.filterwarnings('ignore')

# import logging library

import logging
logging.basicConfig(filename = "model.log",
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

url = 'https://raw.githubusercontent.com/adnansayyedd/World-Bank-Data-Analysis/refs/heads/main/loan_dataset.csv' 

df = pd.read_csv(url) 


#  Data Cleaning      
# Drop the Loan_ID column

df.drop("Loan_ID", axis=1, inplace=True)

# Fill missing values

categorical_cols = df.select_dtypes(include='object').columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)
    


    # Encode Categorical 
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

  #  Handle Outliers (IQR) 

def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper, upper,
                          np.where(df[column] < lower, lower, df[column]))

for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
    cap_outliers(df, col)

    


# Feature and Target

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)

# Kfold Cross Validation (10 splits)
from sklearn.model_selection import cross_val_score

# Logistic Regression
lr_scores = cross_val_score(lr_model, X, y, cv=10)
print("Logistic Regression Cross Validation Scores:", lr_scores)
print("Logistic Regression Cross Validation Accuracy:", lr_scores.mean())



