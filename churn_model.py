import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- STEP 1: LOAD DATA ---
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# --- STEP 2: CLEANING ---
# TotalCharges has some empty spaces; we turn them to 0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# We keep a copy of the original data for Power BI, but create a numeric version for ML
ml_df = df.copy()
le = LabelEncoder()

# Encode every string column into a number
for col in ml_df.columns:
    if ml_df[col].dtype == 'object' and col != 'customerID':
        ml_df[col] = le.fit_transform(ml_df[col])

# --- STEP 3: TRAIN THE MODEL ---
X = ml_df.drop(['customerID', 'Churn'], axis=1)
y = ml_df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# --- STEP 4: GENERATE PREDICTIONS ---
# We add the probability scores back to the ORIGINAL dataframe for the dashboard
df['Churn_Probability'] = model.predict_proba(X)[:, 1]
df['Churn_Prediction'] = model.predict(X)

# --- STEP 5: SAVE FOR POWER BI ---
df.to_csv('Churn_Final_Analysis.csv', index=False)
print("File 'Churn_Final_Analysis.csv' has been created successfully!")