# Import Library
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import pickle

import warnings
warnings.filterwarnings('ignore')

# 1. Load dataset
df = pd.read_csv('diabetes.csv')

cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# 2. Handle missing values
imputer = SimpleImputer(strategy='median')
df[cols_with_zeros] = imputer.fit_transform(df[cols_with_zeros])

# 3. Check if class in imbalanced
X = df.drop('Outcome', axis=1)
Y = df['Outcome']

# Apply SMOTE for handling class imbalance
smote = SMOTE()
transform_feature, transform_label = smote.fit_resample(X, Y)

# Spilt the data intp train & test
X_train, X_test, Y_train, Y_test = train_test_split(transform_feature, transform_label, test_size=0.2, random_state=2, stratify=transform_label )

# Feature Scaling
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaler, Y_train)

# Prediction
Y_pred = model.predict(X_test_scaler)

# Evaluation
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

# Save the model
with open('app/diabetes_model.pkl', 'wb') as file:
    pickle.dump((scaler, model), file)

print("Model saved successfully as 'app/diabetes_model.pkl")