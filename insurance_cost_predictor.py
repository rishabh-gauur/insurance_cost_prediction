import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

df = pd.read_csv('medical_insurance.csv')

cols_to_keep = [
    'age','sex','region','urban_rural','income','employment_status','household_size',
    'dependents','bmi','smoker','alcohol_freq','days_hospitalized_last_3yrs','medication_count','systolic_bp','diastolic_bp','plan_type','deductible','copay','annual_premium','diabetes','asthma','cardiovascular_disease',
    'cancer_history','kidney_disease','liver_disease','mental_health'
]

df = df[cols_to_keep]

X = df.drop(columns=['annual_premium'])
y = df['annual_premium']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist() #identify the data type

# in this i am defining the transformerss

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# we create pipeline because we dont have to train our data again and imply encoding etc after model training to test values
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(max_depth=5))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

score = r2_score(y_test, y_pred)

with open('insurance_pipeline.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("Model trained and saved.")