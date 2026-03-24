import pandas as pd
import pickle


with open('insurance_pipeline.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

all_features = [
    'age','sex','region','urban_rural','income','employment_status','household_size',
    'dependents','bmi','smoker','alcohol_freq','days_hospitalized_last_3yrs','medication_count','systolic_bp','diastolic_bp','plan_type','deductible','copay','annual_premium','diabetes','asthma','cardiovascular_disease',
    'cancer_history','kidney_disease','liver_disease','mental_health'
]


print("--- Insurance Predictor (Testing Mode) ---")
user_input_dict = {}

for feature in all_features:
        val = input(f"Enter {feature}: ")
    
    
    
        user_input_dict[feature] = [val]


input_df = pd.DataFrame(user_input_dict)

for col in all_features:
    try:
        input_df[col] = pd.to_numeric(input_df[col])
    except:
        pass 

prediction = loaded_model.predict(input_df)
print(f"Predicted Annual Premium: ${prediction[0]:.2f}")