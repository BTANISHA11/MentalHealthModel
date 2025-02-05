import joblib
import pandas as pd

def predict(symptoms):
    model = joblib.load('mental_health_model.pkl')
    df = pd.DataFrame([symptoms])
    
    # Ensure the DataFrame has the same columns as the model was trained on
    feature_names = model.feature_names_in_
    df = df.reindex(columns=feature_names, fill_value=0)
    
    prediction = model.predict(df)
    return 'Needs Treatment' if prediction[0] == 1 else 'No Treatment Needed'

if __name__ == "__main__":
    symptoms = {
        'Age': 30,
        'Gender_Male': 1,
        'Gender_Female': 0,
        'family_history_Yes': 1,
        'work_interfere_Rarely': 0,
        'work_interfere_Sometimes': 1,
        'work_interfere_Often': 0
    }
    print("Predicted Condition:", predict(symptoms))