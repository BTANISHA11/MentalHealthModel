import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('mental_health_model.pkl')

# Get feature names from the model
feature_names = model.feature_names_in_

# Function to predict mental health condition
def predict(input_data):
    df = pd.DataFrame([input_data])
    # Ensure the DataFrame has the same columns as the model was trained on
    df = df.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(df)
    return prediction[0]

# Streamlit UI
def mental_health_ui():
    st.title("Mental Health Condition Predictor")
    
    # Input Fields
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    family_history = st.selectbox("Family History of Mental Health Issues", ['Yes', 'No'])
    work_interfere = st.selectbox("Work Interference Due to Mental Health", ['Never', 'Rarely', 'Sometimes', 'Often'])
    
    # Process Input Data
    input_data = {
        'Age': age,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
        'family_history_Yes': 1 if family_history == 'Yes' else 0,
        'work_interfere_Rarely': 1 if work_interfere == 'Rarely' else 0,
        'work_interfere_Sometimes': 1 if work_interfere == 'Sometimes' else 0,
        'work_interfere_Often': 1 if work_interfere == 'Often' else 0
    }

    # Prediction
    if st.button("Predict"):
        prediction = predict(input_data)
        st.success(f"Predicted Mental Health Condition: **{'Needs Treatment' if prediction == 1 else 'No Treatment Needed'}**")

if __name__ == "__main__":
    mental_health_ui()