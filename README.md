# MentalHealthModel
## Mental Health Condition Predictor
A machine learning model that predicts whether an individual requires mental health treatment based on survey responses. The model is trained using a Random Forest Classifier and provides predictions via a Streamlit UI and CLI inference script.

1️⃣ Dataset Preprocessing Steps
The dataset used is survey.csv, which contains various mental health indicators. The following preprocessing steps were applied:

Feature Selection: Used relevant columns:
Age, Gender, family_history, treatment, work_interfere
Handling Missing Data:
Dropped rows with NaN values.
Gender Normalization:
Mapped all gender variations to Male, Female, and Other.
One-Hot Encoding:
Converted categorical features into numerical format.
Splitting Data:
80% for training, 20% for testing.
Feature Consistency:
Saved feature names in feature_columns.pkl to ensure compatibility during prediction.
2️⃣ Model Selection Rationale
We selected a Random Forest Classifier because:
✅ Handles categorical data well.
✅ Resistant to overfitting.
✅ Provides feature importance insights.

Other models considered:

Logistic Regression (too simplistic)
SVM (slower on large datasets)
Neural Networks (overkill for this problem)
Final choice: Random Forest (n_estimators=100, random_state=42) for balanced accuracy and interpretability.

3️⃣ How to Run the Inference Script
Setup Instructions
📌 Step 1: Install Dependencies


pip install -r requirements.txt
📌 Step 2: Train the Model


python train.py
📌 Step 3: Run CLI Inference


python predict.py
Example Input (CLI)

symptoms = {
    'Age': 30,
    'Gender_Male': 1, 'Gender_Female': 0,
    'family_history_Yes': 1, 'family_history_No': 0,
    'work_interfere_Often': 1, 'work_interfere_Never': 0
}
Example Output (CLI)

Predicted Condition: Needs Treatment
4️⃣ UI/CLI Usage Instructions
Running the UI (Streamlit App)

streamlit run mental_health_ui.py
Using the UI
Enter Age
Select Gender, Work Interference, and Family History
Click Predict
View Predicted Mental Health Condition
5️⃣ Files & Directory Structure

📂 MentalHealthModel/
 ├── 📄 survey.csv                # Dataset
 ├── 📄 train.py                   # Model training script
 ├── 📄 predict.py                 # Inference script
 ├── 📄 mental_health_ui.py        # Streamlit UI
 ├── 📄 feature_columns.pkl        # Saved feature names
 ├── 📄 mental_health_model.pkl    # Trained model
 ├── 📄 requirements.txt           # Python dependencies
 ├── 📄 README.md                  # Documentation
6️⃣ Dependencies
Install all required packages:

pip install -r requirements.txt
Main Libraries Used
pandas
joblib
scikit-learn
streamlit
7️⃣ Future Enhancements 🚀
✅ Improve model interpretability using SHAP/LIME
✅ Experiment with XGBoost & Neural Networks
✅ Deploy UI on Streamlit Cloud or Vercel
