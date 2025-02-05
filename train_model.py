import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    df = pd.read_csv("survey.csv")
    df = df[['Age', 'Gender', 'family_history', 'treatment', 'work_interfere']].dropna()
    df['Gender'] = df['Gender'].apply(lambda x: 'Male' if 'M' in x else ('Female' if 'F' in x else 'Other'))
    df = pd.get_dummies(df, columns=['Gender', 'family_history', 'work_interfere'], drop_first=True)
    return df

def preprocess_data(df):
    X = df.drop(columns=['treatment_Yes'])  # Predicting treatment
    y = df['treatment_Yes']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'mental_health_model.pkl')
    return model

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))