import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

#Large Dataset
try:
    print("Loading 200,000 records...")
    data = pd.read_csv('patient_data.csv')
except FileNotFoundError:
    print("❌ Error: Run generate_data.py first!")
    exit()

feature_cols = [
    'Age', 'Gender', 'Family_History', 'BMI', 'Systolic_BP', 'Diastolic_BP', 
    'Heart_Rate', 'SpO2', 'Temperature', 'Glucose', 'Albumin', 
    'Cholesterol', 'WBC', 'Platelets', 'Smoking'
]

X = data[feature_cols]
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training AI on 160,000 samples...")

model = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42) 
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"✅ MODEL TRAINED SUCCESSFULLY!")
print(f"🎯 Accuracy on Test Data: {acc:.2%}")
print("\nClassification Report:\n", classification_report(y_test, preds))

# Save
joblib.dump(model, 'model.pkl')