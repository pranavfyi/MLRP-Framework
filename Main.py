import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


df = pd.read_csv("model1.csv")

features = [
    "Age", "sex", "BMI", "waist", "Fasting Blood sugar",
    "Triglyceride", "Cholestrol", "HDL", "LDL",
    "GGT", "AST (SGOT)", "ALT (SGPT)", "HbA1c", "Insulin"
]

X = df[features].copy()
X.loc[:, "sex"] = X["sex"].map({"M": 1, "F": 0})
y = df["Steatosis stage"].apply(lambda x: 0 if x <= 1 else 1)


imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

print(f"Original Class Distribution: {np.bincount(y)}")
print(f"Resampled Class Distribution: {np.bincount(y_resampled)}")


X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    scale_pos_weight=1,
    eval_metric="logloss"
)

model.fit(X_train, y_train)


y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"✅ Improved ROC-AUC: {auc:.4f}")


y_pred = (y_prob >= 0.4).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


joblib.dump(model, "mlrp_model_v1.pkl")
print("\n🚀 Model saved as 'mlrp_model_v1.pkl'")