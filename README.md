# MLRP-Framework (Liver Risk Predictor)

 MLRP-Framework is a predictive diagnostic tool that leverages machine learning to assess the likelihood of liver disease based on patient clinical data. By analyzing key medical markers, the application provides a quick, non-invasive risk stratification to assist in early-stage health screening.

# 

# 🚀 Live Application

 Access the tool directly via the web here: \https://liver-risk-predictor.streamlit.app/


# 🛠️ How it Works

The framework uses an XGBoost Classifier, a powerful algorithm designed to handle complex clinical datasets with high precision.



Input Data: Users enter clinical parameters such as Age, Gender, Bilirubin levels, and Liver Enzyme values (ALT, AST).
Risk Processing: The model processes these inputs to determine if a patient falls into a "Liver-Disease" (LD) or "Non-Liver-Disease" (Non-LD) category.
Instant Results: The web interface displays a clear risk assessment, allowing for rapid preliminary insights.

# 

# 📊 Key Features

XGBoost Integration: Utilizes state-of-the-art gradient boosting for reliable classification on tabular medical data.
Simplified Interface: A user-friendly frontend built with Streamlit, designed for both researchers and non-technical users.
Cloud Accessibility: Fully deployed online for instant use without any local configuration.

# 🗺️ Roadmap
To improve the accuracy and utility of the framework, the following updates are planned:

 Data Balancing: Implementation of SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance and reduce "Low Risk" bias.
Ensemble Modeling: Developing a Voting Classifier that combines XGBoost with Logistic Regression for more robust predictions.
Model Interpretability: Integrating SHAP or LIME visualizations to explain which clinical features most influenced a specific risk score.
Expanded Feature Set: Incorporating additional health markers such as BMI, Albumin levels, and Glucose.

# ⚠️ Important Notes

 Current Status: This project is a functional prototype. The current model may exhibit a bias toward "Low Risk" results due to class imbalance in the training data, a known technical limitation currently under refinement.
Medical Disclaimer: This tool is intended for research and educational purposes only. It is not a clinical diagnostic tool and should not replace professional medical consultation.



