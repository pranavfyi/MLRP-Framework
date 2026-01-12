import pandas as pd
import numpy as np
import random

NUM_SAMPLES = 200000 
data = []

print(f"Generating {NUM_SAMPLES} realistic medical records. This may take a moment...")

for _ in range(NUM_SAMPLES):
    age = random.randint(18, 90)
    gender = random.choice([0, 1]) # 0=Female, 1=Male
    family_history = random.choice([0, 1]) # 1=Has history of disease
    bmi = round(random.uniform(15.0, 45.0), 1)
    systolic_bp = random.randint(90, 200)
    diastolic_bp = random.randint(60, 120)
    heart_rate = random.randint(50, 120)
    spo2 = random.randint(85, 100)       # Normal is >95
    temperature = round(random.uniform(96.0, 104.0), 1) # Fever > 100.4
    glucose = random.randint(60, 350)
    albumin = round(random.uniform(1.5, 5.5), 1)
    cholesterol = random.randint(120, 350)
    wbc = random.uniform(2.5, 18.0)      # Normal 4.5-11.0 (in thousands)
    platelets = random.randint(100, 500) # Normal 150-450 (in thousands)
    smoking = random.choice([0, 1])

    # Logic
    score = 0
    
    # 1. Chronic Factors
    if age > 50: score += (age - 50) * 0.3
    if bmi > 30: score += (bmi - 25) * 1.5
    if smoking == 1: score += 10
    if family_history == 1: score += 10
    # 2. Acute/Vital Factors
    if systolic_bp > 140: score += (systolic_bp - 130) * 0.5
    if glucose > 125: score += (glucose - 100) * 0.4
    if cholesterol > 200: score += (cholesterol - 200) * 0.1
    if albumin < 3.5: score += (3.5 - albumin) * 10  # Liver/Kidney risk
    # 3. Critical Flags (These spike the risk immediately)
    if spo2 < 92: score += 25  # Respiratory distress
    if temperature > 101: score += 15 # Infection
    if wbc > 12.0 or wbc < 4.0: score += 10 # Immune issue

    # Calculation
    probability = 1 / (1 + np.exp(-(score - 60) / 20))
    outcome = 1 if random.random() < probability else 0

    data.append([
        age, gender, family_history, bmi, systolic_bp, diastolic_bp, 
        heart_rate, spo2, temperature, glucose, albumin, 
        cholesterol, wbc, platelets, smoking, outcome
    ])

columns = [
    'Age', 'Gender', 'Family_History', 'BMI', 'Systolic_BP', 'Diastolic_BP', 
    'Heart_Rate', 'SpO2', 'Temperature', 'Glucose', 'Albumin', 
    'Cholesterol', 'WBC', 'Platelets', 'Smoking', 'Outcome'
]

df = pd.DataFrame(data, columns=columns)
df.to_csv('patient_data_large.csv', index=False)

print(f"✅ SUCCESSFULLY GENERATED {NUM_SAMPLES} RECORDS WITH 15 PARAMETERS.")