# APPENDIX A: SOURCE CODE LISTINGS

This appendix contains the complete source code for all major modules of the Tourism Carbon Footprint Prediction System.

---

## A.1 Web Application Module (app.py)

This module provides the Streamlit-based user interface for the carbon footprint prediction system.

```python
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load model and features
model_path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
features_path = os.path.join(os.path.dirname(__file__), "../model/features.pkl")

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)

st.title("Tourism Carbon Footprint Calculator")
st.write("This app predicts the overall carbon emission level for a tourist's trip.")

# Create organized input sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trip Details")
    trip_days = st.number_input("Trip Duration (days)", 1, 30, 6)
    distance_km = st.number_input("Travel Distance (km)", 1.0, 1000.0, 54.8)
    occupancy = st.number_input("Vehicle Occupancy", 1, 100, 27)
    
    st.subheader("Environmental Factors")
    congestion_factor = st.slider("Congestion Factor (0.8-1.2)", 0.8, 1.2, 0.91, 0.01)
    terrain_factor = st.slider("Terrain Factor (1.0-1.3)", 1.0, 1.3, 1.15, 0.01)

with col2:
    st.subheader("Accommodation")
    nightly_kwh_est = st.number_input("Nightly Electricity (kWh)", 1.0, 200.0, 8.44)
    grid_ef_kgCO2_per_kWh = st.number_input("Grid Carbon Efficiency", 0.01, 1.0, 0.07)
    
    st.subheader("Food & Waste")
    food_emissions_kgCO2 = st.number_input("Food Emissions (kgCO2)", 0.0, 100.0, 16.884)
    waste_emissions_kgCO2 = st.number_input("Waste Emissions (kgCO2)", 0.0, 50.0, 2.998)
    plastic_emissions_kgCO2 = st.number_input("Plastic Emissions (kgCO2)", 0.0, 10.0, 0.3)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Transport Emissions")
    transport_emissions_kgCO2 = st.number_input("Transport Emissions (kgCO2)", 0.0, 500.0, 0.037)

with col4:
    st.subheader("Accommodation Emissions")
    accommodation_elec_kgCO2 = st.number_input("Electricity Emissions (kgCO2)", 0.0, 200.0, 4.805)
    accommodation_gen_kgCO2 = st.number_input("Generator Emissions (kgCO2)", 0.0, 200.0, 11.095)

# Recommendation Function
def get_recommendation(level):
    if level == "low":
        return "Low emissions. Keep using eco-friendly transport and sustainable practices."
    elif level == "medium":
        return "Medium emissions. Reduce meat consumption and minimize plastic usage."
    elif level == "high":
        return "High emissions! Switch to public transport and choose eco-certified hotels."
    else:
        return "No recommendation available."

def align_features(input_df, expected_columns):
    df_local = pd.get_dummies(input_df)
    for col in expected_columns:
        if col not in df_local.columns:
            df_local[col] = 0
    df_local = df_local[expected_columns]
    return df_local

# Prediction Button
if st.button("Predict Emission Level", use_container_width=True):
    input_dict = {
        'trip_days': trip_days,
        'distance_km': distance_km,
        'occupancy': occupancy,
        'congestion_factor': congestion_factor,
        'terrain_factor': terrain_factor,
        'nightly_kwh_est': nightly_kwh_est,
        'grid_ef_kgCO2_per_kWh': grid_ef_kgCO2_per_kWh,
        'food_emissions_kgCO2': food_emissions_kgCO2,
        'waste_emissions_kgCO2': waste_emissions_kgCO2,
        'plastic_emissions_kgCO2': plastic_emissions_kgCO2,
        'transport_emissions_kgCO2': transport_emissions_kgCO2,
        'accommodation_elec_kgCO2': accommodation_elec_kgCO2,
        'accommodation_gen_kgCO2': accommodation_gen_kgCO2
    }
    
    input_df = pd.DataFrame([input_dict])
    aligned_df = align_features(input_df, feature_columns)
    input_data = aligned_df.values
    
    prediction = model.predict(input_data)[0]

    st.subheader(f"Emission Level: {prediction.upper()}")
    st.write("Recommendation:")
    st.success(get_recommendation(prediction))
```

---

## A.2 Training Module (training.py)

This module handles the machine learning model training process using the tourism dataset.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib, os

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "../data/tourism_5000_rows.csv")
df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Create emission level classification based on total_trip_emissions_kgCO2
total_emissions = df['total_trip_emissions_kgCO2']
percentile_33 = total_emissions.quantile(0.33)
percentile_67 = total_emissions.quantile(0.67)

def classify_emission(value):
    if value <= percentile_33:
        return 'low'
    elif value <= percentile_67:
        return 'medium'
    else:
        return 'high'

df['emission_level'] = df['total_trip_emissions_kgCO2'].apply(classify_emission)

print("\nEmission Level Distribution:")
print(df['emission_level'].value_counts())

# REMOVE DATA LEAKAGE: Drop columns that directly contribute to total emissions
leak_cols = [
    'emission_level',
    'total_trip_emissions_kgCO2',
    'record_id',
    'trip_start_date',
    'transport_emissions_kgCO2',
    'accommodation_elec_kgCO2',
    'accommodation_gen_kgCO2',
    'festival_gen_emissions_kgCO2',
    'pilgrimage_emissions_kgCO2',
    'food_emissions_kgCO2',
    'rice_emissions_kgCO2',
    'waste_emissions_kgCO2',
    'plastic_emissions_kgCO2'
]

print("\nRemoving data leakage columns:")
for col in leak_cols:
    if col in df.columns:
        print(f"  - {col}")

feature_df = df.drop(columns=[col for col in leak_cols if col in df.columns])

# Handle categorical columns
cat_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)
feature_df = feature_df.fillna(0)

feature_columns = feature_df.columns.tolist()
X = feature_df
y = df['emission_level']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ML model (optimized Random Forest)
model = RandomForestClassifier(
    n_estimators=800,
    random_state=42,
    max_depth=20,
    max_features='sqrt',
    min_samples_leaf=2,
    min_samples_split=4,
    class_weight='balanced',
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predict test set
pred = model.predict(X_test)

# Print evaluation
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# Feature importance
print("\nTop 5 Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head())

# Save model
model_dir = os.path.join(os.path.dirname(__file__), "../model")
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.pkl"))
joblib.dump(feature_columns, os.path.join(model_dir, "features.pkl"))

print("\nModel saved successfully!")
print("Features saved successfully!")
```

---

## A.3 Prediction Module (predict.py)

This module loads the trained model and performs predictions on new input data.

```python
import joblib
import numpy as np
import pandas as pd
import os
from recomendation import get_recommendation

# Load trained model and features
model_path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
features_path = os.path.join(os.path.dirname(__file__), "../model/features.pkl")

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)

# Example: Create sample data for prediction
sample_data = {
    'trip_days': 6,
    'distance_km': 54.8,
    'occupancy': 27,
    'congestion_factor': 0.91,
    'terrain_factor': 1.15,
    'nightly_kwh_est': 8.44,
    'grid_ef_kgCO2_per_kWh': 0.07,
    'food_emissions_kgCO2': 16.884,
    'waste_emissions_kgCO2': 2.998,
    'plastic_emissions_kgCO2': 0.3,
    'transport_emissions_kgCO2': 0.037,
    'accommodation_elec_kgCO2': 4.805,
    'accommodation_gen_kgCO2': 11.095
}

def align_features(input_df, expected_columns):
    df_local = pd.get_dummies(input_df)
    for col in expected_columns:
        if col not in df_local.columns:
            df_local[col] = 0
    df_local = df_local[expected_columns]
    return df_local

# Create DataFrame and ensure column order
data_df = pd.DataFrame([sample_data])
aligned_df = align_features(data_df, feature_columns)
data = aligned_df.values

pred = model.predict(data)[0]

print("\n" + "="*50)
print("PREDICTION RESULT")
print("="*50)
print("Predicted Emission Level:", pred.upper())
print("\nRecommendation:")
print(get_recommendation(pred))
print("="*50)
```

---

## A.4 Recommendation Module (recomendation.py)

This module provides eco-friendly recommendations based on the predicted emission level.

```python
def get_recommendation(level):
    if level == "low":
        return (
            "Low emissions. Continue sustainable habits: eco-friendly travel, "
            "low meat use, low plastic, and energy saving."
        )

    elif level == "medium":
        return (
            "Medium emissions. Reduce meat meals, minimize plastic, "
            "use less electricity, and choose fewer high-impact activities."
        )

    elif level == "high":
        return (
            "High emissions! Switch to EV/public transport, reduce AC use, "
            "eat more vegetarian meals, avoid plastic, and choose eco-certified hotels."
        )

    return "No recommendation available."
```

---

## A.5 Requirements File (requirements.txt)

This file lists all Python dependencies required to run the system.

```
streamlit>=1.51.0
pandas>=1.4.0
numpy>=1.22.0
scikit-learn>=1.7.0
joblib>=1.3.0
```

---

## A.6 Dataset Sample

The first few records from the tourism dataset (tourism_5000_rows.csv):

| Column | Description | Sample Value |
|--------|-------------|--------------|
| record_id | Unique identifier | 1 |
| trip_start_date | Start date of trip | 2025-04-30 |
| trip_days | Duration in days | 6 |
| distance_km | Travel distance | 54.8 |
| transport_mode | Type of transport | Public Bus |
| occupancy | Vehicle occupancy | 27 |
| congestion_factor | Traffic congestion | 0.91 |
| terrain_factor | Terrain difficulty | 1.15 |
| hotel_class | Hotel category | Budget Hotel |
| nightly_kwh_est | Nightly electricity | 8.44 |
| food_emissions_kgCO2 | Food carbon emissions | 16.884 |
| waste_emissions_kgCO2 | Waste emissions | 2.998 |
| plastic_emissions_kgCO2 | Plastic emissions | 0.3 |
| total_trip_emissions_kgCO2 | Total emissions | 36.119 |

---

*End of Appendix A*
