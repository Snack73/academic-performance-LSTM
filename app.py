# app.py

import streamlit as st
from keras.saving import load_model
import joblib
import numpy as np
import os

# Check if the model files exist
required_files = ['model.h5', 'scaler_X.pkl', 'scaler_y.pkl', 'label_encoders.pkl']
missing = [file for file in required_files if not os.path.exists(file)]

if missing:
    st.error(f"Missing model files: {', '.join(missing)}. Please run training first.")
    st.stop()

# Load model and preprocessing tools
model = load_model('model.h5')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define UI options
gender_options = ['female', 'male']
race_options = ['group A', 'group B', 'group C', 'group D', 'group E']
parental_education_options = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]
lunch_options = ['standard', 'free/reduced']
test_prep_options = ['none', 'completed']

# Streamlit UI
st.title('📊 Student Math Score Prediction')
st.write('Enter student details below to predict their **Math Score**.')

# Input fields
reading_score = st.slider('📘 Reading Score', min_value=0, max_value=100, value=50)
writing_score = st.slider('✏️ Writing Score', min_value=0, max_value=100, value=50)
gender = st.selectbox('Gender', gender_options)
race = st.selectbox('Race/Ethnicity', race_options)
parental_education = st.selectbox('Parental Level of Education', parental_education_options)
lunch = st.selectbox('Lunch', lunch_options)
test_prep = st.selectbox('Test Preparation Course', test_prep_options)

# Prediction logic
if st.button('🎯 Predict Math Score'):
    try:
        # Encode categorical inputs using saved LabelEncoders
        gender_encoded = label_encoders['gender'].transform([gender])[0]
        race_encoded = label_encoders['race/ethnicity'].transform([race])[0]
        parental_education_encoded = label_encoders['parental level of education'].transform([parental_education])[0]
        lunch_encoded = label_encoders['lunch'].transform([lunch])[0]
        test_prep_encoded = label_encoders['test preparation course'].transform([test_prep])[0]

        # Feature vector
        X_input = np.array([[reading_score, writing_score, gender_encoded, race_encoded,
                             parental_education_encoded, lunch_encoded, test_prep_encoded]])
        
        # Scale input
        X_input_scaled = scaler_X.transform(X_input)

        # Predict and inverse transform
        y_pred_scaled = model.predict(X_input_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]

        st.success(f"🧮 Predicted Math Score: **{y_pred:.2f}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
