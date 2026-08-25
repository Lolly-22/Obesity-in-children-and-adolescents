import streamlit as st
import pandas as pd
import joblib
import os
import requests

# Ensure all necessary scikit-learn classes are available when loading the pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Path to the saved model on GitHub
joblib_file = "https://github.com/Lolly-22/Obesity-in-children-and-adolescents/releases/download/latest/obesity2.joblib"

# Load the trained pipeline
@st.cache_resource(show_spinner=False)
def load_model(url, local_path="obesity2.joblib"):
    # 1. Download the file if we don't have it locally yet
    if not os.path.exists(local_path):
        try:
            with st.spinner("Downloading model from GitHub... this may take a moment."):
                response = requests.get(url, stream=True)
                response.raise_for_status() # Check for HTTP errors (like 404 Not Found)
                
                with open(local_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
        except Exception as e:
            st.error(f"Failed to download the model from GitHub: {e}")
            return None

    # 2. Load the downloaded local file
    try:
        return joblib.load(local_path)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Initialize the pipeline
pipeline = load_model(joblib_file)

if pipeline is None:
    st.stop()

# Pre-defined unique values for the dropdowns
countries = [
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde',
    'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo', "Côte d'Ivoire",
    'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea',
    'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea-Bissau',
    'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania',
    'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda',
    'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia',
    'South Africa', 'South Sudan', 'Sudan', 'Togo', 'Uganda', 'United Republic of Tanzania',
    'Zambia', 'Zimbabwe'
]
sex_options = ['FEMALE', 'MALE', 'TOTAL']
age_groups = ['5-9 years', '10-19 years', '5-19 years']

# Streamlit app layout
st.title("Obesity Rate Prediction in Africa")
st.write("Predict the obesity rate based on year, country, sex, and age group.")

# Input widgets
with st.sidebar:
    st.header("Input Features")
    year = st.number_input("Year", min_value=1990, max_value=2027, value=2022, step=1)
    country = st.selectbox("Country", options=sorted(countries))
    sex = st.selectbox("Sex", options=sex_options)
    age_group = st.selectbox("Age Group", options=age_groups)

# Create a DataFrame for prediction
input_data = pd.DataFrame([{
    'YEAR': year,
    'COUNTRY': country,
    'SEX': sex,
    'AGE GROUP': age_group
}])

st.subheader("Provided Input:")
st.write(input_data)

# Make prediction
if st.button("Predict Obesity Rate"):
    try:
        prediction = pipeline.predict(input_data)[0]
        st.success(f"Predicted Obesity Rate (%): {prediction:.2f}%")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please ensure all model dependencies are correctly loaded and data formats match training data.")
