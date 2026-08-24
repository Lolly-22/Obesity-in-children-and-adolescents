import streamlit as st
import pandas as pd
import joblib
import os

# Ensure all necessary scikit-learn classes are available when loading the pipeline
# This is crucial because the pipeline contains these objects
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Path to the saved model
joblib_file = "https://drive.google.com/file/d/1oaHLCgr7ns8WFpowA7tuaetHhsEIcVAm/view?usp=sharing"

# Load the trained pipeline
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

pipeline = load_model(joblib_file)

if pipeline is None:
    st.stop()

# Extract unique values for dropdowns from the Data1 DataFrame (assuming Data1 is available in the environment)
# In a real-world Streamlit app, you might load these from a file or hardcode them if they are static.
# For this context, we will assume these values are known or can be manually copied.
# Let's get these from the `Data1` variable which is in the kernel state.
# For a standalone app, you would load your original dataset or store these values separately.

# As a workaround for a standalone app, if Data1 is not directly accessible, 
# one would usually save these unique values alongside the model or load the original data to get them.
# For the purpose of generating code in Colab, we'll assume `Data1` is conceptually available for getting these lists.

# Example values (replace with actual unique values from your Data1['COUNTRY'].unique(), etc. if running standalone)
# If running directly after the notebook, Data1 is in memory, so we can access it.
# If running as a standalone .py file, you'd need to load the original data or hardcode these lists.

# Since `Data1` was used to train the model, we should have its values available.
# Assuming `Data1` is a global variable from previous cells (for notebook context)
# In a production app, these lists would be pre-saved or fetched from a database/file.

# If Data1 is not available, these lists must be explicitly defined using the values observed during training.
# For this demonstration within the notebook, we will use placeholder lists if Data1 is not directly callable in this isolated cell's execution context.

countries = ['Gambia', 'Ghana', 'Zambia', 'Namibia', 'Niger', 'Nigeria', 'Benin', 'Cabo Verde',
             'Sierra Leone', 'Somalia', 'South Africa', 'Zimbabwe', 'South Sudan', "Côte d'Ivoire",
             'Angola', 'Guinea-Bissau', 'Djibouti', 'Gabon', 'Morocco', 'Mozambique', 'Congo',
             'Democratic Republic of the Congo', 'Togo', 'Comoros', 'Lesotho', 'Botswana',
             'Burundi', 'Cameroon', 'Senegal', 'Seychelles', 'Algeria', 'Eritrea',
             'United Republic of Tanzania', 'Burkina Faso', 'Eswatini', 'Ethiopia', 'Egypt',
             'Mauritania', 'Mauritius', 'Central African Republic', 'Chad', 'Sudan', 'Kenya',
             'Rwanda', 'Madagascar', 'Malawi', 'Mali', 'Liberia', 'Libya', 'Uganda',
             'Sao Tome and Principe', 'Equatorial Guinea']
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
