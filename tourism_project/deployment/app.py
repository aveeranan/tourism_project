import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="asveeranan/vws-tourism-model", filename="best_tourism_model.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Visit With Us Tourism App")
st.write("Theis is an internal tool for agent to target the correct cutomers whom they can target to sell the tourist packages.")
st.write("Kindly enter the customer details to check whether they are likely to buy the package or not.")

# Collect user input
# -------------------------------
# Numerical Inputs
# -------------------------------
age = st.number_input("Age", min_value=18, max_value=100, value=30)
city_tier = st.selectbox("City Tier", options=[1, 2, 3], help="Tier 1 > Tier 2 > Tier 3")
duration_of_pitch = st.number_input("Duration Of Pitch (minutes)", min_value=0, value=10)
number_of_person_visiting = st.number_input("Number Of Persons Visiting", min_value=1, value=1)
number_of_followups = st.number_input("Number Of Followups", min_value=0, value=0)
preferred_property_star = st.selectbox("Preferred Property Star",options=[1, 2, 3, 4, 5])
number_of_trips = st.number_input("Number Of Trips per year", min_value=0, value=1)
passport = st.radio("Passport", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
own_car = st.radio("Own Car", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
number_of_children_visiting = st.number_input("Number Of Children Visiting (<5 yrs)", min_value=0, value=0)
monthly_income = st.number_input("Monthly Income", min_value=0, value=30000)

# -------------------------------
# Categorical Inputs
# -------------------------------
typeofcontact = st.selectbox("Type of Contact",options=["Company Invited", "Self Inquiry"])
occupation = st.selectbox("Occupation",options=["Salaried", "Freelancer", "Business", "Other"])
gender = st.selectbox("Gender",options=["Male", "Female"])
product_pitched = st.selectbox("Product Pitched",options=["Basic", "Standard", "Deluxe", "Super Deluxe"])
marital_status = st.selectbox("Marital Status",options=["Single", "Married", "Divorced"])
designation = st.selectbox("Designation",options=["Manager", "Senior Manager", "AVP", "VP","Executive", "Director", "Other"])

# -------------------------------
# Collect Input into Dictionary
# -------------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "PreferredPropertyStar": preferred_property_star,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "MonthlyIncome": monthly_income,
    "TypeofContact": typeofcontact,
    "Occupation": occupation,
    "Gender": gender,
    "ProductPitched": product_pitched,
    "MaritalStatus": marital_status,
    "Designation": designation
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Take the product" if prediction == 1 else "NOT Take the product"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
