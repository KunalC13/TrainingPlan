# streamlit_app.py
import streamlit as st
from src.WorkoutGenerator import WorkoutGenerator

# Initialize the WorkoutGenerator instance
wg = WorkoutGenerator()

st.title("Personalized Fitness Workout Generator")
st.write("Enter your profile details below to generate your personalized workout plan:")

# Collect user inputs via Streamlit widgets
user_name = st.text_input("Name")
age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)
weight = st.number_input("Weight (Kg)", min_value=20, max_value=300, step=1)
fitness_goals = st.text_input("Fitness Goals (e.g., muscle gain, fat loss, endurance)")
fitness_level = st.selectbox("Fitness Level", options=["Beginner", "Intermediate", "Advanced"])
workout_frequency = st.number_input("Workout Frequency (days per week)", min_value=1, max_value=7, step=1)
workout_duration = st.text_input("Workout Duration (e.g., 45 minutes, 1 hour)")
workout_location = st.text_input("Workout Location (e.g., Gym, Home)")
equipment = st.text_input("Equipment Available (if home)")
focus_areas = st.text_input("Focus Areas (e.g., legs, arms)")
cardio_preference = st.text_input("Cardio Preference (e.g., running, cycling, none)")
daily_activity = st.text_input("Daily Activity Level (e.g., sedentary, active)")
injury = st.text_input("Any Injury/Medical Condition? (Yes/No)")
injury_details = st.text_area("Injury Details (if any)")

user_data = {
    "Age": age,
    "Gender": gender,
    "Height (cm)": height,
    "Weight (Kg)": weight,
    "Fitness Goals": fitness_goals,
    "Fitness Level": fitness_level,
    "Workout Frequency": workout_frequency,
    "Workout Duration": workout_duration,
    "Workout Location": workout_location,
    "Equipment": equipment,
    "Focus Areas": focus_areas,
    "Cardio Preference": cardio_preference,
    "Daily Activity": daily_activity,
    "Injury": injury,
    "Injury Details": injury_details,
}

if st.button("Generate Workout Plan"):
    with st.spinner("Generating workout plan..."):
        workout_plan = wg.generate_workout_plan(user_name, user_data)
    st.subheader("Your Personalized Workout Plan:")
    st.write(workout_plan)
