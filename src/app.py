import streamlit as st
from WorkoutGenerator import WorkoutGenerator

# Ensure a wide layout
st.set_page_config(layout="wide")

# Initialize the WorkoutGenerator
wg = WorkoutGenerator()

st.title("Personalized Fitness Workout Generator")

# Two columns: left for user input, right for displaying the plan
left_col, right_col = st.columns(2)

##############################################
# Left Column: Form Inputs
##############################################
with left_col:
    with st.form(key="workout_form"):
        st.subheader("Enter Your Details")

        # Basic user profile inputs
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

        # Additional options in an expander for a cleaner look
        with st.expander("Additional Options"):
            cardio_preference = st.text_input("Cardio Preference (e.g., running, cycling, none)")
            daily_activity = st.text_input("Daily Activity Level (e.g., sedentary, active)")
            injury = st.text_input("Any Injury/Medical Condition? (Yes/No)")
            injury_details = st.text_area("Injury Details (if any)")

        # Submit button
        submitted = st.form_submit_button("Generate Workout Plan")

        if submitted:
            # Build the user_data dictionary
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

            # Generate the plan and store it in session state
            with st.spinner("Generating your workout plan..."):
                plan_model = wg.generate_workout_plan(user_name, user_data)
            st.session_state["workout_plan"] = plan_model

##############################################
# Right Column: Display the Workout Plan
##############################################
with right_col:
    st.subheader("Your Generated Workout Plan")
    plan_model = st.session_state.get("workout_plan", None)

    if plan_model:
        for day in plan_model.WorkoutPlan:
            st.markdown(f"**Day: {day.Day}**")
            st.markdown(f"**Focus:** {day.Focus}")

            st.markdown("**Warm-Up:**")
            if day.WarmUp:
                for exercise in day.WarmUp:
                    st.write(f"- {exercise}")
            else:
                st.write("No warm-up specified.")

            st.markdown("**Main Exercises:**")
            if day.MainExercises:
                for ex in day.MainExercises:
                    st.write(f"- {ex.Exercise} | Sets: {ex.Sets}, Reps: {ex.Reps} | Notes: {ex.Notes}")
            else:
                st.write("No main exercises specified.")

            st.markdown("**Cool Down:**")
            if day.CoolDown:
                for exercise in day.CoolDown:
                    st.write(f"- {exercise}")
            else:
                st.write("No cool down specified.")

            if day.AdditionalNotes:
                st.markdown("**Additional Notes:**")
                st.write(day.AdditionalNotes)

            st.markdown("---")
    else:
        st.info("Fill out the form on the left and click 'Generate Workout Plan' to see your plan here.")






