# test_workout_generator.py
from WorkoutGenerator import WorkoutGenerator

def main():
    wg = WorkoutGenerator()
    
    sample_user_data = {
        "Age": "30",
        "Gender": "Male",
        "Height (cm)": "180",
        "Weight (Kg)": "75",
        "Fitness Goals": "Hypertrophy, Strength",
        "Fitness Level": "Intermediate",
        "Workout Frequency": "4",
        "Workout Duration": "90 minutes",
        "Workout Location": "Gym",
        "Equipment": "N/A",
        "Focus Areas": "Upper body, Core",
        "Cardio Preference": "Yes",
        "Daily Activity": "Moderate",
        "Injury": "No",
        "Injury Details": "N/A"
    }
    sample_user_name = "John Doe"
    
    print("Generating workout plan for testing...")
    workout_plan_model = wg.generate_workout_plan(sample_user_name, sample_user_data)
    
    if workout_plan_model:
        print("-" * 50)
        print("Generated Workout Plan (Structured):")
        wg.display_workout_plan(workout_plan_model)
    else:
        print("Failed to generate or parse the workout plan.")

if __name__ == "__main__":
    main()