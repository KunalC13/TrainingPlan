import pandas as pd
import json
import os
from tqdm import tqdm  # Import tqdm for progress tracking
from langchain_ollama import OllamaLLM  # Using langchain-ollama

# Ensure output directory exists
output_dir = "plans"
os.makedirs(output_dir, exist_ok=True)

# Load user profiles from Excel
def load_user_data(file_path):
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name="Form Responses 1")
    return df

# Load expert rules
def load_rules(rules_file):
    with open(rules_file, "r", encoding="utf-8") as file:
        return file.read().strip()

# Generate a personalized training plan
def generate_training_plan(rules, user_data_row, llm):
    user_name = user_data_row.get("Name", "Unknown").strip()

    print(f"\n📝 Generating plan for: {user_name}...")

    prompt = f"""
    You are a fitness coach specializing in personalized training programs. Use the expert-designed guidelines below to create a **4-week** workout routine tailored to the user's profile. Ensure the program follows these guidelines: {rules} while addressing the user's unique attributes and fitness goals.

    **User Profile:**
    - Name: {user_name}
    - Age: {user_data_row.get("Age", "N/A")}
    - Gender: {user_data_row.get("Gender", "N/A")}
    - Height: {user_data_row.get("Height (cm)", "N/A")} cm
    - Weight: {user_data_row.get("Weight (Kg)", "N/A")} kg
    - Fitness Goal(s): {user_data_row.get("What are your fitness goals? (You can select multiple options)", "N/A")}
    - Fitness Level: {user_data_row.get("How would you rate your fitness level?", "N/A")}
    - Workout Frequency: {user_data_row.get("How many days per week are you planning to working out?", "N/A")} days per week
    - Workout Duration: {user_data_row.get("How much time can you dedicate to each workout session?", "N/A")}
    - Workout Location: {user_data_row.get("Where are you planning to work out?", "N/A")}
    - Equipment (if home): {user_data_row.get("if you answered Home or mix, what equipment do you have available?", "N/A")}
    - Focus Areas: {user_data_row.get("Do you have any specific areas of the body you want to focus on?", "N/A")}
    - Cardio Preference: {user_data_row.get("Would you like to include cardio in your fitness program?", "N/A")}
    - Daily Activity Level: {user_data_row.get("What is your daily activity?", "N/A")}
    - Injury/Medical Condition: {user_data_row.get("Do you have any injury or medical condition?", "N/A")}
    - Injury Details (if any): {user_data_row.get("If you answered yes to above question, please provide a short description.", "N/A")}

    ---

    **Your Task:**
    1. Based on the user's profile and the expert-designed guidelines, create a 4-week workout plan tailored to their needs.
    2. Include:
    - Exercise names, sets, reps, intensity (%1RM), and rest times.
    - Weekly progression details (e.g., increasing weight by 5% in Week 2).
    - Adjustments for weak points or health considerations (e.g., knee pain).
    - Optional supersets or circuits if time availability is limited.
    3. Generate the program in the following format (Training Plan overview example):
        
        Goal: General strength with hypertrophy focus.
        Frequency: 3 days per week.

        Structure:
        Day 1: Lower Body (Squat Focus)
        Day 2: Upper Body (Bench Focus)
        Day 3: Full Body (Deadlift Focus)

        Week 1: Training Details
            Day 1: Lower Body
            Main Lift:
            Back Squat: 4 sets × 6 reps × 100 kg = 2,400 kg
            Accessory Work:
            Romanian Deadlift: 3 sets × 10 reps × 70 kg = 2,100 kg
            Bulgarian Split Squat: 3 sets × 12 reps (per leg) × 20 kg = 1,440 kg
            Leg Curls: 3 sets × 15 reps × 40 kg = 1,800 kg
            Total Volume: 7,740 kg

            Day 2: Upper Body
            Main Lift:
            Bench Press: 4 sets × 6 reps × 80 kg = 1,920 kg
            Accessory Work:
            Dumbbell Incline Press: 3 sets × 12 reps × 25 kg = 1,800 kg
            Pull-Ups: 4 sets × 10 reps × Bodyweight (75 kg) = 3,000 kg
            Tricep Pushdowns: 3 sets × 15 reps × 30 kg = 1,350 kg
            Total Volume: 8,070 kg

            Day 3: Full Body
            Main Lift:
            Deadlift: 4 sets × 5 reps × 120 kg = 2,400 kg
            Accessory Work:
            Front Squat: 3 sets × 8 reps × 70 kg = 1,680 kg
            Bent-Over Row: 4 sets × 8 reps × 60 kg = 1,920 kg
            Plank Holds: 3 sets × 60 seconds (bodyweight) = Not calculated as load-based.
            Total Volume: 6,000 kg
            
            Week 1 Summary
            Day 1 Total: 7,740 kg
            Day 2 Total: 8,070 kg
            Day 3 Total: 6,000 kg
            Weekly Volume: Sum= 21,810 kg


    Ensure the program is logically structured and aligned with the user's fitness goals and preferences.
    """

    try:
        response = llm.invoke(prompt)
        if not response:
            raise ValueError("❌ No response from LLM!")  # Ensure API isn't failing silently
        
        print(f"✅ Plan generated for {user_name}!\n")
        return response.strip()
    
    except Exception as e:
        error_msg = f"⚠️ Error generating plan for {user_name}: {e}"
        print(error_msg)
        return "Error generating plan"

# Main function to execute workflow
def main():
    file_path = "User_profiles.xlsx"
    rules_file = "rules.txt"

    print("Loading rules...")
    rules = load_rules(rules_file)

    print("Loading user data...")
    user_data = load_user_data(file_path)

    print("Initializing LLM...")
    llm = OllamaLLM(model="llama3.2:latest")  # Using the correct model

    print("\n🚀 Generating training plans...\n")

    training_plans = {}
    error_logs = []

    # Use tqdm progress bar
    for _, row in tqdm(user_data.iterrows(), total=len(user_data), desc="Generating Plans", unit="user"):
        user_name = row.get("Name", "Unknown").strip()
        if user_name == "Unknown" or not user_name:
            continue  # Skip entries without a name

        training_plan = generate_training_plan(rules, row, llm)

        if training_plan != "Error generating plan":
            # Save each user's plan as a text file
            user_file = os.path.join(output_dir, f"{user_name}.txt")
            with open(user_file, "w", encoding="utf-8") as f:
                f.write(training_plan)

            training_plans[user_name] = training_plan
        else:
            error_logs.append(f"Error generating plan for {user_name}")

    # Save all training plans as a JSON file
    with open("user_training_plans.json", "w", encoding="utf-8") as json_file:
        json.dump(training_plans, json_file, indent=4)

    # Save errors (if any) in a log file
    if error_logs:
        with open("generation_errors.log", "w", encoding="utf-8") as err_file:
            err_file.write("\n".join(error_logs))
        print("\n⚠️ Errors encountered! Check 'generation_errors.log'.\n")

    print("\n✅ All training plans generated successfully! 🎯\n")

# Execute
if __name__ == "__main__":
    main()
