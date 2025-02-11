import pandas as pd
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

def load_rules(rules_file):
    with open(rules_file, "r") as file:
        rules = file.read()
    return rules

def load_user_data(csv_file):
    return pd.read_csv(csv_file)

def extract_user_profile(user_data_row, small_llm):
    """
    Uses a smaller model (llama3.1:3b) to extract a structured user profile from the raw CSV responses.
    """
    profile_prompt = """
    You are extracting structured user information from survey responses to create a fitness profile.
    
    The raw data is given as follows:
    {user_data}

    Format the data into a structured user profile like this:

    Example format:
    - Age: 35
    - Gender: Male
    - Height: 182 cm
    - Weight: 85 kg
    - Fitness Goals: Building muscle strength, Enhancing flexibility and mobility
    - Fitness Level: Beginner
    - Workout Days: 3
    - Session Time: 60 minutes
    - Workout Location: Gym
    - Equipment: 
    - Focus Areas: Legs (quadriceps, hamstrings, calves), Back (upper and lower back), Hips (hip flexors and abductors)
    - Cardio: 
    - Activity Level: Home Office (e.g., remote work or studying from home)
    - Injury: No

    Make sure the output follows this structured format and is free of unnecessary text.

    """
    user_data_text = "\n".join(f"{key}: {value if pd.notna(value) else 'Not provided'}" for key, value in user_data_row.items())
    prompt = profile_prompt.format(user_data=user_data_text)
    return small_llm(prompt).strip()

def generate_training_plan(rules, user_profile, large_llm):
    """
    Uses a larger model to generate a structured training plan based on the extracted user profile and fitness rules.
    """
    plan_prompt = """
    You are a personal trainer creating a structured fitness training plan for a user.
    
    The user's profile is:
    {user_profile}

    The training plan should follow these rules:
    {rules}

    Format the training plan like this example, this is just an example, you can use your knowledge to suggest exercises.

    -------------------------------
    | User: [Name]                 |
    |                              |
    | Day 1                        |
    | Exercises         Sets  Reps |
    | Back Squats       3     10-12|
    | Leg Press         3     10-12|
    | Single Leg Ext    3     10-12|
    |                              |
    | Day 2                        |
    | Exercises         Sets  Reps |
    | Lat Pulldown      3     10-12|
    | Bench Press       3     10-12|
    | Back Ext          3     10-12|
    |                              |
    | Day 3                        |
    | Exercises         Sets  Reps |
    | Deadlifts         3     10-12|
    | Leg Press         3     10-12|
    | Hip Abduction     2     10-12|
    -------------------------------

    If there is an injury, explain how the plan is adapted to accommodate it.
    Ensure clear time division based on user availability.
    Strictly adhere to Phase 1 of training for now, without including advanced phases.
    Make sure the plan is structured and easy to follow.
    Cool down time should not be included in the session time.
    Warm up time should be included in the session time.
    Give around 4-6 exercises based on goals and user profile, you can adjust the intensity and volume based on the user's fitness level.

    """
    prompt = plan_prompt.format(user_profile=user_profile, rules=rules)
    return large_llm(prompt).strip()

def main():
    # File paths
    rules_file = "rules.md"
    csv_file = "Responses_sample.csv"

    print("Loading rules...")
    rules = load_rules(rules_file)
    print("Loading user data...")
    user_data = load_user_data(csv_file)

    print("Initializing LLMs...")
    small_llm = Ollama(model="llama3.2:3b")
    large_llm = Ollama(model="deepseek-r1:7b")  # Replace 'xyz' with actual larger model name

    print("Generating training plans...")
    training_plans = []
    structured_profiles = []

    for _, row in user_data.iterrows():
        print(f"Processing user: {row['Name ']}")

        user_profile = extract_user_profile(row, small_llm)
        structured_profiles.append(user_profile)  
        training_plan = generate_training_plan(rules, user_profile, large_llm)
        training_plans.append(training_plan)

    output_data = user_data.copy()
    output_data["Training_Plan"] = training_plans

    # Drop any extra index or automatically added columns
    output_data.to_csv("user_training_plans_chained.csv", index=False)
    print("Training plans saved to 'user_training_plans_chained.csv'.")


if __name__ == "__main__":
    main()
