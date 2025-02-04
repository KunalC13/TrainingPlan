import pandas as pd
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

def load_rules(rules_file):
    with open(rules_file, "r") as file:
        rules = file.read()
    return rules

def load_user_data(csv_file):
    return pd.read_csv(csv_file)

def generate_training_plan(rules, user_data_row, llm):
    # Updated prompt template
    prompt_template = """
    You are a personal trainer creating a personalized training plan for a user. You have asked a few questions to the users to understand their goals, preferences, and constraints.
    
    The questions are as follows:
    1. What is your age?
    2. What is your gender?
    3. What is your height?
    4. What is your weight?
    5. What are your fitness goals? (e.g., lose weight, gain muscle, improve endurance)
    6. How would you rate your fitness level? (Beginner, Intermediate, Advanced)
    7. How many days a week can you commit to working out?
    8. How much time can you spend on each workout session?
    9. Where are you planning to work out? (Gym, Home, Mix) If it's mix, spread the days at both gym and home
    10. If you answered Home or Mix, what equipment do you have available?
    11. Do you have any specific areas of the body you want to focus on? (e.g., Upper body, Legs)
    12. Would you like to include cardio in your fitness program? If yes, please select the type(s) of cardio.
    13. What is your daily activity level? (e.g., Sedentary, Moderate, Active)
    14. Do you have any injury or medical condition? (Don't provide exercises that may worsen the injury)
    15. If you answered yes to the above question, please provide a short description.
    

    Based on the following rules for creating training plans:
    {rules}
    Example file
    Exercise pools (gym or home and equipment availability will be considered when creating the training plan)
    
    
    And this user's data:
    {user_data}
    
    Create a personalized training plan for the user in the following format:

    Example format of user responses for user 'John':
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
    
    Example training plan format  for user 'John':
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
    
    If there is an injury, give reasoning for the exercises you are choosing or how you have considered that injury.
    Also, give time division based on time given by the user.
    Ensure the plan is structured, clear, and tailored to the user's data and preferences.
    
    """
    user_data_text = "\n".join(f"{key}: {value if pd.notna(value) else 'Not provided'}" for key, value in user_data_row.items())

    prompt = prompt_template.format(rules=rules, user_data=user_data_text)
    
    response = llm(prompt)
    return response.strip()


def main():
    # File paths
    rules_file = "rules.txt"
    csv_file = "Training_Plan_Responses.csv"

    print("Loading rules...")
    rules = load_rules(rules_file)
    print("Loading user data...")
    user_data = load_user_data(csv_file)

    print("Initializing LLM...")
    llm = Ollama(model="llama3.2:1b")


    print("Generating training plans...")
    training_plans = []
    for _, row in user_data.iterrows():
        
        training_plan = generate_training_plan(rules, row, llm)
        training_plans.append(training_plan)
    
    user_data["Training_Plan"] = training_plans
    user_data.to_csv("user_training_plans_3.csv", index=False)
    print("Training plans saved to 'user_training_plans.csv'.")

if __name__ == "__main__":
    main()
