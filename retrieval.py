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
    prompt_template = """
    Based on the following rules for creating training plans:
    {rules}
    
    And this user's data:
    {user_data}
    
    Create a personalized training plan for the user.
    """
    user_data_text = "\n".join(f"{key}: {value}" for key, value in user_data_row.items())
    prompt = prompt_template.format(rules=rules, user_data=user_data_text)
    
    response = llm(prompt)
    return response.strip()

def main():
    # File paths
    rules_file = "rules.txt"
    csv_file = "qa.csv"

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
    user_data.to_csv("user_training_plans.csv", index=False)
    print("Training plans saved to 'user_training_plans.csv'.")

if __name__ == "__main__":
    main()
