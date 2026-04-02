import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=key)

model = genai.GenerativeModel("gemini-2.5-flash") 

def generate_summary(results_df):
    prompt = f'''You are a data scientist expert.
    Here are the model results:
    
    {results_df.to_string()}
    
    1. Identify the best model based on the evaluation metrics.
    2. Explain why it is best.
    3. Summarize the performance of the models'''
    
    response = model.generate_content(prompt)
    return response.text

def suggest_improvements(results_df):
    prompt = f'''You are a data scientist expert.
    Here are the model results:
    
    {results_df.to_string()}
    
    Suggest:
    - Ways to improve the model performance.
    - Better algorithms if needed.
    - Data Preprocessing techniques that could enhance the results.'''
   
    response = model.generate_content(prompt)
    return response.text