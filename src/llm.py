import json
import re
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import settings
from src import prompts

client = genai.Client(api_key=settings.API_KEY)

@retry(
    retry=retry_if_exception_type(genai.errors.ClientError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5)
)
def call_gemini_api(prompt: str):
    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL_NAME,
            contents=[{"role": "user", "parts": [{"text": prompt}]}]
        )
        return response.text
    except genai.errors.ClientError as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("Quota hit... retrying safely.")
            raise e 
        return f"Permanent Error: {e}"

def classify_and_extract(user_input: str) -> dict:
    prompt = prompts.make_classify_prompt(user_input)
    response = call_gemini_api(prompt)
    try:
        # Clean potential markdown wrappers
        clean_json = re.sub(r'```json|```', '', response).strip()
        return json.loads(clean_json)
    except json.JSONDecodeError:
        return {"intent": "other", "village_name": None}

def improvment_suggestion(text: str) -> str:
    prompt = prompts.make_suggestions(text)
    suggestion = call_gemini_api(prompt)
    return suggestion.strip()

def analyze_village_data(village_data: dict, lang: str = 'en') -> str:
    """Passes village data to Gemini to generate insights and solutions."""
    
    # Convert the MongoDB dictionary to a string so the LLM can read it. 
    # default=str ensures things like MongoDB ObjectIds don't crash the JSON parser.
    data_str = json.dumps(village_data, indent=2, default=str)
    
    # Format the prompt with the data
    prompt = prompts.VILLAGE_ANALYSIS_PROMPT.format(village_data=data_str)
    
    # If the user selected Punjabi, instruct the model to translate its analysis
    if lang == 'pa':
        prompt += "\n\nIMPORTANT: Please provide your entire response in the Punjabi language."
        
    try:
        # Re-use your existing Gemini API calling function here
        response = call_gemini_api(prompt) 
        return response
    except Exception as e:
        return f"⚠️ Could not generate insights at this time. Error: {e}"