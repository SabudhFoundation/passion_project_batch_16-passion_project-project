import json
import ast
import re
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import settings
from src import prompts

client = genai.Client(api_key=settings.API_KEY)

@retry(
    retry=retry_if_exception_type(genai.errors.ClientError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5)
)
def call_gemini_api(prompt: str):
    """Unified API caller with exponential backoff for quota limits."""
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

# ==========================================
# SCHOOL ROUTING & EXTRACTION
# ==========================================

def classify_school_intent(user_input: str):
    """Used strictly for the School Report UI. Returns a tuple: (extracted_dict, category_string)"""
    prompt = prompts.make_school_classify_prompt(user_input=user_input)
    llm_text = call_gemini_api(prompt)
    
    try:
        data = json.loads(llm_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", llm_text, re.DOTALL)
        if match:
            json_str = match.group()
            try:
                data = json.loads(json_str) 
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(json_str)
                except (ValueError, SyntaxError):
                    data = {"category": "other"}  
        else:
            data = {"category": "other"}

    data = {**{"category": "other", "username": None, "school_name": None, "udisecode": None}, **data}

    extracted = {
        "username": data.get("username"),
        "school_name": data.get("school_name"),
        "udisecode": data.get("udisecode"),
    }
    category = data.get("category", "other")

    return extracted, category

def llm_reply(user_input: str, intent: str) -> str:
    """General chatter routing for School UI."""
    if intent == "help_request":
        prompt = prompts.make_school_help_prompt(user_input=user_input)
    elif intent == "salutation":
        prompt = prompts.make_school_salutation_prompt(user_input=user_input)
    elif intent == "status_report":
        prompt = prompts.make_school_report_prompt(user_input=user_input)
    else:
        if re.match(r"^\s*(yes|ok|okay|sure|yep|yeah)\b", user_input.strip(), re.I):
            return "Sorry, I didn’t catch your intent. Could you please clarify or provide a school name or UDISE code?"
        prompt = prompts.make_school_fallback_prompt(user_input=user_input)

    return call_gemini_api(prompt)

def rephrase_remark(text: str, max_retries: int = settings.MAX_REPHRASE_RETRIES) -> str:
    """School report specific remark rephrasing."""
    paraphrase_prompt = prompts.make_school_rephrase_prompt(text=text)

    for attempt in range(max_retries):
        try:
            rephrased = call_gemini_api(paraphrase_prompt)
        except Exception:
            return text

        cleaned = re.sub(r"^\s*Here\s(is|’s|’re| are).*?:\s*", "", rephrased, flags=re.I)
        cleaned = re.sub(r"^\s*Rephrased message:?[\s\n]*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"^\s*Output:?[\s\n]*", "", cleaned, flags=re.I)

        return cleaned.strip() if cleaned.strip() else text

def improvment_suggestion(text: str) -> str:
    """Used by School Report to generate domain improvement suggestions."""
    improvment_prompt = prompts.make_school_suggestions(text=text)
    suggestion = call_gemini_api(improvment_prompt)

    cleaned_suggestion = re.sub(r"^\s*Here\s(is|’s|’re| are).*?:\s*", "", suggestion, flags=re.I)
    cleaned_suggestion = re.sub(r"^\s*improvement\s(message|suggestion):?[\s\n]*", "", cleaned_suggestion, flags=re.I)
    cleaned_suggestion = re.sub(r"^\s*Output:?[\s\n]*", "", cleaned_suggestion, flags=re.I)

    return cleaned_suggestion.strip()

# ==========================================
# VILLAGE ROUTING & EXTRACTION
# ==========================================

def classify_village_intent(user_input: str) -> dict:
    """Used strictly for the Village Report UI. Returns a dict: {'intent': str, 'village_name': str}"""
    prompt = prompts.make_village_classify_prompt(user_input)
    response = call_gemini_api(prompt)
    try:
        clean_json = re.sub(r'```json|```', '', response).strip()
        return json.loads(clean_json)
    except json.JSONDecodeError:
        return {"intent": "other", "village_name": None}

def analyze_village_data(village_data: dict, lang: str = 'en') -> str:
    """Generates AI insights for the Village Report."""
    data_str = json.dumps(village_data, indent=2, default=str)
    prompt = prompts.VILLAGE_ANALYSIS_PROMPT.format(village_data=data_str)
    
    if lang == 'pa':
        prompt += "\n\nIMPORTANT: Please provide your entire response in the Punjabi language."
        
    try:
        return call_gemini_api(prompt) 
    except Exception as e:
        return f"⚠️ Could not generate insights at this time. Error: {e}"