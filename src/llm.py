import json
import re
import time
from google import genai
from src.config import settings
from src import prompts

client = genai.Client(api_key=settings.API_KEY)

def call_gemini_api(prompt: str, retries=3):
    """
    Crash-proof API caller. Attempts the prompt up to 3 times with exponential backoff.
    If all retries fail, returns a safe fallback string instead of crashing Streamlit.
    """
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL_NAME,
                contents=[{"role": "user", "parts": [{"text": prompt}]}]
            )
            return response.text
        except Exception as e:
            err_str = str(e)
            
            # ---> ADD THIS LINE HERE <---
            print(f"🚨 Gemini API Error (Attempt {attempt}): {err_str}") 
            
            # If we haven't exhausted our retries, wait and try again
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Sleeps for 1s, then 2s
                continue
            # If all retries fail, return a safe string so the app continues running
            return "⚠️ AI Service temporarily unavailable due to high server load. Please try again."

def classify_school_intent(user_input: str):
    prompt = prompts.make_school_classify_prompt(user_input=user_input)
    llm_text = call_gemini_api(prompt)
    try:
        clean_json = re.sub(r'```json|```', '', llm_text).strip()
        data = json.loads(clean_json)
        return data, data.get("intent", "other")
    except:
        return {"intent": "other", "school_name": None, "udisecode": None, "username": None}, "other"

def classify_village_intent(user_input: str) -> dict:
    prompt = prompts.make_classify_prompt(user_input)
    response = call_gemini_api(prompt)
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        return json.loads(match.group(0)) if match else {"intent": "other", "village_names": []}
    except:
        return {"intent": "other", "village_names": []}

def llm_reply(user_input: str, intent: str) -> str:
    prompt = prompts.make_conversational_prompt(user_input, intent)
    return call_gemini_api(prompt)

def rephrase_remark(text: str) -> str:
    return re.sub(r"^\s*Output:?[\s\n]*", "", call_gemini_api(prompts.make_rephrase_prompt(text))).strip()

def improvment_suggestion(text: str) -> str:
    return call_gemini_api(prompts.make_suggestions(text)).strip()

def analyze_village_data(village_data, lang: str = 'en') -> str:
    if isinstance(village_data, list) and len(village_data) == 2:
        prompt = prompts.VILLAGE_COMPARISON_PROMPT.format(
            v1_name=village_data[0].get('village_name', 'V1'), v1_data=json.dumps(village_data[0], default=str),
            v2_name=village_data[1].get('village_name', 'V2'), v2_data=json.dumps(village_data[1], default=str)
        )
    else:
        v_data = village_data[0] if isinstance(village_data, list) else village_data
        prompt = prompts.VILLAGE_ANALYSIS_PROMPT.format(village_data=json.dumps(v_data, default=str))
    
    if lang == 'pa': 
        prompt += "\n\nIMPORTANT: Provide response entirely in Punjabi."
        
    return call_gemini_api(prompt)


# ==========================================
# TAVILY WEB SEARCH INTEGRATION
# ==========================================
def get_external_village_data(village_name: str) -> str:
    """Uses Tavily to fetch web context for a missing village."""
    from tavily import TavilyClient
    try:
        tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        # UPGRADE: A broader, more natural query that guarantees hits on Wikipedia and Village aggregator sites
        query = f'"{village_name}" village Punjab facilities amenities water schools healthcare census'
        
        # We increase the max_tokens to ensure we grab the entire webpage content
        context = tavily_client.get_search_context(query=query, search_depth="advanced", max_tokens=4000)
        return context
    except Exception as e:
        print(f"Tavily Search Error: {e}")
        return ""

def generate_report_from_web(village_name: str, web_context: str, lang: str) -> str:
    """Instructs Gemini to extract web context into strict JSON with explicit extraction hints."""
    prompt = f"""
    You are an expert AI Rural Development Data Extraction Agent. 
    Synthesize an informative overview for the village '{village_name}' in Punjab.
    
    --- WEB CONTEXT ---
    {web_context}
    --- END CONTEXT ---
    
    CRITICAL RULES FOR DATA INTEGRITY:
    1. SMART EXTRACTION: Extract exact numbers where available. If missing, summarize qualitative facts. 
       **EXTRACTION HINTS (CRITICAL):** - For Water/Sanitation, explicitly scan the text for words like: "hand pump", "tap", "well", "drinking water", "drainage", "toilet", "bath".
       - For Economy, scan for: "agriculture", "farming", "bank", "cooperative".
    2. BALANCED BULLET POINTS: 1 to 2 concise sentences maximum per point.
    3. STRICT JSON: Output ONLY a valid JSON object. Do not include markdown formatting outside the JSON values.
    4. TRANSLATION: The VALUES of the JSON must be translated into {'Punjabi' if lang == 'pa' else 'English'}.
    
    REQUIRED JSON STRUCTURE:
    {{
        "summary": "Write a professional 2-sentence overview of the village.",
        "population": "Extract exact population number here, or 'Data Unavailable'",
        "literacy_rate": "Extract exact literacy rate % here, or 'Data Unavailable'",
        "domains": {{
            "Basic Profile & Demographics": [
                "**Population Demographics:** 1-2 sentences here.", 
                "**Literacy Profile:** 1-2 sentences here.",
                "**Geographic Details:** 1-2 sentences here."
            ],
            "Health & Education": [
                "**Educational Facilities:** 1-2 sentences here.", 
                "**Healthcare Infrastructure:** 1-2 sentences here.",
                "**Veterinary Services:** 1-2 sentences here."
            ],
            "Water, Sanitation & Infrastructure": [
                "**Drinking Water Sources:** 1-2 sentences here.", 
                "**Sanitation Coverage:** 1-2 sentences here.",
                "**Connectivity:** 1-2 sentences here."
            ],
            "Governance & Economy": [
                "**Economic Structure:** 1-2 sentences here.", 
                "**Local Amenities:** 1-2 sentences here."
            ]
        }}
    }}
    """
    return call_gemini_api(prompt)