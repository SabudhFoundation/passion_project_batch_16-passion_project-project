import textwrap

# ==========================================
# VILLAGE REPORT PROMPTS
# ==========================================

VILLAGE_WELCOME_PROMPT = "Welcome to the Village Status Report Generator. How can I assist you today?"

def make_village_nofound_message(user_input: str) -> str:
    return f"Sorry, I couldn't find a village named '{user_input}'. Please check the spelling."

def make_village_classify_prompt(user_input: str) -> str:
    return textwrap.dedent(f"""
            You are a village status report assistant.
            Classify the user's intent as one of:
            - status_report: User wants a village's status report or wants to compare multiple villages. Extract a list of all mentioned village names into 'village_names'. 
              CRITICAL: Always return the village names transliterated into English.
            - salutation: Greeting, thanks, or polite courtesies.
            - help_request: Asking how you can help or asking about your scope.
            - other: Something else.

            Return ONLY a raw, valid JSON dictionary. Do not include markdown blocks (```json) and do not include conversational text.
            Schema: {{"intent": "status_report|salutation|help_request|other", "village_names": ["extracted_name_1", "extracted_name_2"]}}
            
            User message: {user_input}
            """).strip()

VILLAGE_ANALYSIS_PROMPT = """
You are an expert rural development analyst. Analyze the following village data.

Village Data:
{village_data}

Please provide exactly three sections:
1. **Key Insights:** 3-4 bullet points highlighting the most notable positive and negative aspects from the data.
2. **Recommended Solutions:** 2-3 highly actionable, practical solutions addressing the specific areas that need the most improvement based on the data.
3. **Conclusion:** A brief summary explicitly stating which domains the village is performing well in, and which domains require immediate focus.

Format the output clearly using Markdown. Be objective, precise, and base your analysis strictly on the provided data.
"""

VILLAGE_COMPARISON_PROMPT = """
You are an expert rural development analyst. Compare the following two villages based on their data.

Village 1 Data ({v1_name}):
{v1_data}

Village 2 Data ({v2_name}):
{v2_data}

Please provide exactly three sections:
1. **Head-to-Head Comparison:** 3-4 bullet points comparing their performance across major domains (Sanitation, Governance, Water Security, Employment). Explicitly highlight who is outperforming the other and cite the specific scores.
2. **Shared Weaknesses:** 1-2 bullet points identifying areas where BOTH villages are struggling and could benefit from a shared block-level intervention.
3. **Strategic Conclusion:** A brief summary determining which village requires more immediate administrative attention and funding based on the data.

Format the output clearly using Markdown. Be objective, precise, and base your analysis strictly on the provided data.
"""


def make_classify_prompt(user_input: str) -> str:
    return textwrap.dedent(f"""
            You are a village status report assistant.
            Classify the user's intent as one of:
            - status_report: User wants a village's status report or wants to compare multiple villages. Extract a list of all mentioned village names into 'village_names'. 
              CRITICAL: If the user just types a name (e.g. "Baluana" or "ਬਲੂਆਣਾ"), assume their intent is 'status_report' and extract that name.
              CRITICAL: If the user types in Punjabi, you MUST translate/transliterate the extracted village names back into English (e.g., "ਬਲੂਆਣਾ" -> "Baluana") in the JSON output.
            - salutation: Greeting, thanks, or polite courtesies.
            - help_request: Asking how you can help or asking about your scope.
            - other: Something else.

            Return ONLY a raw, valid JSON dictionary. Do not include markdown blocks (```json) and do not include conversational text.
            Schema: {{"intent": "status_report|salutation|help_request|other", "village_names": ["extracted_name_1", "extracted_name_2"]}}
            
            User message: {user_input}
            """).strip()


# ==========================================
# SCHOOL REPORT PROMPTS
# ==========================================

SCHOOL_WELCOME_PROMPT = "Welcome to the School Status Report Generator. How can I assist you today?"

def make_school_nofound_message(user_input: str) -> str:
    return f"Sorry, I couldn't find a school matching '{user_input}'. Please check the spelling or UDISE code."

def make_school_classify_prompt(user_input: str) -> str:
    return textwrap.dedent(f"""
            You are a school status report assistant.
            Classify the user's intent as one of:
            - status_report: User wants a school's status report. Extract the school name, UDISE code, or username (YL name).
            - salutation: Greeting, thanks, or polite courtesies.
            - help_request: Asking how you can help or asking about your scope.
            - other: Something else.

            Return ONLY a raw, valid JSON dictionary. Do not include markdown blocks (```json) and do not include conversational text.
            Schema: {{"intent": "status_report|salutation|help_request|other", "school_name": "extracted_name_or_null", "udisecode": "extracted_code_or_null", "username": "extracted_username_or_null"}}
            
            User message: {user_input}
            """).strip()

def make_rephrase_prompt(text: str) -> str:
    return textwrap.dedent(f"""
                    Paraphrase the following text into a simple, professional, and clear tone. 
                    Do NOT add any suggestions, improvements, or extra information. Just rephrase the exact meaning.
                    Text: {text}
                """).strip()

def make_suggestions(text: str) -> str:
    return textwrap.dedent(f"""
                You are an Educational Improvement Specialist.
                Provide a single sentence, clear, actionable suggestion for this school metric:
                ```{text}```
                Output ONLY the suggestion. No formatting, no extra text.
                """).strip()

# ==========================================
# GENERAL CONVERSATIONAL PROMPT
# ==========================================

def make_conversational_prompt(user_input: str, intent: str) -> str:
    return textwrap.dedent(f"""
        You are a helpful assistant for a Rural & Educational Development Dashboard.
        The user said: "{user_input}"
        Their intent was classified as: {intent}
        
        Respond politely and concisely (1-2 sentences). 
        If it's a salutation, greet them back. 
        If it's a help request, tell them you can generate School Status Reports (using UDISE code or School name) and Village Status Reports.
        If it's 'other', politely let them know you can only help with school and village reports.
    """).strip()