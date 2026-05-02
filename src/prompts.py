import textwrap

WELCOME_PROMPT = "Welcome to the Village Status Report Generator. How can I assist you today?"

def make_nofound_message(user_input: str) -> str:
    return f"Sorry, I couldn't find a village named '{user_input}'. Please check the spelling."

def make_classify_prompt(user_input: str) -> str:
    return textwrap.dedent(f"""
            You are a village status report assistant.
            Classify the user's intent as one of:
            - status_report: User wants a village's status report. Extract village_name.
            - salutation: Greeting, thanks, or polite courtesies.
            - help_request: Asking how you can help or asking about your scope.
            - other: Something else.

            Return ONLY valid JSON:
            {{"intent": "status_report|salutation|help_request|other", "village_name": "extracted_name_or_null"}}
            
            User message: {user_input}
            """).strip()

def make_suggestions(text: str) -> str:
    return textwrap.dedent(f"""
                You are a Rural Development Specialist.
                Provide a single sentence, clear, actionable suggestion for this village metric:
                ```{text}```
                Output ONLY the suggestion. No formatting, no extra text.
                """).strip()


VILLAGE_ANALYSIS_PROMPT = """
You are an expert rural development analyst. Analyze the following village data and provide a structured evaluation.

Village Data:
{village_data}

Please provide exactly three sections:
1. **Key Insights:** 3-4 bullet points highlighting the most notable positive and negative aspects from the data.
2. **Recommended Solutions:** 2-3 highly actionable, practical solutions addressing the specific areas that need the most improvement based on the data.
3. **Conclusion:** A brief summary explicitly stating which domains the village is performing well in, and which domains require immediate focus.

Format the output clearly using Markdown. Be objective, precise, and base your analysis strictly on the provided data.
"""