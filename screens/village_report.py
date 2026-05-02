import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from deep_translator import GoogleTranslator

from src import utils, llm, database, prompts
from src.config import settings

def init_village_session_states():
    """Initialize session states specific to the Village module to prevent collisions"""
    if 'village_messages' not in st.session_state:
        st.session_state['village_messages'] = [{"role": "assistant", "content": prompts.VILLAGE_WELCOME_PROMPT}]
    if 'village_candidates' not in st.session_state:
        st.session_state['village_candidates'] = []
    if 'village_confirmed' not in st.session_state:
        st.session_state['village_confirmed'] = False
    if 'village_selected_name' not in st.session_state:
        st.session_state['village_selected_name'] = None

def village_report_app():
    st.title("🏡 Village Status Report Generation")
    
    init_village_session_states()

    # --- Sidebar Language Toggle (Matches School Report) ---
    with st.sidebar:
        st.header("Settings / ਸੈਟਿੰਗਾਂ")
        lang_choice = st.radio("Language / ਭਾਸ਼ਾ", ["English", "ਪੰਜਾਬੀ"], key="village_lang_toggle")
        
        if lang_choice == "ਪੰਜਾਬੀ":
            st.session_state['detected_lang'] = 'pa'
            if not settings.PUNJABI_FONT_LOADED:
                utils.register_fonts()
        else:
            st.session_state['detected_lang'] = 'en'
    # -------------------------------------------------------

    lang = st.session_state['detected_lang']

    # --- Render Chat History ---
    for message in st.session_state['village_messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Handle User Input ---
    input_placeholder = "Enter village name or ask a question..."
    if lang == 'pa':
        input_placeholder = "ਪਿੰਡ ਦਾ ਨਾਮ ਦਰਜ ਕਰੋ ਜਾਂ ਸਵਾਲ ਪੁੱਛੋ..."

    if prompt := st.chat_input(input_placeholder):
        
        # Reset the report state so the old report disappears
        st.session_state['village_confirmed'] = False
        st.session_state['village_selected_name'] = None
        st.session_state['village_candidates'] = None
        
        st.session_state['village_messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)

        with st.spinner("Analyzing..."):
            # Classify Intent using the LLM
            classification = llm.classify_village_intent(prompt)
            intent = classification.get("intent")
            village_name = classification.get("village_name")

            # Translate village name to English for MongoDB search if detected as Punjabi
            if village_name and lang == 'pa':
                try:
                    village_name = GoogleTranslator(source='pa', target='en').translate(village_name)
                except:
                    pass

            # Handle different user intents
            if intent == "help_request":
                msg = "I can help you generate status reports for villages. Just type the name of the village, e.g., 'Show me the report for Baluana'."
                if lang == 'pa': msg = utils.get_translation(msg)
                with st.chat_message("assistant"): st.info(msg)
                st.session_state['village_messages'].append({"role": "assistant", "content": msg})

            elif intent == "salutation":
                msg = "Hello! How can I assist you with village reports today?"
                if lang == 'pa': msg = utils.get_translation(msg)
                with st.chat_message("assistant"): st.success(msg)
                st.session_state['village_messages'].append({"role": "assistant", "content": msg})

            elif intent == "status_report" and village_name:
                candidates = database.search_villages_for_grid(village_name)
                
                if candidates:
                    st.session_state['village_candidates'] = candidates
                    msg = f"Found matches for '{village_name}'. Please select one from the table below."
                    if lang == 'pa': msg = utils.get_translation(msg)
                    with st.chat_message("assistant"): st.success(msg)
                    st.session_state['village_messages'].append({"role": "assistant", "content": msg})
                else:
                    msg = f"Sorry, I couldn't find a village named '{village_name}'. Please check the spelling."
                    if lang == 'pa': msg = utils.get_translation(msg)
                    with st.chat_message("assistant"): st.error(msg)
                    st.session_state['village_messages'].append({"role": "assistant", "content": msg})

    # --- AgGrid Selection Block ---
    if st.session_state.get('village_candidates') and not st.session_state['village_confirmed']:
        df = pd.DataFrame(st.session_state['village_candidates'])
        
        display_df = df.rename(columns={
            "village_name": "Village Name",
            "gp_name": "Gram Panchayat",
            "block_name": "Block"
        })
        
        gb = GridOptionsBuilder.from_dataframe(display_df)
        gb.configure_selection('single', use_checkbox=True)
        grid_options = gb.build()

        st.write("### Select a Village:")
        grid_response = AgGrid(
            display_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            fit_columns_on_grid_load=True,
            theme='streamlit',
            key="village_grid"
        )

        selected = grid_response.get('selected_rows')
        
        if selected is not None and len(selected) > 0:
            if st.button("✅ Confirm Selection & Generate Report"):
                if isinstance(selected, pd.DataFrame):
                    st.session_state['village_selected_name'] = selected.iloc[0]['Village Name']
                else:
                    st.session_state['village_selected_name'] = selected[0]['Village Name']
                
                st.session_state['village_confirmed'] = True
                st.rerun()

    # --- Render Final Report ---
    if st.session_state['village_confirmed'] and st.session_state['village_selected_name']:
        village_data = database.get_village_by_name(st.session_state['village_selected_name'])
        
        if village_data:
            utils.render_village_view(village_data, lang)
            
            st.divider() 
            insight_header = "🧠 AI-ਦੁਆਰਾ ਤਿਆਰ ਕੀਤੀਆਂ ਗਈਆਂ ਜਾਣਕਾਰੀਆਂ ਅਤੇ ਸਿਫ਼ਾਰਸ਼ਾਂ" if lang == 'pa' else "🧠 AI-Generated Insights & Recommendations"
            st.subheader(insight_header)
            
            cache_key = f"insights_{st.session_state['village_selected_name']}_{lang}"
            
            if cache_key not in st.session_state:
                with st.spinner("Analyzing village data for insights..."):
                    st.session_state[cache_key] = llm.analyze_village_data(
                        village_data=village_data, 
                        lang=lang
                    )
            
            st.markdown(st.session_state[cache_key])
            
            st.divider()
            
            pdf_bytes = utils.generate_village_pdf(
                village_data, 
                lang, 
                st.session_state[cache_key]
            )
            
            btn_label = "📥 ਪੂਰੀ ਪੀਡੀਐਫ ਰਿਪੋਰਟ ਡਾਊਨਲੋਡ ਕਰੋ" if lang == 'pa' else "📥 Download Complete PDF Report"
            st.download_button(
                label=btn_label, 
                data=pdf_bytes, 
                file_name=f"{village_data['village_name']}_report.pdf", 
                mime="application/pdf"
            )