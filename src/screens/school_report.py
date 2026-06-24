import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from deep_translator import GoogleTranslator
from pathlib import Path

from src import utils, llm, prompts, constants
from src.config import settings

@st.cache_data
def load_data() -> pd.DataFrame:
    return utils.load_school_list(Path(settings.SCHOOL_LIST_PATH).expanduser())

def init_school_session_states(df):
    defaults = {
        's_messages': [{"role": "assistant", "content": prompts.SCHOOL_WELCOME_PROMPT}],
        's_candidates_df': pd.DataFrame(columns=df.columns),
        's_selected_df': pd.DataFrame(columns=df.columns),
        's_confirmed': False,
        's_lang': 'en'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def school_report_app():
    st.title("🏫 School Status Report Generation")
    df = load_data()
    init_school_session_states(df)

    lang = st.session_state['s_lang']
    for m in st.session_state['s_messages'][-20:]:
        with st.chat_message(m['role']):
            st.markdown(m['content'])

    input_placeholder = "Enter school name, UDISE code, or YL name..." if lang == 'en' else "ਸਕੂਲ ਦਾ ਨਾਮ, UDISE ਕੋਡ, ਜਾਂ YL ਦਾ ਨਾਮ ਦਰਜ ਕਰੋ..."
    
    if user_input := st.chat_input(input_placeholder):
        st.session_state['s_confirmed'] = False
        st.session_state['s_selected_df'] = pd.DataFrame(columns=df.columns)
        st.session_state['s_candidates_df'] = pd.DataFrame(columns=df.columns)
        
        st.session_state['s_messages'].append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)

        with st.spinner("Analyzing..."):
            # 1. Translate FIRST
            detected_input_lang = utils.detect_lang(user_input)
            st.session_state['s_lang'] = detected_input_lang
            
            english_input = GoogleTranslator(source='pa', target='en').translate(user_input) if detected_input_lang == 'pa' else user_input
            
            # 2. Now classify the translated input
            extracted, intent = llm.classify_school_intent(english_input)
            
            # 3. Handle intent with the translated data
            if intent == "help_request":
                # ... (rest of your existing logic remains the same)                reply = llm.llm_reply(user_input, "help_request")
                if lang == 'pa': reply = utils.get_translation(reply)
                st.session_state['s_messages'].append({"role":"assistant", "content": reply})
                with st.chat_message("assistant"): st.success(reply)
            
            elif intent == "salutation":
                reply = "Hello! How can I assist you with school reports?"
                if lang == 'pa': reply = "ਜੀ ਆਇਆਂ ਨੂੰ! ਅੱਜ ਮੈਂ ਸਕੂਲ ਦੀਆਂ ਰਿਪੋਰਟਾਂ ਵਿੱਚ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?"
                st.session_state['s_messages'].append({"role":"assistant", "content": reply})
                with st.chat_message("assistant"): st.info(reply)
                                    
            elif intent == "status_report":
                if extracted["school_name"] or extracted["udisecode"] or extracted["username"]:
                    matches_df = utils.extract_school_from_message(extracted, df).drop_duplicates(subset=['UDISE_Code'], keep='first')
                    if matches_df.empty:
                        reply = prompts.make_school_nofound_message(user_input=next((v for v in extracted.values() if v is not None), None))
                        if lang == 'pa': reply = utils.get_translation(reply)
                        st.session_state['s_messages'].append({"role": "assistant", "content": reply})
                        with st.chat_message("assistant"): st.error(reply)
                    else:
                        st.session_state['s_candidates_df'] = matches_df
                        reply = "Found matches. Please select a school below."
                        if lang == 'pa': reply = utils.get_translation(reply)
                        st.session_state['s_messages'].append({"role": "assistant", "content": reply})
                        with st.chat_message("assistant"): st.success(reply)
                else:
                    reply = llm.llm_reply(user_input, "status_report")
                    if lang == 'pa': reply = utils.get_translation(reply)
                    st.session_state['s_messages'].append({"role":"assistant", "content": reply})
                    with st.chat_message("assistant"): st.info(reply)
            else:
                reply = llm.llm_reply(user_input, "other")
                if lang == 'pa': reply = utils.get_translation(reply)
                st.session_state['s_messages'].append({"role":"assistant", "content": reply})
                with st.chat_message("assistant"): st.warning(reply) 

    candidates_df = st.session_state.get("s_candidates_df", pd.DataFrame()).reset_index(drop=True)

    if not candidates_df.empty and not st.session_state['s_confirmed']:
        st.markdown("### 📍 Select a School" if st.session_state['s_lang'] == 'en' else "### 📍 ਇੱਕ ਸਕੂਲ ਚੁਣੋ")
        gb = GridOptionsBuilder.from_dataframe(candidates_df)
        gb.configure_selection(selection_mode="single", use_checkbox=True)
        grid_response = AgGrid(candidates_df, gridOptions=gb.build(), update_mode=GridUpdateMode.SELECTION_CHANGED, theme='streamlit', height=200)

        selected_df = pd.DataFrame(grid_response.get('selected_rows', []))
        if not selected_df.empty:
            st.divider()
            report_lang = st.radio("📄 Select Report Language:", options=["English", "Punjabi (ਪੰਜਾਬੀ)"], index=1 if st.session_state['s_lang'] == 'pa' else 0, horizontal=True)
            
            if st.button("✅ Generate Report", type="primary"):
                st.session_state['s_lang'] = 'pa' if report_lang == "Punjabi (ਪੰਜਾਬੀ)" else 'en'
                st.session_state['s_selected_df'] = selected_df
                st.session_state['s_confirmed'] = True
                if st.session_state['s_lang'] == 'pa' and not settings.PUNJABI_FONT_LOADED: utils.register_fonts()
                st.rerun()

    if st.session_state.get('s_confirmed') and not st.session_state['s_selected_df'].empty:
        st.divider()
        with st.spinner("Compiling report data..."):
            long_report = utils.report_card(df)
            school_info = utils.info(df)

        row = st.session_state['s_selected_df'].iloc[0]
        code = row['UDISE_Code']
        data = df[df['UDISE_Code'] == code].reset_index(drop=True)
        
        if data.shape[0] > 1:
            view_mode = st.radio("Choose report type:" if st.session_state['s_lang'] == 'en' else "ਰਿਪੋਰਟ ਦਾ ਕਿਸਮ ਚੁਣੋ:", ["Latest assessment", "Comparative analysis"] if st.session_state['s_lang'] == 'en' else ["ਨਵੀਨਤਮ ਮੁਲਾਂਕਣ", "ਤੁਲਨਾਤਮਕ ਵਿਸ਼ਲੇਸ਼ਣ"], horizontal=True, key=f"view_{code}")
            if view_mode in ["Latest assessment", "ਨਵੀਨਤਮ ਮੁਲਾਂਕਣ"]:
                utils.render_latest_view(data, row, code, long_report, school_info, st.session_state['s_lang'])
                
                # --- FIXED: Passing row and school_info here ---
                utils.improvement_interaction(data, row, code, long_report, school_info, st.session_state['s_lang'])
            else:
                utils.render_comparative_view(data, row, code, long_report, school_info, st.session_state['s_lang'])
        else:
            utils.render_latest_view(data, row, code, long_report, school_info, st.session_state['s_lang'])
            
            # --- FIXED: Passing row and school_info here ---
            utils.improvement_interaction(data, row, code, long_report, school_info, st.session_state['s_lang'])