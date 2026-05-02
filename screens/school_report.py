import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import re
from pathlib import Path
from deep_translator import GoogleTranslator

from src.config import settings
from src import utils, llm, prompts, constants

@st.cache_data
def load_data() -> pd.DataFrame:
    schools = utils.load_school_list(Path(settings.SCHOOL_LIST_PATH).expanduser())
    return schools

def init_school_session_states(df):
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'extracted_candidates_df' not in st.session_state:
        st.session_state['extracted_candidates_df'] = pd.DataFrame(columns=df.columns)
    if 'selected_school_df' not in st.session_state:
        st.session_state['selected_school_df'] = pd.DataFrame(columns=df.columns)
    if 'last_intent' not in st.session_state:
        st.session_state['last_intent'] = None
    if 'confirmed' not in st.session_state:
        st.session_state['confirmed'] = False
    if 'detected_lang' not in st.session_state:
        st.session_state['detected_lang'] = 'en' 

def school_report_app():
    """This contains all your original logic for the School Report Generation"""
    st.title("School Status Report Generation")

    df = load_data()
    init_school_session_states(df)

    # --- NEW: Sidebar Language Toggle ---
    with st.sidebar:
        st.header("Settings / ਸੈਟਿੰਗਾਂ")
        lang_choice = st.radio("Language / ਭਾਸ਼ਾ", ["English", "ਪੰਜਾਬੀ"])
        
        if lang_choice == "ਪੰਜਾਬੀ":
            st.session_state['detected_lang'] = 'pa'
            if not settings.PUNJABI_FONT_LOADED:
                utils.register_fonts()
        else:
            st.session_state['detected_lang'] = 'en'
    # ------------------------------------

    for m in st.session_state['messages'][-20:]:
        with st.chat_message(m['role']):
            st.markdown(m['content'])

    user_input = st.chat_input("Enter school name or UDISE...")
    if user_input:
        
        # Translate to English for the LLM if Punjabi is selected in the sidebar
        if st.session_state['detected_lang'] == 'pa':
            english_input = GoogleTranslator(source='auto', target='en').translate(user_input)
        else:
            english_input = user_input

        st.session_state['confirmed'] = False
        st.session_state['messages'].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        chat_container = st.container()

        extracted, intent = llm.classify_school_intent(english_input)

        last_intent = st.session_state.get("last_intent")

        if last_intent == "status_report" :
            st.session_state['extracted_candidates_df'] = pd.DataFrame(columns=df.columns)
            st.session_state['selected_school_df'] = pd.DataFrame(columns=df.columns)
            st.session_state['confirmed'] = False

        # Store the current intent for next interaction
        st.session_state["last_intent"] = intent
   
        if intent == "help_request" or re.search(r"\bhow can you help\b|\bwhat can you do\b|\bhow can you assist\b", user_input, re.I):
            reply = llm.llm_reply(user_input, "help_request")
            if st.session_state['detected_lang'] == 'pa':
                reply = GoogleTranslator(source='auto', target='pa').translate(reply)
            if "school status" not in reply.lower():
                reply = "I can provide school status reports. Give me a school name, UDISE code, or YL name."
                if st.session_state['detected_lang'] == 'pa':
                    reply = GoogleTranslator(source='auto', target='pa').translate(reply)
            with chat_container:
                st.session_state['messages'].append({"role":"assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.success(reply)
        
        elif intent == "salutation":
            reply = llm.llm_reply(user_input, "salutation")
            if st.session_state['detected_lang'] == 'pa':
                reply = GoogleTranslator(source='auto', target='pa').translate(reply)
            with chat_container:
                st.session_state['messages'].append({"role":"assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.info(reply)
                                
        elif intent == "school_list" or re.match(r"\bschool list\b|\bwhole list\b", user_input, re.I):
            response_text = "Here is the complete list schools I can provide details for."
            if st.session_state['detected_lang'] == 'pa':
                response_text = GoogleTranslator(source='auto', target='pa').translate(response_text)
            with chat_container:
                st.session_state['messages'].append({"role":"assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)

            school_list = df[['School_Name','District', 'Block']]
            st.dataframe(school_list.head(10))
            csv_data = school_list.to_csv(index = False).encode('utf-8')
            if st.session_state['detected_lang'] == 'pa':
                download = "ਪੂਰੀ ਸਕੂਲ ਸੂਚੀ ਡਾਊਨਲੋਡ ਕਰੋ"
            else:
                download = "📥 Download Full School List"
                st.download_button(
                    label = download,
                    data = csv_data,
                    file_name = "school_list.csv",
                    mime = "text/csv"
                )

        elif intent == "status_report":
            if extracted["school_name"] or extracted["udisecode"] or extracted["username"]:
                # --- MATCHING FUNCTION ---
                matches_df = utils.extract_school_from_message(extracted,df)
                matches_df = matches_df.drop_duplicates(subset=['UDISE_Code'], keep='first')

                st.session_state['extracted_candidates_df'] = matches_df

                if matches_df.empty:
                    response_text = prompts.make_school_nofound_message(user_input = next((v for v in extracted.values() if v is not None), None))
                    if st.session_state['detected_lang'] == 'pa':
                        response_text = GoogleTranslator(source='auto', target='pa').translate(response_text)
                else:
                    response_text = f"Found {matches_df.shape[0]} matching rows. Please select from the table below."
                    if st.session_state['detected_lang'] == 'pa':
                        response_text = GoogleTranslator(source='auto', target='pa').translate(response_text)

                with chat_container:
                    st.session_state['messages'].append({"role": "assistant", "content": response_text})
                    with st.chat_message("assistant"):
                        st.markdown(response_text)

                # --- BOT RESPONSE ---
                if matches_df.empty:
                    school_list = df[['School_Name','District', 'Block']]
                    st.dataframe(school_list.head(10))
                    csv_data = school_list.to_csv(index = False).encode('utf-8')
                    if st.session_state['detected_lang'] == 'pa':
                        download = "ਪੂਰੀ ਸਕੂਲ ਸੂਚੀ ਡਾਊਨਲੋਡ ਕਰੋ"
                    else:
                        download = "📥 Download Full School List"

                    st.download_button(
                        label = download,
                        data = csv_data,
                        file_name = "school_list.csv",
                        mime = "text/csv"
                    )
            else:
                reply = llm.llm_reply(user_input, "status_report")
                if st.session_state['detected_lang'] == 'pa':
                    reply = GoogleTranslator(source='auto', target='pa').translate(reply)
                with chat_container:
                    st.session_state['messages'].append({"role":"assistant", "content": reply})
                    with st.chat_message("assistant"):
                        st.info(reply)
        else:
            reply = llm.llm_reply(user_input, "other")
            if st.session_state['detected_lang'] == 'pa':
                reply = GoogleTranslator(source='auto', target='pa').translate(reply)
            with chat_container:
                st.session_state['messages'].append({"role":"assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.warning(reply) 

    # --- DISPLAY MATCHES WITH AGGRID ---
    candidates_df = st.session_state.get("extracted_candidates_df", pd.DataFrame()).reset_index(drop=True)
    detected_lang = st.session_state['detected_lang']
    

    if not candidates_df.empty:
        if detected_lang == 'pa':
            st.subheader("ਰਿਕਾਰਡ ਮਿਲੇ ਹਨ")
        else:
            st.subheader("Matches found")

        gb = GridOptionsBuilder.from_dataframe(candidates_df)
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)
        gb.configure_grid_options(domLayout='normal')
        grid_options = gb.build()

        grid_response = AgGrid(
            candidates_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            fit_columns_on_grid_load=True,
            enable_enterprise_modules=False,
            height=350,
            allow_unsafe_jscode=False,
        )

        selected_df = pd.DataFrame(grid_response.get('selected_rows', []))
        st.session_state['selected_school_df'] = selected_df

        if detected_lang == 'pa':
            st.markdown("ਚੁਣੀ ਗਈ ਝਲਕ:")
        else:
            st.markdown("**Selected preview:**")
        if not selected_df.empty:
            st.dataframe(selected_df.reset_index(drop=True), use_container_width=True)

            col1, col2 = st.columns([1,1])
            with col1:
                if detected_lang == 'pa':
                    Confirm = st.button("ਚੋਣ ਦੀ ਪੁਸ਼ਟੀ ਕਰੋ")
                else:
                    Confirm = st.button("Confirm selection")
                if Confirm:
                    st.session_state['confirmed'] = True
                    if detected_lang == 'pa':
                        st.success("ਚੋਣ ਦੀ ਪੁਸ਼ਟੀ ਹੋਈ। ਹੇਠਾਂ ਅੰਤਿਮ ਚੋਣ ਵੇਖੋ।")
                    else:
                        st.success("Selection confirmed. See final selection below.")
                    st.rerun()
            with col2:
                if detected_lang == 'pa':
                    Edit = st.button("ਚੋਣ ਦਾ ਸੰਪਾਦਨ ਕਰੋ")
                else:
                    Edit = st.button("Edit selection")
                if Edit:
                    st.session_state['selected_school_df'] = pd.DataFrame(columns=selected_df.columns)
                    st.session_state['confirmed'] = False
                    if detected_lang == 'pa':
                        st.info(GoogleTranslator(source='auto', target='pa').translate("ਚੋਣ ਕਲੀਅਰ ਕੀਤੀ ਗਈ ਹੈ — ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਚੁਣੋ।"))
                    else:
                        st.info("Selection cleared — please select again.")
                    st.rerun()

    if st.session_state.get('confirmed') and not st.session_state['selected_school_df'].empty:
        st.markdown("---")
        final_df = st.session_state['selected_school_df']

        with st.chat_message("assistant"):
            if detected_lang == 'pa':
                st.markdown("ਇਹ ਤੁਹਾਡੇ ਚੁਣੇ ਹੋਏ ਵਿਕਲਪ ਹਨ। ਜੇਕਰ ਤੁਸੀਂ ਕੋਈ ਬਦਲਾਅ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ ਤਾਂ ਉੱਪਰ 'ਚੋਣ ਦਾ ਸੰਪਾਦਨ ਕਰੋ' ਤੇ ਕਲਿੱਕ ਕਰੋ।")
            else:
                st.markdown("These are your selected options. If you would like to do any changes then click 'Edit selection' above.")

        long_report = utils.report_card(df)
        school_info = utils.info(df)

        tabs = st.tabs([row['School_Name'].upper() for _, row in final_df.iterrows()])

        for (_, row), tab in zip(final_df.iterrows(), tabs):
            with tab:
                code = row['UDISE_Code']
                data = df[df['UDISE_Code'] == code].reset_index(drop=True)
                if data.shape[0]>1:
                    if detected_lang == 'pa':
                        view_mode = st.radio("ਰਿਪੋਰਟ ਦਾ ਕਿਸਮ ਚੁਣੋ", ["ਨਵੀਨਤਮ ਮੁਲਾਂਕਣ", "ਤੁਲਨਾਤਮਕ ਵਿਸ਼ਲੇਸ਼ਣ"], horizontal=True, key=f"view_{code}")
                    else:
                        view_mode = st.radio("Choose report type:", ["Latest assessment", "Comparative analysis"], horizontal=True, key=f"view_{code}")
                    if view_mode == "Latest assessment" or view_mode == "ਨਵੀਨਤਮ ਮੁਲਾਂਕਣ":
                        utils.render_latest_view(data, row, code, long_report, school_info, detected_lang)
                        utils.improvement_interaction(data, code, long_report, detected_lang)

                    else:
                        utils.render_comparative_view(data, row, code, long_report, school_info, detected_lang)
                else:
                    utils.render_latest_view(data, row, code, long_report, school_info, detected_lang)
                    utils.improvement_interaction(data, code, long_report, detected_lang)