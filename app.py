import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from src import utils, llm, database, prompts
from config import settings
from deep_translator import GoogleTranslator

def main():
    st.set_page_config(page_title=settings.PAGE_TITLE, layout=settings.PAGE_LAYOUT)
    st.title("🏡 Village Status Report Generation")

    # --- 1. Session State Initialization ---
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": prompts.WELCOME_PROMPT}]
    if 'detected_lang' not in st.session_state:
        st.session_state['detected_lang'] = 'en'
    if 'extracted_candidates' not in st.session_state:
        st.session_state['extracted_candidates'] = []
    if 'confirmed' not in st.session_state:
        st.session_state['confirmed'] = False
    if 'selected_village_name' not in st.session_state:
        st.session_state['selected_village_name'] = None

    # --- 2. Render Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- 3. Handle User Input ---
    # Dynamically translate the placeholder based on current session language
    input_placeholder = "Enter village name or ask a question..."
    if st.session_state['detected_lang'] == 'pa':
        input_placeholder = "ਪਿੰਡ ਦਾ ਨਾਮ ਦਰਜ ਕਰੋ ਜਾਂ ਸਵਾਲ ਪੁੱਛੋ..."

    if prompt := st.chat_input(input_placeholder):
        
        # ⬅️ NEW: Reset the report state so the old report disappears!
        st.session_state['confirmed'] = False
        st.session_state['selected_village_name'] = None
        st.session_state['extracted_candidates'] = None
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)

        with st.spinner("Analyzing..."):
            # Detect language and update the GLOBAL session state immediately
            detected_input_lang = utils.detect_lang(prompt)
            st.session_state['detected_lang'] = detected_input_lang
            lang = st.session_state['detected_lang']

            # Classify Intent using the LLM
            classification = llm.classify_and_extract(prompt)
            intent = classification.get("intent")
            village_name = classification.get("village_name")

            # ⬅️ Translate village name to English for MongoDB search if detected as Punjabi
            if village_name and lang == 'pa':
                try:
                    from deep_translator import GoogleTranslator
                    village_name = GoogleTranslator(source='pa', target='en').translate(village_name)
                except:
                    pass

            # Handle different user intents
            if intent == "help_request":
                msg = "I can help you generate status reports for villages. Just type the name of the village, e.g., 'Show me the report for Baluana'."
                if lang == 'pa': msg = utils.get_translation(msg)
                with st.chat_message("assistant"): st.info(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

            elif intent == "salutation":
                msg = "Hello! How can I assist you with village reports today?"
                if lang == 'pa': msg = utils.get_translation(msg)
                with st.chat_message("assistant"): st.success(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

            elif intent == "status_report" and village_name:
                candidates = database.search_villages_for_grid(village_name)
                
                if candidates:
                    st.session_state['extracted_candidates'] = candidates
                    msg = f"Found matches for '{village_name}'. Please select one from the table below."
                    if lang == 'pa': msg = utils.get_translation(msg)
                    with st.chat_message("assistant"): st.success(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                else:
                    msg = f"Sorry, I couldn't find a village named '{village_name}'. Please check the spelling."
                    if lang == 'pa': msg = utils.get_translation(msg)
                    with st.chat_message("assistant"): st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})

    # --- 4. AgGrid Selection Block ---
    if st.session_state.get('extracted_candidates') and not st.session_state['confirmed']:
        df = pd.DataFrame(st.session_state['extracted_candidates'])
        
        # Rename columns for a cleaner UI
        display_df = df.rename(columns={
            "village_name": "Village Name",
            "gp_name": "Gram Panchayat",
            "block_name": "Block"
        })
        
        # Configure AgGrid
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
        
        # Show Language options and Confirm button if a row is selected
        if selected is not None and len(selected) > 0:
            
            st.write("### Report Settings")
            default_idx = 1 if st.session_state['detected_lang'] == 'pa' else 0
            report_lang = st.radio(
                "Select Language for the Report:", 
                options=["English", "Punjabi (ਪੰਜਾਬੀ)"], 
                index=default_idx, # Pre-selects based on chat language
                horizontal=True
            )

            if st.button("✅ Confirm Selection & Generate Report"):
                # Save the user's language choice
                if report_lang == "Punjabi (ਪੰਜਾਬੀ)":
                    st.session_state['detected_lang'] = 'pa'
                else:
                    st.session_state['detected_lang'] = 'en'

                # Extract the selected village name (handling list vs dataframe formats)
                if isinstance(selected, pd.DataFrame):
                    st.session_state['selected_village_name'] = selected.iloc[0]['Village Name']
                else:
                    st.session_state['selected_village_name'] = selected[0]['Village Name']
                
                st.session_state['confirmed'] = True
                st.rerun() # Refresh the page to show the final report

    # --- 5. Render Final Report ---
    if st.session_state['confirmed'] and st.session_state['selected_village_name']:
        village_data = database.get_village_by_name(st.session_state['selected_village_name'])
        
        if village_data:
            # Render the UI (Text + Charts inside expanders)
            utils.render_latest_view(village_data, st.session_state['detected_lang'])
            
            # --- AI Insights Section ---
            st.divider() 
            insight_header = "🧠 AI-Generated Insights & Recommendations"
            if st.session_state['detected_lang'] == 'pa':
                insight_header = "🧠 AI-ਦੁਆਰਾ ਤਿਆਰ ਕੀਤੀਆਂ ਗਈਆਂ ਜਾਣਕਾਰੀਆਂ ਅਤੇ ਸਿਫ਼ਾਰਸ਼ਾਂ"
            st.subheader(insight_header)
            
            # Cache the AI insights in session state so it doesn't regenerate when you click Download
            cache_key = f"insights_{st.session_state['selected_village_name']}_{st.session_state['detected_lang']}"
            
            if cache_key not in st.session_state:
                with st.spinner("Analyzing village data for insights..."):
                    st.session_state[cache_key] = llm.analyze_village_data(
                        village_data=village_data, 
                        lang=st.session_state['detected_lang']
                    )
            
            # Display Insights on the Screen
            st.markdown(st.session_state[cache_key])
            
            # --- PDF Download Button ---
            # Now that insights are generated, we can pass them into the PDF!
            st.divider()
            
            pdf_bytes = utils.generate_pdf_report(
                village_data, 
                st.session_state['detected_lang'], 
                st.session_state[cache_key] # Passing the AI insights text here
            )
            
            btn_label = "📥 Download Complete PDF Report" if st.session_state['detected_lang'] != 'pa' else "📥 ਪੂਰੀ ਪੀਡੀਐਫ ਰਿਪੋਰਟ ਡਾਊਨਲੋਡ ਕਰੋ"
            st.download_button(
                label=btn_label, 
                data=pdf_bytes, 
                file_name=f"{village_data['village_name']}_report.pdf", 
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()