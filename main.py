import streamlit as st
from src.config import settings

# Import your screens
from screens.school_report import school_report_app
from screens.village_report import village_report_app

def main():
    st.set_page_config(page_title=settings.PAGE_TITLE, layout=settings.PAGE_LAYOUT)

    if 'app_mode' not in st.session_state:
        st.session_state['app_mode'] = 'Home'

    if st.session_state['app_mode'] == 'Home':
        st.title("Young Leaders Development Tool")
        st.markdown("Please select the report generation module you want to access:")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏫 School Status Report Generation", use_container_width=True):
                st.session_state['app_mode'] = 'School'
                st.rerun()
        with col2:
            if st.button("🏡 Village Status Report Generation", use_container_width=True):
                st.session_state['app_mode'] = 'Village'
                st.rerun()

    elif st.session_state['app_mode'] == 'School':
        if st.button("⬅️ Back to Home Menu"):
            st.session_state['app_mode'] = 'Home'
            st.rerun()
        st.markdown("---")
        school_report_app()

    elif st.session_state['app_mode'] == 'Village':
        if st.button("⬅️ Back to Home Menu"):
            st.session_state['app_mode'] = 'Home'
            st.rerun()
        st.markdown("---")
        village_report_app()

if __name__ == "__main__":
    main()