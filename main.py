import streamlit as st
from src.config import settings

# Import your screens
from src.screens.school_report import school_report_app
from src.screens.village_report import village_report_app

def main():
    st.set_page_config(page_title=settings.PAGE_TITLE, layout=settings.PAGE_LAYOUT)

    if 'app_mode' not in st.session_state:
        st.session_state['app_mode'] = 'Home'

    if st.session_state['app_mode'] == 'Home':
        # Hero Section
        st.markdown("<h1 style='text-align: center;'>Report Generation Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Select a module below to analyze infrastructure, track historical progress, and generate AI-powered insights.</p>", unsafe_allow_html=True)
        st.divider()
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            with st.container(border=True):
                st.markdown("### 🏫 School Analytics")
                st.markdown("Evaluate school environments, hygiene, and academic facilities using historical UDISE data.")
                st.markdown("""
                **Key Features:**
                - 📊 Infrastructure & Safety Tracking
                - 📈 Comparative Assessment Mode
                - 🧠 AI-Generated Improvement Suggestions
                - 📥 One-Click PDF Export
                """)
                st.write("") 
                if st.button("School Report Generation", use_container_width=True, type="primary"):
                    st.session_state['app_mode'] = 'School'
                    st.rerun()
                    
        with col2:
            with st.container(border=True):
                st.markdown("### 🏡 Village Analytics")
                st.markdown("Analyze village-level development across sanitation, water security, and local governance.")
                st.markdown("""
                **Key Features:**
                - 💧 Water Security & Sanitation Metrics
                - 👥 MGNREGA Employment Tracking
                - 📊 Multi-Village Comparison
                - 🧠 AI-Driven Strategic Insights
                """)
                st.write("") 
                if st.button("Village Report Generation", use_container_width=True, type="primary"):
                    st.session_state['app_mode'] = 'Village'
                    st.rerun()

    elif st.session_state['app_mode'] == 'School':
        if st.button("⬅️ Back to Home"):
            st.session_state['app_mode'] = 'Home'
            st.rerun()
        st.markdown("---")
        school_report_app()

    elif st.session_state['app_mode'] == 'Village':
        if st.button("⬅️ Back to Home"):
            st.session_state['app_mode'] = 'Home'
            st.rerun()
        st.markdown("---")
        village_report_app()

if __name__ == "__main__":
    main()