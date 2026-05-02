import streamlit as st
import io
import os
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch

from src import constants, llm
from config import settings

# --- Language & Translation ---
def detect_lang(text: str) -> str:
    try:
        lang = detect(text)
        return 'pa' if lang == 'pa' else 'en'
    except LangDetectException:
        return 'en'

def get_translation(text: str, target_lang='pa') -> str:
    if not text or text == "N/A": return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(str(text))
    except Exception:
        return text

def get_nested(data: dict, path: str):
    keys = path.split('.')
    val = data
    for key in keys:
        if isinstance(val, dict): val = val.get(key, "N/A")
        else: return "N/A"
    return val

@st.cache_data(show_spinner=False)
def cached_suggestion(text: str) -> str:
    return llm.improvment_suggestion(text)

# --- Fonts Setup ---
def setup_fonts():
    if os.path.exists(settings.PUNJABI_REGULAR_FONT_PATH) and os.path.exists(settings.PUNJABI_BOLD_FONT_PATH):
        pdfmetrics.registerFont(TTFont('PunjabiFont', settings.PUNJABI_REGULAR_FONT_PATH))
        pdfmetrics.registerFont(TTFont('PunjabiFont-Bold', settings.PUNJABI_BOLD_FONT_PATH))
        return True
    return False

# --- NEW: Upgraded PDF Generation (Platypus + Matplotlib) ---
def generate_pdf_report(village_data: dict, detected_lang: str, ai_insights: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    
    fonts_loaded = setup_fonts()
    font_reg = 'PunjabiFont' if (detected_lang == 'pa' and fonts_loaded) else 'Helvetica'
    font_bold = 'PunjabiFont-Bold' if (detected_lang == 'pa' and fonts_loaded) else 'Helvetica-Bold'
    
    # Styles for wrapping text
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontName=font_bold, fontSize=16, spaceAfter=12)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontName=font_bold, fontSize=14, spaceAfter=8, spaceBefore=12)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName=font_reg, fontSize=10, spaceAfter=6, leading=14)
    
    # 1. Document Title
    title_text = f"Village Report: {village_data.get('village_name', 'Unknown')}"
    if detected_lang == 'pa': title_text = get_translation(title_text)
    elements.append(Paragraph(title_text, title_style))
    elements.append(Spacer(1, 10))
    
    # 2. Domains and Charts
    for domain_idx, domain_title, main_score_key, metrics in constants.domains:
        disp_title = f"Domain {domain_idx}: {domain_title}"
        if detected_lang == 'pa': disp_title = f"{disp_title} / {get_translation(domain_title)}"
        elements.append(Paragraph(disp_title, h2_style))
        
        metric_names = []
        metric_values = []
        
        # Add text metrics
        for label, json_path in metrics:
            val = get_nested(village_data, json_path)
            disp_label = get_translation(label) if detected_lang == 'pa' else label
            elements.append(Paragraph(f"<b>{disp_label}:</b> {val}", body_style))
            
            try:
                num_val = float(val)
                metric_names.append(disp_label)
                metric_values.append(num_val)
            except (ValueError, TypeError):
                pass
                
        # Generate and attach Matplotlib chart for this domain
        if metric_values:
            fig, ax = plt.subplots(figsize=(6, max(2.5, len(metric_names) * 0.5)))
            ax.barh(metric_names, metric_values, color='#4682B4')
            
            # Apply Punjabi font to Matplotlib if needed
            prop = fm.FontProperties(fname=settings.PUNJABI_REGULAR_FONT_PATH) if (detected_lang == 'pa' and fonts_loaded) else None
            
            for i, v in enumerate(metric_values):
                ax.text(v, i, f" {v}", va='center', fontproperties=prop)
            
            if prop:
                ax.set_yticklabels(metric_names, fontproperties=prop)
                
            plt.tight_layout()
            
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            img_buf.seek(0)
            
            elements.append(Spacer(1, 10))
            elements.append(RLImage(img_buf, width=6*inch, height=max(2.5, len(metric_names) * 0.5)*inch))
            elements.append(Spacer(1, 10))
            
    # 3. AI Insights
    elements.append(Spacer(1, 20))
    insight_title = "AI-Generated Insights & Recommendations"
    if detected_lang == 'pa': insight_title = get_translation(insight_title)
    elements.append(Paragraph(insight_title, h2_style))
    
    # Strip markdown symbols so it renders nicely in ReportLab
    if ai_insights:
        for line in ai_insights.split('\n'):
            if line.strip():
                clean_line = line.replace('**', '').replace('#', '').strip()
                elements.append(Paragraph(clean_line, body_style))
                
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# --- NEW: Combined UI Logic (Data + Plotly Charts below each domain) ---
def render_latest_view(village_data: dict, detected_lang: str = 'en'):
    st.markdown(f"### Village: {village_data.get('village_name', 'Unknown')}")
    st.markdown(f"**GP Name:** {village_data.get('gp_name', 'N/A')} | **Block:** {village_data.get('block_name', 'N/A')}")
    st.markdown("---")

    for domain_idx, domain_title, main_score_key, metrics in constants.domains:
        display_title = f"Domain {domain_idx}: {domain_title}"
        if detected_lang == 'pa': display_title += f" / {get_translation(domain_title)}"
            
        with st.expander(display_title, expanded=True):
            metric_names = []
            metric_values = []
            
            # Print the text bullet points
            for label, json_path in metrics:
                val = get_nested(village_data, json_path)
                display_label = get_translation(label) if detected_lang == 'pa' else label
                st.markdown(f"- **{display_label}:** {val}")
                
                try:
                    num_val = float(val)
                    metric_names.append(display_label)
                    metric_values.append(num_val)
                except (ValueError, TypeError):
                    pass
            
            # Render the Plotly Chart just below the bullet points!
            if len(metric_values) > 0:
                df = pd.DataFrame({"Metric": metric_names, "Value": metric_values}).sort_values(by="Value")
                fig = px.bar(
                    df, x="Value", y="Metric", orientation='h', text="Value", 
                    color="Value", color_continuous_scale="Blues" if int(domain_idx) % 2 == 0 else "Teal"
                )
                fig.update_layout(xaxis_title="", yaxis_title="", showlegend=False, height=max(200, len(metric_names) * 45), margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig, use_container_width=True)