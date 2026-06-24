import re
import io
import os
import time
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
import altair as alt
import plotly.express as px
import streamlit as st
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image as RLImage
from reportlab.lib.units import inch

from src import llm, constants
from src.config import settings


# ==========================================
# 1. LANGUAGE & BASE HELPERS
# ==========================================

# ===================================
# CACHING WRAPPERS
# =================================== 

@st.cache_data(show_spinner=False)
def cached_rephrase(val):
    return llm.rephrase_remark(val)

@st.cache_data(show_spinner=False)
def cached_suggestion(val):
    # Note: 'improvment_suggestion' is spelled without the 'e' here 
    # because that is exactly how it is spelled in your llm.py file.
    return llm.improvment_suggestion(val)

def detect_lang(text: str) -> str:
    try:
        lang = detect(text)
        return 'pa' if lang == 'pa' else 'en'
    except LangDetectException:
        return 'en'

@st.cache_data(show_spinner=False)
def get_translation(text: str, target_lang='pa') -> str:
    if not text or text == "N/A": 
        return text
    
    if target_lang == 'pa':
        if hasattr(constants, 'PUNJABI_LABELS') and text in constants.PUNJABI_LABELS:
            return constants.PUNJABI_LABELS[text]
        try:
            return GoogleTranslator(source='auto', target=target_lang).translate(str(text))
        except Exception:
            return text
            
    return text

def get_translation_batch(text_list, dest='pa'):
    if not text_list:
        return []
    try:
        clean_list = [str(t) for t in text_list]
        return GoogleTranslator(source='en', target=dest).translate_batch(clean_list)
    except Exception:
        return text_list

def get_nested(data: dict, path: str):
    keys = path.split('.')
    val = data
    for key in keys:
        if isinstance(val, dict): val = val.get(key, "N/A")
        else: return "N/A"
    return val

def register_fonts():
    """The Ultimate Font Locator with UI Debugging."""
    import os
    import streamlit as st
    
    try:
        # 1. Determine absolute root directory safely
        current_dir = os.path.dirname(os.path.abspath(__file__)) # This is src/
        root_dir = os.path.dirname(current_dir) # This is the main project folder
        
        # 2. Strip any accidental quotes from the .env variables
        env_reg = str(settings.PUNJABI_REGULAR_FONT_PATH).strip("'\" ")
        env_bold = str(settings.PUNJABI_BOLD_FONT_PATH).strip("'\" ")
        
        # 3. Construct the absolute paths
        reg_path = env_reg if os.path.isabs(env_reg) else os.path.join(root_dir, env_reg)
        bold_path = env_bold if os.path.isabs(env_bold) else os.path.join(root_dir, env_bold)
        
        # Normalize paths for Windows (changes / to \)
        reg_path = os.path.normpath(reg_path)
        bold_path = os.path.normpath(bold_path)

        if os.path.exists(reg_path) and os.path.exists(bold_path):
            # Register with ReportLab
            pdfmetrics.registerFont(TTFont('Gurmukhi', reg_path))
            pdfmetrics.registerFont(TTFont('Gurmukhi-Bold', bold_path))
            pdfmetrics.registerFont(TTFont('PunjabiFont', reg_path))
            pdfmetrics.registerFont(TTFont('PunjabiFont-Bold', bold_path))
            
            pdfmetrics.registerFontFamily('Gurmukhi', normal='Gurmukhi', bold='Gurmukhi-Bold', italic='Gurmukhi', boldItalic='Gurmukhi-Bold')
            
            # Store the successful absolute paths back in settings
            settings.PUNJABI_REGULAR_FONT_PATH = reg_path
            settings.PUNJABI_BOLD_FONT_PATH = bold_path
            settings.PUNJABI_FONT_LOADED = True
            return True
        else:
            # FIRE ALARM: Print a massive red error on the UI telling the user exactly what is wrong
            st.error(f"🚨 **FONT MISSING!** The PDF generator cannot find your Punjabi fonts.\n\nIt is looking exactly here:\n`{reg_path}`\n\nPlease check your `.env` file and make sure the file exists at that exact location!")
            settings.PUNJABI_FONT_LOADED = False
            return False
            
    except Exception as e:
        st.error(f"🚨 Font Registration Crash: {e}")
        settings.PUNJABI_FONT_LOADED = False
        return False

# ==========================================
# 2. DATA LOADING & MATCHING (SCHOOLS)
# ==========================================

def load_school_list(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str)
    df['School_Name'] = df['School_Name'].str.lower().str.strip()
    df['YL_Name'] = df['YL_Name'].str.lower().str.strip()
    df['UDISE_Code'] = df['UDISE_Code'].astype(str).str.zfill(11) 
    return df

def norm_text(s: Optional[str]) -> str:
    text = (s or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'\bschool\b', '', text)
    text = re.sub(r'(?<=\w)\sanjhisikhiya(?=\w)', '', text)
    text = re.sub(r"(?<=\w)org(?=\w)", "", text)
    return text.strip()

def generic_fuzzy_match(text: str, df: pd.DataFrame, col_name: str, top_n: int = 20) -> pd.DataFrame:
    query = norm_text(text)
    clean_query = re.sub(r'(?<=\w)\.(?=\w)', '', query) if col_name == 'School_Name' else re.split(r'[.@_]', query, 1)[0]
    
    candidates = df[df[col_name].str.contains(clean_query, na=False)]
    if not candidates.empty:
        return candidates[['District','Block','School_Name','UDISE_Code', 'YL_Name']]

    choices = df[col_name].tolist()
    results = process.extract(text, choices, scorer=fuzz.WRatio, limit=top_n)
    candidates_idx = [idx for _, score, idx in results if score > settings.FUZZY_SCORE_THRESHOLD]

    if candidates_idx:
        return df.iloc[candidates_idx][['District','Block','School_Name','UDISE_Code', 'YL_Name']]
    
    return pd.DataFrame(columns=['District','Block','School_Name','UDISE_Code','YL_Name'])

def extract_school_from_message(report_json, df: pd.DataFrame):
    if report_json.get("udisecode"):
        matched = df[df['UDISE_Code'] == report_json["udisecode"]]
        if not matched.empty: return matched[['District','Block','School_Name','UDISE_Code']]

    if report_json.get("school_name"):
        res = generic_fuzzy_match(report_json["school_name"], df, 'School_Name')
        if not res.empty: return res

    if report_json.get("username"):
        res = generic_fuzzy_match(report_json["username"], df, 'YL_Name')
        if not res.empty: return res
    
    return pd.DataFrame(columns=['District','Block','School_Name','UDISE_Code'])

# ==========================================
# 3. REPORT DATA PROCESSING (SCHOOLS)
# ==========================================

@st.cache_data
def report_card(school_data: pd.DataFrame) -> pd.DataFrame:
    report_card_df = pd.DataFrame(school_data[["School_Name", "YL_Name", "UDISE_Code", "Assessment_Date"]])
    
    mappings = {
        "Safety and Hygiene / Toilets": ("Infra_Toilets", constants.mapping_dict_toilets),
        "Safety and Hygiene / Handwash Facilities": ("Infra_Handwash", constants.mapping_dict_handwash),
        "Safety and Hygiene / Drinking Water": ("Infra_DrinkingWater", constants.mapping_dict_Water),
        "Safety and Hygiene / Mid day meal": ("Infra_MiddayMeal", constants.mapping_dict_meal),
        "Safety and Hygiene / School Building": ("Infra_SchoolBuilding", constants.mapping_dict_schoolBuilding),
        "Safety and Hygiene / Safe Surrounding": ("Infra_SafeSurroundings", constants.mapping_dict_safeSurrounding),
        "Stimulating School Environment/ Classroom Resources": ("Env_ClassroomResources", constants.mapping_dict_ClassRoom),
        "Stimulating School Environment/ Wall Painting": ("Env_WallPainting", constants.mapping_dict_Wall),
        "Stimulating School Environment/ Print Rich Classrooms": ("Env_PrintRich", constants.mapping_dict_print),
        "Stimulating School Environment/ Green Premises": ("Env_GreenPremises", constants.mapping_dict_plant),
        "Physical Development Opportunities/ Playground": ("Physical_Playground", constants.mapping_dict_playgroud),
        "Physical Development Opportunities/ Sports Equipment": ("Physical_SportsEquipment", constants.mapping_dict_sports),
        "Physical Development Opportunities/ Other Physical Activity Spaces": ("Physical_OtherSpaces", constants.mapping_dict_Otherspace),
        "Smart School Facilities/Library": ("Smart_Library", constants.mapping_dict_Library),
        "Smart School Facilities/ Digital Learning Resources": ("Smart_DigitalResources", constants.mapping_dict_digital),
        "Smart School Facilities/Education Park": ("Smart_EducationPark", constants.mapping_dict_Park),
        "Smart School Facilities/Centre Resources": ("TRC_Resources", constants.mapping_dict_trc),
    }

    for target, (src, mapper) in mappings.items():
        if src in school_data.columns:
            report_card_df[target] = school_data[src].map(mapper)

    col2domain = {}
    domain_map = {
        "Safety and Hygiene": constants.safety_cols,
        "Stimulating School Environment": constants.stim_cols,
        "Physical Development Opportunities": constants.physical_cols,
        "Smart School Facilities": constants.smart_cols
    }

    for d_name, cols in domain_map.items():
        for c in cols:
            if c in report_card_df.columns: col2domain[c] = (d_name, c.split("/",1)[-1].strip())

    long_reportcard = report_card_df.melt(
        id_vars=['School_Name', 'YL_Name', 'UDISE_Code', 'Assessment_Date'],
        value_vars=list(col2domain.keys()),
        var_name="DomainFeatureCol",
        value_name="Domain Score"
    )

    long_reportcard[['Domain', 'Feature']] = long_reportcard['DomainFeatureCol'].apply(lambda x: pd.Series(col2domain[x]))
    long_reportcard['Score Percentage'] = ((long_reportcard['Domain Score']-1)*100/3).round(0).astype(str) + "%"
    
    return long_reportcard.drop(columns=['DomainFeatureCol'])

def pivot_table(udise_code: str, long_reportcard: pd.DataFrame, date) -> pd.DataFrame:
    report_one = long_reportcard[(long_reportcard['UDISE_Code'] == udise_code) & (long_reportcard['Assessment_Date'] == date)]
    dfs = []
    for idx, d_name, *rest in constants.domains:
        d_data = report_one[report_one['Domain'] == d_name]
        if d_data.empty: continue
        score = d_data.pivot(index='Feature', columns=['Domain'], values=['Domain Score'])
        perc = d_data.pivot(index='Feature', columns=['Domain'], values=['Score Percentage'])
        combined = pd.concat([score, perc], axis=1)
        stacked = combined.stack(level='Domain', future_stack=True).reset_index().drop('Domain', axis=1)
        stacked.rename(columns={"Feature": d_name, 'Domain Score': f'Domain{idx} Score', 'Score Percentage': f'Domain{idx} Score Percentage'}, inplace=True)
        dfs.append(stacked)

    return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

def overall_score(report_df: pd.DataFrame) -> pd.DataFrame:
    domains = ['Domain1 Score Percentage', 'Domain2 Score Percentage', 'Domain3 Score Percentage', 'Domain4 Score Percentage']
    domain_names = ['Safety and Hygiene', 'Stimulating School Environment', 'Physical Development Opportunities', 'Smart School Facilities']
    values = []
    for col in domains:
        if col in report_df.columns:
            s = report_df[col].astype(str).str.rstrip('%').replace('', np.nan)
            s = pd.to_numeric(s, errors='coerce')
            mean = s.mean()
            values.append(f"{mean:.2f}%" if not pd.isna(mean) else "N/A")
        else:
            values.append("N/A")
    return pd.DataFrame({'Domain': domain_names, 'Score Percentage': values})

def info(df: pd.DataFrame) -> pd.DataFrame:
    df[constants.object_to_int_col] = df[constants.object_to_int_col].apply(lambda x: pd.to_numeric(x, errors="coerce")).fillna(0).astype(int)
    school_info = df.reindex(columns=constants.School_info_cols).copy()
    school_info["Total_girls"] = sum(df[f"Students_{g}_Girls"] for g in ["PrePrimary","Grade1","Grade2","Grade3","Grade4","Grade5"])
    school_info["Total_boys"] = sum(df[f"Students_{g}_Boys"] for g in ["PrePrimary","Grade1","Grade2","Grade3","Grade4","Grade5"])
    school_info["Total_student"] = np.where((school_info["Total_girls"] == 0) & (school_info["Total_boys"] == 0), school_info["Students_Total_GPS"], school_info["Total_girls"] + school_info["Total_boys"]).astype(int)
    return school_info

def info_para(school_info: pd.DataFrame) -> str:
    girls_pct = round((school_info['Total_girls'].iloc[0] / school_info['Total_student'].iloc[0] * 100), 1) if school_info['Total_girls'].iloc[0] else 0
    teacher_fill_pct = round((school_info['Teachers_Present'].iloc[0] / school_info['Teacher_Positions_Sanctioned'].iloc[0] * 100), 1) if school_info['Teacher_Positions_Sanctioned'].iloc[0] else 0

    parts = []
    if school_info["Total_student"].iloc[0]:
        stud_part = f"The school has total of {school_info['Total_student'].iloc[0]:,} students"
        extras = []
        if school_info['Total_girls'].iloc[0]: extras.append(f"including {school_info['Total_girls'].iloc[0]:,} girls ({girls_pct}% )")
        if school_info['Total_boys'].iloc[0]: extras.append(f"and {school_info['Total_boys'].iloc[0]:,} boys")
        if extras: stud_part += " — " + ", ".join(extras)
        stud_part += "."
        parts.append(stud_part)

    if school_info['Teacher_Positions_Sanctioned'].iloc[0]:
        teacher_part = f"There are {school_info['Teacher_Positions_Sanctioned'].iloc[0]:,} sanctioned teacher positions"
        subparts = []
        if school_info['Teachers_Present'].iloc[0]:
            subparts.append(f"out of which {school_info['Teachers_Present'].iloc[0]:,} are currently filled ({teacher_fill_pct}% filled)")
        if school_info['Teachers_Deputation'].iloc[0]:
            subparts.append(f"{school_info['Teachers_Deputation'].iloc[0]:,} are on deputation")
        if school_info['Teachers_New_Recruits'].iloc[0]:
            subparts.append(f"{school_info['Teachers_New_Recruits'].iloc[0]:,} new teachers are recruited")
        if subparts: teacher_part += " , " + "; ".join(subparts)
        teacher_part += "."
        parts.append(teacher_part)

    child_parts = []
    if school_info['Students_Disability_Count'].iloc[0]: child_parts.append(f"There are {school_info['Students_Disability_Count'].iloc[0]:,} students with disabilities")
    if school_info['Children_PrivateSchool'].iloc[0]: child_parts.append(f"{school_info['Children_PrivateSchool'].iloc[0]:,} children attend private schools")
    if school_info['Children_Anganwadi_0_3'].iloc[0]: child_parts.append(f"{school_info['Children_Anganwadi_0_3'].iloc[0]:,} children aged 0-3 are enrolled in Anganwadi centers")
    if child_parts: parts.append(", and ".join(child_parts) + ".")

    paragraph = " ".join(parts).strip()
    return paragraph if paragraph else "No student/teacher details available."

# ==========================================
# 4. CHARTS & PLOTTING 
# ==========================================  

def bar_graph_pdf(x, y, ylabel, colour, title: str = None, ylim: int=None, font_path=None):
    prop = FontProperties(fname=font_path) if font_path else None
    fig, ax = plt.subplots()
    ax.bar(x, y, color=colour)
    if prop: ax.set_ylabel(ylabel, fontproperties=prop, fontsize=10)
    else: ax.set_ylabel(ylabel)
    if ylim is not None: ax.set_ylim(0, ylim)
    if title:
        if prop: ax.set_title(title, fontproperties=prop, fontsize=14)
        else: ax.set_title(title, fontsize=17, fontweight='normal')
    plt.xticks(rotation=45, ha='right')
    if prop:
        for label in ax.get_xticklabels(): label.set_fontproperties(prop)

    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def teacher_pie_buf(present, sanctioned, title, font_path=None):
    prop = FontProperties(fname=font_path) if font_path else None
    fig, ax = plt.subplots(figsize=(3.2, 1.8))
    absent = max(sanctioned - present, 0)
    wedges, texts = ax.pie([present, absent], colors=["#2ca02c", (0, 0, 0, 0)], pctdistance=0.75, labeldistance=1.15)
    centre_circle = plt.Circle((0, 0), 0.60, color='white', fc='white', linewidth=0)
    ax.add_artist(centre_circle)
    if prop: ax.set_title(title, fontproperties=prop, fontsize=11, pad=5)
    else: ax.set_title(title, fontsize=11, fontweight='normal', pad=5)
    for t in texts: t.set_visible(False)
    ax.legend([wedges[0]], ["ਭਰਿਆ"] if prop else ["Filled"], loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8, prop=prop)
    ax.axis("equal")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf

def bar_graph(x, y, label, colour, ylim=None):
    g_df = pd.DataFrame({"Label": x, "Count": y})
    scale = alt.Scale(domain=[0, ylim]) if ylim else alt.Undefined
    chart = (alt.Chart(g_df).mark_bar().encode(
            x=alt.X("Label:N", title="", axis=alt.Axis(labelAngle=45)),
            y=alt.Y("Count:Q", title=label, scale=scale),
            color=alt.value(colour)).properties(height=400))
    return chart

def comparison_chart(first, latest, col1, col2, detected_lang, first_date, latest_date):
    first = first[[col1, col2]].dropna().set_index(col1)
    latest = latest[[col1, col2]].dropna().set_index(col1)
    line1, line2 = first[col2].rename(first_date), latest[col2].rename(latest_date)
    combined_df = pd.concat([line1,line2],axis = 1).reset_index()
    for col in [first_date, latest_date]:
        if combined_df[col].dtype == 'object':
            combined_df[col] = combined_df[col].str.rstrip('%').astype(float)
    col1 = combined_df.columns[0]
    combined_df = combined_df.rename(columns={col1: 'Category'})
    if detected_lang == 'pa': combined_df['Category'] = get_translation_batch(combined_df['Category'].astype(str).tolist())
    source = combined_df.melt(id_vars=['Category'], value_vars=[first_date, latest_date], var_name='Assessment_Date', value_name='Score')
    color_map = {first_date: '#94c0e1', latest_date: '#144481'}
    chart = alt.Chart(source).mark_bar().encode(
        x=alt.X('Category', axis=alt.Axis(labelAngle=90, title=get_translation(col1) if detected_lang == 'pa' else col1)),
        y=alt.Y('Score', scale=alt.Scale(domain=[0, 100]), title='ਸਕੋਰ (%)' if detected_lang == 'pa' else 'Score (%)'),
        color=alt.Color('Assessment_Date', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())), legend=None),
        xOffset='Assessment_Date',                        
        tooltip=['Category', 'Assessment_Date', 'Score']
    ).properties().interactive()
    return chart

def comparison_chart_pdf(first, latest, col1, col2, detected_lang, first_date, latest_date, font_path=None):
    first_clean = first[[col1, col2]].dropna().set_index(col1)
    latest_clean = latest[[col1, col2]].dropna().set_index(col1)
    line1 = first_clean[col2].rename(first_date)
    line2 = latest_clean[col2].rename(latest_date)
    combined_df = pd.concat([line1, line2], axis=1).reset_index()

    for col in [first_date, latest_date]:
        if combined_df[col].dtype == 'object': combined_df[col] = combined_df[col].str.rstrip('%').astype(float)

    col_name = combined_df.columns[0]
    categories = combined_df[col_name].astype(str).tolist()
    if detected_lang == 'pa': categories = get_translation_batch(categories)

    y1 = combined_df[first_date].fillna(0).tolist()
    y2 = combined_df[latest_date].fillna(0).tolist()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, y1, width, label=first_date, color='#94c0e1')
    ax.bar(x + width/2, y2, width, label=latest_date, color='#144481')

    prop = FontProperties(fname=font_path) if font_path else None
    ax.set_ylabel('ਸਕੋਰ (%)' if detected_lang == 'pa' else 'Score (%)', fontproperties=prop)
    title_str = get_translation(col1) if detected_lang == 'pa' else col1
    ax.set_title(title_str, fontproperties=prop, pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontproperties=prop)
    ax.legend(prop=prop, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 105)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def draw_paragraph_on_pdf(c, paragraph: str, y_position: float, page_size=A4, left: int = 50, right: int = 50, leading: int = 14, font_name: str = "Helvetica", font_size: int = 11):
    width, height = page_size
    styles = getSampleStyleSheet()
    style_custom = ParagraphStyle('CustomStyle', parent=styles['Normal'], fontName=font_name, fontSize=font_size, leading=leading)
    p = Paragraph(paragraph or "", style_custom)
    max_width = width - left - right
    w, h = p.wrap(max_width, y_position) 
    if y_position - h < 72:  
        c.showPage(); y_position = height - 50  
    p.drawOn(c, left, y_position - h)
    return y_position - h - 10  

# ==========================================
# 5. SCHOOL CORE RENDERERS & PDFS
# ==========================================

def render_domain_section(domain_no, domain_title, score_col, perc_col, remarks, report, data, detected_lang, font_name=None, bold_name=None, pdf_canvas=None, y_position=None, page_size=A4, st_container=None):
    cleaned_report = report[[domain_title, score_col, perc_col]].dropna()
    clean_report = cleaned_report.rename(columns={score_col: "Score", perc_col: "Score Percentage"})
    
    if st_container is not None:
        container = st_container.container() if hasattr(st_container, "container") else st_container
        with container:
            if not clean_report.empty:
                st.dataframe(clean_report.reset_index(drop=True))
                x = get_translation_batch(clean_report[domain_title].astype(str).tolist()) if detected_lang == 'pa' else clean_report[domain_title].astype(str)
                label = get_translation(score_col) if detected_lang == 'pa' else score_col             
                st.altair_chart(bar_graph(x, clean_report["Score"], label, colour="#4682B4", ylim=4), use_container_width=True)
            st.markdown("#### ਟਿੱਪਣੀਆਂ:" if detected_lang == 'pa' else "#### Remarks:")
            for remark_name, col in remarks:
                val = data[col].iloc[0] if col in data else ""
                if settings.ENABLE_REMARK_REPHRASE: val = llm.rephrase_remark(val)
                if detected_lang == 'pa':
                    val = get_translation(val)
                    remark_name = get_translation(remark_name)
                st.markdown(f"- **{remark_name}:** {val}")

    if pdf_canvas is not None and y_position is not None:
        c, width, height = pdf_canvas, page_size[0], page_size[1]
        c.setFont(bold_name if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED else "Helvetica-Bold", 14)
        c.drawString(50, y_position, f"ਡੋਮੇਨ {domain_no}: {get_translation(domain_title)}" if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED else f"Domain {domain_no}: {domain_title}")
        y_position -= 25
        c.setFont(font_name, 10)
        for _, row_report in cleaned_report.iterrows():
            c.drawString(50, y_position, f"{row_report[domain_title]} | {score_col}: {row_report[score_col]} | {perc_col}: {row_report[perc_col]}")
            y_position -= 15
            if y_position < 120: c.showPage(); y_position = height - 50
        y_position -= 10
        if not cleaned_report.empty:
            x = get_translation_batch(cleaned_report[domain_title].astype(str).tolist()) if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED else cleaned_report[domain_title].astype(str)
            label = get_translation(str(score_col)) if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED else str(score_col)
            chart_buf = bar_graph_pdf(x, cleaned_report[score_col], label, ylim=4, colour="#4682B4", font_path=settings.PUNJABI_REGULAR_FONT_PATH if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED else None)
            if y_position < 300: c.showPage(); y_position = height - 50
            c.drawImage(ImageReader(chart_buf), 50, y_position - 250, width=500, height=270)
            y_position -= 300
        
        return y_position

@st.cache_data(show_spinner="Generating Report PDF...")
def cached_generate_pdf(row, latest_date, school_info, overall_report, report, data, detected_lang, include_suggestions=False, suggestions_dict=None):
    return generate_pdf(row, latest_date, school_info, overall_report, report, data, detected_lang, include_suggestions, suggestions_dict)

def generate_pdf(row, latest_date, school_info, overall_report, report, data, detected_lang, include_suggestions=False, suggestions_dict=None):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4
    y_position = height - 50 
    
    regular_font = "Gurmukhi" if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED else "Helvetica"
    bold_font = "Gurmukhi-Bold" if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED else "Helvetica-Bold"
    
    c.setFont(bold_font, 16)
    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED: c.drawCentredString(width / 2, y_position, get_translation(str(row['School_Name']).strip().upper()))
    else: c.drawCentredString(width / 2, y_position, row['School_Name'].upper())
    y_position -= 30

    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
        c.setFont(bold_font, 16)
        c.drawString(50 , y_position, f"{get_translation('District')}: {get_translation(row['District'])} - {get_translation('Block')}: {get_translation(row['Block'])}")
        c.drawString(50, y_position - 20, f"{get_translation('UDISE Code')}: {row['UDISE_Code']}")
        c.drawString(50, y_position - 40, f"{get_translation('Assessment Date')}: {pd.to_datetime(latest_date).date()}")
    else:
        c.setFont(bold_font, 12)
        c.drawString(50, y_position, f"District: {row['District']} - Block: {row['Block']}")
        c.drawString(50, y_position - 20, f"UDISE Code: {row['UDISE_Code']}")
        c.drawString(50, y_position - 40, f"Assessment Date: {pd.to_datetime(latest_date).date()}")
    
    y_position -= 70
    c.line(50, y_position, width - 50, y_position)
    y_position -= 30 

    paragraph = get_translation(info_para(school_info)) if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED else info_para(school_info)
    y_position = draw_paragraph_on_pdf(c, paragraph, y_position, page_size=A4, left=50, right=50, font_name=regular_font)
    
    fp = settings.PUNJABI_REGULAR_FONT_PATH if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED else None
    g_buf = bar_graph_pdf(["ਕੁੜੀਆਂ", "ਮੁੰਡੇ"] if fp else ["Girls", "Boys"], [school_info['Total_girls'].iloc[0], school_info['Total_boys'].iloc[0]], "ਕੁੱਲ ਗਿਣਤੀ" if fp else "Count", "#4682B4", "ਲਿੰਗ ਅਨੁਪਾਤ ਬਾਰ" if fp else "Gender split bar", font_path=fp)
    t_buf = teacher_pie_buf(school_info['Teachers_Present'].iloc[0], school_info['Teacher_Positions_Sanctioned'].iloc[0], "ਮਨਜ਼ੂਰਸ਼ੁਦਾ ਬਨਾਮ ਭਰੇ ਹੋਏ ਅਧਿਆਪਕ ਅਹੁਦੇ" if fp else "Sanctioned vs. Filled Teachers Positions", font_path=fp)
    
    c.drawImage(ImageReader(g_buf), 50, y_position - 120, width=220, height=120, mask='auto')
    c.drawImage(ImageReader(t_buf), 290, y_position - 120, width=220, height=120, mask='auto')
    y_position -= 140

    for i, domain_title, score_col, perc_col, remarks in constants.domains:
        y_position = render_domain_section(i, domain_title, score_col, perc_col, remarks, report, data, detected_lang, font_name=regular_font, bold_name=bold_font, pdf_canvas=c, y_position=y_position, page_size=A4)

    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED: c.setFont(bold_font, 14); c.drawString(50, y_position, "ਕੁੱਲ ਸਕੋਰ:")
    else: c.setFont(bold_font, 14); c.drawString(50, y_position, "Overall Score:")
    y_position -= 25
    
    for i, row_overall in overall_report.iterrows():
        c.setFont("Helvetica", 10)
        c.drawString(50, y_position, f"{row_overall['Domain']} | Percentage: {row_overall['Score Percentage']}")
        y_position -= 15
        if y_position < 150: c.showPage(); y_position = height - 50

    y_position -= 25
    x_doms = get_translation_batch(overall_report['Domain'].tolist()) if fp else overall_report['Domain'].astype(str)
    chart_buf = bar_graph_pdf(x_doms, overall_report['Score Percentage'].str.rstrip('%').astype(float), "ਸਕੋਰ ਪ੍ਰਤੀਸ਼ਤ" if fp else "Score Percentage", "#3F51B5", ylim=100, font_path=fp)
    
    if y_position < 300: c.showPage(); y_position = height - 50
    c.drawImage(ImageReader(chart_buf), 50, y_position - 250, width=500, height=270)
    y_position -= 280
    
    if include_suggestions and suggestions_dict:
        c.showPage()
        y_position = height - 50
        c.setFont(bold_font, 16)
        title_text = "ਸੁਧਾਰ ਸੁਝਾਅ:" if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED else "Improvement Suggestions:"
        c.drawString(50, y_position, title_text)
        y_position -= 30
        
        for domain, items in suggestions_dict.items():
            if not items: continue
            c.setFont(bold_font, 12)
            d_text = f"{domain} / {get_translation(domain)}" if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED else domain
            c.drawString(50, y_position, d_text)
            y_position -= 20
            
            for remark_name, sugg in items:
                p_text = f"<b>{remark_name}:</b> {sugg}"
                y_position = draw_paragraph_on_pdf(c, p_text, y_position, page_size=A4, left=50, right=50, font_name=regular_font)
                y_position -= 10
                
    c.showPage()
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

@st.cache_data(show_spinner="Generating Comparison PDF...")
def cached_generate_school_comparison_pdf(row, earlier_date, latest_date, si_early, si_late, report_first, report_latest, Overall_first, Overall_latest, detected_lang):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4
    y_position = height - 50 
    
    regular_font = "Gurmukhi" if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED else "Helvetica"
    bold_font = "Gurmukhi-Bold" if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED else "Helvetica-Bold"
    
    first = str(pd.to_datetime(earlier_date).date())
    second = str(pd.to_datetime(latest_date).date())

    c.setFont(bold_font, 16)
    title = f"Comparison Report: {row['School_Name'].upper()}"
    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED: c.drawCentredString(width / 2, y_position, get_translation(title))
    else: c.drawCentredString(width / 2, y_position, title)
    y_position -= 30

    c.setFont(bold_font, 12)
    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
        c.drawString(50 , y_position, f"{get_translation('District')}: {get_translation(row['District'])} - {get_translation('Block')}: {get_translation(row['Block'])}")
        c.drawString(50, y_position - 20, f"{get_translation('UDISE Code')}: {row['UDISE_Code']}")
        c.drawString(50, y_position - 40, f"ਤੁਲਨਾ: {first} vs {second}")
    else:
        c.drawString(50, y_position, f"District: {row['District']} - Block: {row['Block']}")
        c.drawString(50, y_position - 20, f"UDISE Code: {row['UDISE_Code']}")
        c.drawString(50, y_position - 40, f"Comparison: {first} vs {second}")
    
    y_position -= 70
    c.line(50, y_position, width - 50, y_position)
    y_position -= 30 

    stu_prev = si_early.loc[0,"Total_student"] if not si_early.empty else 0
    tch_prev = si_early.loc[0,"Teachers_Present"] if not si_early.empty else 0
    stu_curr = si_late.loc[0,"Total_student"] if not si_late.empty else 0
    tch_curr = si_late.loc[0,"Teachers_Present"] if not si_late.empty else 0

    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    w = 0.35
    ax[0].bar([0], [stu_prev], w, label=first, color='#94c0e1')
    ax[0].bar([w], [stu_curr], w, label=second, color='#144481')
    ax[0].set_xticks([w/2])
    ax[0].set_xticklabels([get_translation("Students") if detected_lang=='pa' else "Students"], fontproperties=FontProperties(fname=settings.PUNJABI_REGULAR_FONT_PATH) if detected_lang=='pa' else None)
    
    ax[1].bar([0], [tch_prev], w, label=first, color='#94c0e1')
    ax[1].bar([w], [tch_curr], w, label=second, color='#144481')
    ax[1].set_xticks([w/2])
    ax[1].set_xticklabels([get_translation("Teachers") if detected_lang=='pa' else "Teachers"], fontproperties=FontProperties(fname=settings.PUNJABI_REGULAR_FONT_PATH) if detected_lang=='pa' else None)
    ax[1].legend(prop=FontProperties(fname=settings.PUNJABI_REGULAR_FONT_PATH) if detected_lang=='pa' else None)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    c.drawImage(ImageReader(buf), 50, y_position - 150, width=400, height=170, mask='auto')
    y_position -= 180

    for i, (d_name, d_col) in enumerate([("Safety and Hygiene", "Domain1 Score Percentage"), ("Stimulating School Environment", "Domain2 Score Percentage"), ("Physical Development Opportunities", "Domain3 Score Percentage"), ("Smart School Facilities", "Domain4 Score Percentage")]):
        chart_buf = comparison_chart_pdf(report_first, report_latest, d_name, d_col, detected_lang, first, second, font_path=settings.PUNJABI_REGULAR_FONT_PATH if detected_lang == 'pa' else None)
        if y_position - 180 < 50: c.showPage(); y_position = height - 50
        c.drawImage(ImageReader(chart_buf), 50, y_position - 180, width=450, height=180, mask='auto')
        y_position -= 200

    if y_position - 200 < 50: c.showPage(); y_position = height - 50
    overall_buf = comparison_chart_pdf(Overall_first, Overall_latest, "Domain", "Score Percentage", detected_lang, first, second, font_path=settings.PUNJABI_REGULAR_FONT_PATH if detected_lang == 'pa' else None)
    c.drawImage(ImageReader(overall_buf), 50, y_position - 200, width=450, height=190, mask='auto')

    c.showPage()
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer


@st.cache_data(show_spinner="Generating Report PDF...")

# ==========================================
# 6. SCHOOL UI VIEWS
# ==========================================

@st.fragment
def render_latest_view(data, row, code, long_report, school_info, detected_lang):
    latest_date = data["Assessment_Date"].max()
    report = pivot_table(code, long_report, latest_date)
    Overall = overall_score(report)
    sf = school_info[(school_info['UDISE_Code'] == code) & (school_info['Assessment_Date'] == latest_date)]

    st.markdown("---")
    st.header(get_translation(row['School_Name'].upper()) if detected_lang == 'pa' else row['School_Name'].upper())

    c1, c2 = st.columns(2)      
    if detected_lang == 'pa':
        c1.subheader(f"{get_translation('District')}: {get_translation(row['District'])} ")
        c1.write(f"{get_translation('Assessment Date')}: {pd.to_datetime(latest_date).date()}")
        c2.subheader(f"{get_translation('Block')}: {get_translation(row['Block'])}")
        c2.write(f"{get_translation('UDISE Code')}: {row['UDISE_Code']}")
    else:
        c1.subheader(f"District: {row['District']} ")
        c1.write(f"Assessment Date: {pd.to_datetime(latest_date).date()}")
        c2.subheader(f"Block: {row['Block']}")
        c2.write(f"UDISE Code: {row['UDISE_Code']}")

    st.markdown("---")
    st.markdown(get_translation(info_para(sf)) if detected_lang == 'pa' else info_para(sf))
    
    gc1, gc2 = st.columns([1,1])
    with gc1:
        g_df = pd.DataFrame({"ਲਿੰਗ" if detected_lang == 'pa' else "Gender":["ਕੁੜੀਆਂ","ਲੜਕੇ"] if detected_lang == 'pa' else ["Girls","Boys"], "Count":[sf['Total_girls'].iloc[0], sf['Total_boys'].iloc[0]]})
        col_name = "ਲਿੰਗ" if detected_lang == 'pa' else "Gender"
        st.altair_chart(alt.Chart(g_df).mark_bar().encode(x=f"{col_name}:N", y="Count:Q", color=alt.Color(f"{col_name}:N", legend=None)).properties(height=200), use_container_width=True)
    with gc2:
        tp, sanc = int(sf['Teachers_Present'].iloc[0] or 0), int(sf['Teacher_Positions_Sanctioned'].iloc[0] or 0)
        t_df = pd.DataFrame({"Status": ["ਭਰਿਆ", "ਬਾਕੀ"] if detected_lang == 'pa' else ["Filled", "Remainder"], "Count": [tp, max(sanc - tp, 0)]})
        st.altair_chart(alt.Chart(t_df).mark_arc(innerRadius=40).encode(theta="Count:Q", color=alt.Color("Status:N", scale=alt.Scale(domain=["ਭਰਿਆ"] if detected_lang=='pa' else ["Filled"], range=["#2ca02c"])), order=alt.Order("Status", sort="descending")).properties(height=200), use_container_width=True)

    st.markdown("---")
    st.write("ਵਿਸਤ੍ਰਿਤ ਡੋਮੇਨ ਰਿਪੋਰਟ:" if detected_lang == 'pa' else "Detailed Domain Report:")

    for i, d_title, s_col, p_col, rem in constants.domains:
        disp = f"Domain {i}: {d_title} / {get_translation(d_title)}" if detected_lang == 'pa' else f"Domain {i}: {d_title}"
        with st.expander(disp, expanded=True):
            render_domain_section(i, d_title, s_col, p_col, rem, report, data, detected_lang, st_container=st)

    st.subheader("ਕੁੱਲ ਸਕੋਰ:" if detected_lang == 'pa' else "Overall Score:")
    st.dataframe(Overall.reset_index(drop=True))
    if detected_lang == 'pa':
        st.altair_chart(bar_graph(get_translation_batch(Overall['Domain'].astype(str).tolist()), Overall['Score Percentage'].str.rstrip('%').astype(float), "ਸਕੋਰ ਪ੍ਰਤੀਸ਼ਤ", ylim=100, colour="#3F51B5"), use_container_width=True)
    else:
        st.altair_chart(bar_graph(Overall['Domain'].astype(str), Overall['Score Percentage'].str.rstrip('%').astype(float), "Score Percentage", ylim=100, colour="#3F51B5"), use_container_width=True)

@st.fragment
def improvement_interaction(data, row, code, long_report, school_info, detected_lang):
    latest_date = data["Assessment_Date"].max()
    report = pivot_table(code, long_report, latest_date)
    Overall = overall_score(report)
    sf = school_info[(school_info['UDISE_Code'] == code) & (school_info['Assessment_Date'] == latest_date)]

    vk = f"view_improve_{code}"
    nk = f"msg_no_{code}"
    
    if vk not in st.session_state: st.session_state[vk] = False
    if nk not in st.session_state: st.session_state[nk] = False
    
    st.markdown("---")
    with st.chat_message("assistant"):
        st.markdown("ਕੀ ਤੁਸੀਂ ਸੁਝਾਏ ਗਏ ਸੁਧਾਰ ਦੇਖਣਾ ਚਾਹੋਗੇ?" if detected_lang == 'pa' else "Would you like to see suggested improvements?")

    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        if st.button("ਹਾਂ" if detected_lang == 'pa' else "Yes", key=f"y_{code}"): st.session_state[vk] = True; st.session_state[nk] = False; st.rerun()
    with col2:
        if st.button("ਨਹੀਂ" if detected_lang == 'pa' else "No", key=f"n_{code}"): st.session_state[vk] = False; st.session_state[nk] = True; st.rerun()
    
    suggestions_dict = {}

    if st.session_state[vk]:
        options = [d[1] for d in constants.domains]
        if detected_lang == 'pa': st.write("ਆਪਣਾ ਡੋਮੇਨ ਚੁਣੋ:")
        else: st.write("Select your Domain(s):")

        selected = [opt for opt in options if st.checkbox(f"{opt} / {get_translation(opt)}" if detected_lang == 'pa' else opt, key=f"imp_{code}_{opt}")]
        
        st.markdown("#### ਸੁਧਾਰ ਸੁਝਾਅ:" if detected_lang == 'pa' else "#### Improvement Suggestions:")
        for opt in selected:
            d_data = next((d for d in constants.domains if d[1] == opt), None)
            if d_data:
                dom_id, domain_title, score_col, perc_col, remarks = d_data
                st.subheader(f"{domain_title} / {get_translation(domain_title)}" if detected_lang == 'pa' else domain_title)
                
                domain_suggestions = []
                for remark_name, col in remarks:
                    val = str(data[col].iloc[0]) if col in data else ""
                    s_ser = report.loc[report[domain_title] == remark_name, score_col]
                    if not s_ser.empty and s_ser.values[0] != 4:
                        sugg = cached_suggestion(val)
                        disp_rn = get_translation(remark_name) if detected_lang == 'pa' else remark_name
                        disp_sg = get_translation(sugg) if detected_lang == 'pa' else sugg
                        st.markdown(f"- #### **{disp_rn}:**\n<div style='padding-left: 20px; border-left: 3px solid #f0f2f6;'>{disp_sg.replace(chr(10), '<br>')}</div><br>", unsafe_allow_html=True)
                        domain_suggestions.append((disp_rn, disp_sg))
                
                if domain_suggestions:
                    suggestions_dict[domain_title] = domain_suggestions
                st.markdown("---")
                
    elif st.session_state[nk]:
         if detected_lang == 'pa': st.markdown("ਠੀਕ ਹੈ, ਕੋਈ ਗੱਲ ਨਹੀਂ।")
         else: st.markdown("Okay, no problem.")

    # Generate PDF ONLY after the user makes a choice
    if st.session_state[vk] or st.session_state[nk]:
        st.markdown("---")
        pdf_data = cached_generate_pdf(row, latest_date, sf, Overall, report, data, detected_lang, include_suggestions=st.session_state[vk], suggestions_dict=suggestions_dict)
        
        col_spacer1, col_btn, col_spacer2 = st.columns([1, 2, 1])
        with col_btn:
            btn_label = "📥 ਰਿਪੋਰਟ PDF ਡਾਊਨਲੋਡ ਕਰੋ" if detected_lang == 'pa' else "📥 Download Report PDF"
            st.download_button(btn_label, pdf_data, f"{row['School_Name'].upper()}_report.pdf", "application/pdf", use_container_width=True, type="primary", key=f"dl_pdf_fin_{code}")     
# ==========================================
# 7. VILLAGE RENDERERS
# ==========================================

def get_village_metric(data: dict, path: str):
    if path in data: return data[path]
    keys = path.split('.')
    val = data
    for k in keys:
        if isinstance(val, dict): val = val.get(k, 0)
        else: return 0
    return val

def render_village_view(villages_data: list, detected_lang: str):
    is_comp = len(villages_data) == 2
    st.markdown("---")
    
    # 1. Header Rendering
    if not is_comp:
        v = villages_data[0]
        year = v.get('assessment_year', 'N/A')
        if detected_lang == 'pa':
            st.markdown(f"### 🏡 ਪਿੰਡ: {get_translation(v.get('village_name', ''))}")
            st.markdown(f"**ਗ੍ਰਾਮ ਪੰਚਾਇਤ:** {get_translation(v.get('gp_name', ''))} &nbsp;|&nbsp; **ਬਲਾਕ:** {get_translation(v.get('block_name', ''))} &nbsp;|&nbsp; **ਸਾਲ:** {year}")
        else:
            st.markdown(f"### 🏡 Village: {v.get('village_name', '')}")
            st.markdown(f"**Gram Panchayat:** {v.get('gp_name', '')} &nbsp;|&nbsp; **Block:** {v.get('block_name', '')} &nbsp;|&nbsp; **Year:** {year}")
    else:
        v1, v2 = villages_data
        if detected_lang == 'pa':
            st.markdown(f"### 📊 ਪਿੰਡ ਦੀ ਤੁਲਨਾ: {get_translation(v1.get('village_name', ''))} vs {get_translation(v2.get('village_name', ''))}")
        else:
            st.markdown(f"### 📊 Village Comparison: {v1.get('village_name', '')} vs {v2.get('village_name', '')}")
        
    st.markdown("---")

    # 2. Domain Expander Loop
    for d_idx, d_name, _, metrics in constants.VILLAGE_DOMAINS:
        disp_title = f"Domain {d_idx}: {d_name} / {get_translation(d_name)}" if detected_lang == 'pa' else f"Domain {d_idx}: {d_name}"
        with st.expander(disp_title, expanded=True):
            
            if not is_comp:
                v = villages_data[0]
                c1, c2 = st.columns([1, 1.5])
                valid_metrics = []
                
                with c1:
                    for lbl, path in metrics:
                        val = get_nested(v, path)
                        dlbl = get_translation(lbl) if detected_lang == 'pa' else lbl
                        st.markdown(f"**{dlbl}:** {val}")
                        
                        if val is not None and str(val).lower() not in ['nan', 'none', 'n/a', '']:
                            try:
                                clean_float = float(val)
                                valid_metrics.append({"label": dlbl, "value": clean_float})
                            except (ValueError, TypeError): 
                                continue
                
                with c2:
                    if valid_metrics:
                        df = pd.DataFrame(valid_metrics).sort_values(by="value")
                        fig = px.bar(df, x="value", y="label", orientation='h', text="value", color="value", color_continuous_scale="Blues")
                        
                        fig.update_traces(textposition='outside', marker_line_color='black', marker_line_width=1)
                        fig.update_layout(
                            xaxis_title="Score" if detected_lang != 'pa' else "ਸਕੋਰ",
                            yaxis_title="Metric" if detected_lang != 'pa' else "ਮੈਟ੍ਰਿਕ",
                            coloraxis_showscale=False,
                            plot_bgcolor='white',
                            xaxis=dict(showgrid=True, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black'),
                            yaxis=dict(showgrid=False, showline=True, linewidth=1, linecolor='black'),
                            margin=dict(l=0, r=30, t=10, b=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                cols = st.columns(2)
                for idx, (col, v) in enumerate(zip(cols, villages_data)):
                    with col:
                        v_name = get_translation(v.get('village_name')) if detected_lang == 'pa' else v.get('village_name')
                        year = v.get('assessment_year', 'N/A')
                        st.markdown(f"#### 🔹 {v_name} ({year})")
                        
                        valid_metrics = []
                        for lbl, path in metrics:
                            val = get_nested(v, path)
                            dlbl = get_translation(lbl) if detected_lang == 'pa' else lbl
                            st.markdown(f"**{dlbl}:** {val}")
                            
                            if val is not None and str(val).lower() not in ['nan', 'none', 'n/a', '']:
                                try:
                                    clean_float = float(val)
                                    valid_metrics.append({"label": dlbl, "value": clean_float})
                                except (ValueError, TypeError): 
                                    continue
                                
                        if valid_metrics:
                            df = pd.DataFrame(valid_metrics).sort_values(by="value")
                            fig = px.bar(df, x="value", y="label", orientation='h', text="value", color="value", color_continuous_scale="Blues" if idx == 0 else "Teal")
                            
                            fig.update_traces(textposition='outside', marker_line_color='black', marker_line_width=1)
                            fig.update_layout(
                                xaxis_title="Score" if detected_lang != 'pa' else "ਸਕੋਰ",
                                yaxis_title="Metric" if detected_lang != 'pa' else "ਮੈਟ੍ਰਿਕ",
                                coloraxis_showscale=False,
                                plot_bgcolor='white',
                                xaxis=dict(showgrid=True, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black'),
                                yaxis=dict(showgrid=False, showline=True, linewidth=1, linecolor='black'),
                                margin=dict(l=0, r=30, t=10, b=0)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
# ==========================================
# 8. VILLAGE PDF GENERATOR
# ==========================================

def generate_village_pdf(villages_data: list, detected_lang: str, insights: str) -> bytes:
    # --- FIX 1: GUARANTEE FONT REGISTRATION AT PDF CREATION TIME ---
    if detected_lang == 'pa' and not settings.PUNJABI_FONT_LOADED:
        register_fonts()
        
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    
    fp = settings.PUNJABI_REGULAR_FONT_PATH if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED else None
    f_reg = 'Gurmukhi' if fp else 'Helvetica'
    f_bold = 'Gurmukhi-Bold' if fp else 'Helvetica-Bold'
    
    # --- Translates incoming text and reshapes Gurmukhi vowels ---
    def safe_punjabi_pdf_text(text: str) -> str:
        if not text: return ""
        res = get_translation(text) if fp else str(text)
        if fp:
            import re
            res = re.sub(r'([\u0A15-\u0A39\u0A59-\u0A5E](?:\u0A4D[\u0A15-\u0A39\u0A59-\u0A5E])?)([\u0A3F])', r'\2\1', res)
        return res

    # --- Reshaper specifically for hardcoded static Punjabi labels ---
    def reshape_only(text: str) -> str:
        if not fp: return text
        import re
        return re.sub(r'([\u0A15-\u0A39\u0A59-\u0A5E](?:\u0A4D[\u0A15-\u0A39\u0A59-\u0A5E])?)([\u0A3F])', r'\2\1', text)

    s = getSampleStyleSheet()
    t_style = ParagraphStyle('T', parent=s['Heading1'], fontName=f_bold, fontSize=16, spaceAfter=12)
    h2_style = ParagraphStyle('H2', parent=s['Heading2'], fontName=f_bold, fontSize=14, spaceAfter=8, spaceBefore=12)
    b_style = ParagraphStyle('B', parent=s['Normal'], fontName=f_reg, fontSize=10, leading=14)
    
    is_comp = len(villages_data) == 2
    
    if not is_comp:
        v = villages_data[0]
        v_name = safe_punjabi_pdf_text(str(v.get('village_name', 'Unknown')))
        gp = safe_punjabi_pdf_text(str(v.get('gp_name', '')))
        block = safe_punjabi_pdf_text(str(v.get('block_name', '')))
        
        # FIX 2: Prevent massive concatenated strings from crashing the translation API
        title = f"{reshape_only('ਪਿੰਡ ਦੀ ਰਿਪੋਰਟ')}: {v_name}" if fp else f"Village Report: {v_name}"
        elements.append(Paragraph(title, t_style))
        
        info = f"{reshape_only('ਗ੍ਰਾਮ ਪੰਚਾਇਤ')}: {gp} | {reshape_only('ਬਲਾਕ')}: {block}" if fp else f"Gram Panchayat: {gp} | Block: {block}"
        elements.append(Paragraph(info, b_style))
        
        for d_idx, d_name, _, metrics in constants.VILLAGE_DOMAINS:
            d_name_trans = safe_punjabi_pdf_text(d_name)
            dom_title = f"{reshape_only('ਡੋਮੇਨ')} {d_idx}: {d_name_trans}" if fp else f"Domain {d_idx}: {d_name_trans}"
            elements.append(Paragraph(dom_title, h2_style))
            
            m_names, v1_vals, v2_vals = [], [], []
            for lbl, path in metrics:
                val = get_nested(v, path)
                dlbl = safe_punjabi_pdf_text(lbl)
                
                disp_val = safe_punjabi_pdf_text(str(val)) if fp and isinstance(val, str) else str(val)
                elements.append(Paragraph(f"<b>{dlbl}:</b> {disp_val}", b_style))
                
                try:
                    clean_val = float(val)
                    m_names.append(dlbl)
                    m_vals.append(clean_val)
                except (ValueError, TypeError):
                    pass
                    
            if m_vals:
                fig, ax = plt.subplots(figsize=(6, max(2.5, len(m_names) * 0.5)))
                ax.barh(m_names, m_vals, color='#4682B4')
                
                prop = fm.FontProperties(fname=fp) if fp else None
                for i, val_plot in enumerate(m_vals): 
                    ax.text(val_plot, i, f" {val_plot}", va='center', fontproperties=prop)
                
                if prop:
                    ax.set_yticks(range(len(m_names)))
                    ax.set_yticklabels(m_names, fontproperties=prop)
                    
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                elements.append(Spacer(1, 10))
                elements.append(RLImage(buf, width=6*inch, height=max(2.5, len(m_names)*0.5)*inch))
                
    else:
        v1, v2 = villages_data
        n1 = safe_punjabi_pdf_text(str(v1.get('village_name', 'V1')))
        n2 = safe_punjabi_pdf_text(str(v2.get('village_name', 'V2')))
        
        col1_name = f"{n1} (A)" if n1 == n2 else n1
        col2_name = f"{n2} (B)" if n1 == n2 else n2
        
        title = f"{reshape_only('ਤੁਲਨਾ')}: {n1} vs {n2}" if fp else f"Comparison: {n1} vs {n2}"
        elements.append(Paragraph(title, t_style))
        
        for d_idx, d_name, _, metrics in constants.VILLAGE_DOMAINS:
            d_name_trans = safe_punjabi_pdf_text(d_name)
            dom_title = f"{reshape_only('ਡੋਮੇਨ')} {d_idx}: {d_name_trans}" if fp else f"Domain {d_idx}: {d_name_trans}"
            elements.append(Paragraph(dom_title, h2_style))
            
            m_names, v1_vals, v2_vals = [], [], []
            for lbl, path in metrics:
                val1, val2 = get_nested(v1, path), get_nested(v2, path)
                dlbl = safe_punjabi_pdf_text(lbl)
                
                d_val1 = safe_punjabi_pdf_text(str(val1)) if fp and isinstance(val1, str) else str(val1)
                d_val2 = safe_punjabi_pdf_text(str(val2)) if fp and isinstance(val2, str) else str(val2)
                
                elements.append(Paragraph(f"<b>{dlbl}:</b> {n1} ({d_val1}) | {n2} ({d_val2})", b_style))
                
                try:
                    clean_v1 = float(val1)
                    clean_v2 = float(val2)
                    m_names.append(dlbl)
                    v1_vals.append(clean_v1)
                    v2_vals.append(clean_v2)
                except (ValueError, TypeError):
                    pass
                    
            if m_names:
                df = pd.DataFrame({"M": m_names, col1_name: v1_vals, col2_name: v2_vals})
                fig, ax = plt.subplots(figsize=(6.5, max(2.5, len(m_names) * 0.6)))
                
                df.plot(x="M", y=[col1_name, col2_name], kind="barh", ax=ax, color=['#4682B4', '#20B2AA'])
                
                prop = fm.FontProperties(fname=fp) if fp else None
                if prop: 
                    ax.set_yticks(range(len(m_names)))
                    ax.set_yticklabels(m_names, fontproperties=prop)

                    legend = ax.get_legend()
                    if legend:
                        for text in legend.get_texts():
                            text.set_fontproperties(prop)
                    
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                elements.append(Spacer(1, 10))
                elements.append(RLImage(buf, width=6.5*inch, height=max(2.5, len(m_names)*0.6)*inch))
                
    elements.append(Spacer(1, 20))
    insight_title = reshape_only("AI ਸੂਝ") if fp else "AI Insights"
    elements.append(Paragraph(insight_title, h2_style))
    
    for line in (insights or "").split('\n'):
        if line.strip(): 
            clean_line = line.replace('**', '').replace('#', '').strip()
            if fp: 
                clean_line = reshape_only(clean_line)
            elements.append(Paragraph(clean_line, b_style))
            
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()