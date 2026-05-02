import re
import io
from typing import Optional
import time
from pathlib import Path

import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import altair as alt
import streamlit as st
from deep_translator import GoogleTranslator

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph

from src import llm, constants
from src.config import settings

# ==========================================
# 1. DATA LOADING & NORMALIZATION
# ==========================================

def load_school_list(path: str) -> pd.DataFrame:
    """Load and normalise school list. Returns DataFrame with normalized columns."""
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

def register_fonts():
    """
    Registers the Punjabi fonts for ReportLab.
    """
    try:
        pdfmetrics.registerFont(TTFont('Gurmukhi', Path(settings.PUNJABI_REGULAR_FONT_PATH)))
        pdfmetrics.registerFont(TTFont('Gurmukhi-Bold', Path(settings.PUNJABI_BOLD_FONT_PATH)))
        
        pdfmetrics.registerFontFamily(
            'Gurmukhi', 
            normal='Gurmukhi', 
            bold='Gurmukhi-Bold', 
            italic='Gurmukhi',     
            boldItalic='Gurmukhi-Bold'       
        )
        
        settings.PUNJABI_FONT_LOADED = True

    except Exception as e:
        print(f"Font Registration Failed: {e}")
        settings.PUNJABI_FONT_LOADED = False

# =========================
# 3. MATCHING LOGIC
# =========================

def generic_fuzzy_match(text: str, df: pd.DataFrame, col_name: str, top_n: int = 20) -> pd.DataFrame:
    """matching School Name or YL Name."""
    query = norm_text(text)
    
    # 1. Exact Substring Match
    clean_query = re.sub(r'(?<=\w)\.(?=\w)', '', query) if col_name == 'School_Name' else re.split(r'[.@_]', query, 1)[0]
    
    candidates = df[df[col_name].str.contains(clean_query, na=False)]
    if not candidates.empty:
        return candidates[['District','Block','School_Name','UDISE_Code', 'YL_Name']]

    # 2. Fuzzy Matching
    choices = df[col_name].tolist()
    results = process.extract(text, choices, scorer=fuzz.WRatio, limit=top_n)

    # Filter by score > threshold
    candidates_idx = [idx for _, score, idx in results if score > settings.FUZZY_SCORE_THRESHOLD]

    if candidates_idx:
        return df.iloc[candidates_idx][['District','Block','School_Name','UDISE_Code', 'YL_Name']]
    
    return pd.DataFrame(columns=['District','Block','School_Name','UDISE_Code','YL_Name'])

def extract_school_from_message(report_json, df: pd.DataFrame):
    """Router to find schools based on extracted entities."""
    # 1. UDISE Match 
    if report_json.get("udisecode"):
        matched = df[df['UDISE_Code'] == report_json["udisecode"]]
        if not matched.empty:
            return matched[['District','Block','School_Name','UDISE_Code']]

    # 2. School Name Match
    if report_json.get("school_name"):
        res = generic_fuzzy_match(report_json["school_name"], df, 'School_Name')
        if not res.empty: return res

    # 3. YL Name Match
    if report_json.get("username"):
        res = generic_fuzzy_match(report_json["username"], df, 'YL_Name')
        if not res.empty: return res
    
    return pd.DataFrame(columns=['District','Block','School_Name','UDISE_Code'])

# ================================
# 4. REPORT DATA PROCESSING
# ================================

@st.cache_data
def report_card(school_data: pd.DataFrame) -> pd.DataFrame:
    """Pre-processes the dataframe to melt it into long format."""
    report_card_df = pd.DataFrame(school_data[["School_Name", "YL_Name", "UDISE_Code", "Assessment_Date"]])
    
    # Define mapping config
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

    # Map columns to Domains
    col2domain = {}
    domain_map = {
        "Safety and Hygiene": constants.safety_cols,
        "Stimulating School Environment": constants.stim_cols,
        "Physical Development Opportunities": constants.physical_cols,
        "Smart School Facilities": constants.smart_cols
    }

    for d_name, cols in domain_map.items():
        for c in cols:
            if c in report_card_df.columns:
                col2domain[c] = (d_name, c.split("/",1)[-1].strip())

    # Melt
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
        
        # Stack scores and percentages
        score = d_data.pivot(index='Feature', columns=['Domain'], values=['Domain Score'])
        perc = d_data.pivot(index='Feature', columns=['Domain'], values=['Score Percentage'])
        
        combined = pd.concat([score, perc], axis=1)
        stacked = combined.stack(level='Domain', future_stack=True).reset_index().drop('Domain', axis=1)
        
        stacked.rename(columns={
            "Feature": d_name,
            'Domain Score': f'Domain{idx} Score',
            'Score Percentage': f'Domain{idx} Score Percentage'
        }, inplace=True)
        dfs.append(stacked)

    return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

def overall_score(report_df: pd.DataFrame) -> pd.DataFrame:
    """Return the overall domain level percentages.
    """
    domains = [
        'Domain1 Score Percentage', 'Domain2 Score Percentage',
        'Domain3 Score Percentage', 'Domain4 Score Percentage'
        ]
    domain_names = [
        'Safety and Hygiene', 'Stimulating School Environment',
        'Physical Development Opportunities', 'Smart School Facilities'
    ]
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

    df[constants.object_to_int_col] = (
        df[constants.object_to_int_col]
        .apply(lambda x: pd.to_numeric(x, errors="coerce"))
        .fillna(0)
        .astype(int)
    )

    school_info = df.reindex(columns=constants.School_info_cols).copy()

    school_info["Total_girls"] =(
        df["Students_PrePrimary_Girls"]+ df["Students_Grade1_Girls"]+ df["Students_Grade2_Girls"] + 
        df["Students_Grade3_Girls"] + df["Students_Grade4_Girls"] + df["Students_Grade5_Girls"]
    )

    school_info["Total_boys"] = (
        df["Students_PrePrimary_Boys"] + df["Students_Grade1_Boys"]+ df["Students_Grade2_Boys"] +
        df["Students_Grade3_Boys"]+ df["Students_Grade4_Boys"]+ df["Students_Grade5_Boys"]
    )

    school_info["Total_student"] = np.where(
        (school_info["Total_girls"] == 0) & (school_info["Total_boys"] == 0),
        school_info["Students_Total_GPS"],
        school_info["Total_girls"] + school_info["Total_boys"]
    ).astype(int)

    return school_info

def info_para(school_info: pd.DataFrame) -> str:

    # Derived metrics (safe division)
    girls_pct = round((school_info['Total_girls'].iloc[0] / school_info['Total_student'].iloc[0] * 100), 1) if school_info['Total_girls'].iloc[0] else 0
    teacher_fill_pct = round((school_info['Teachers_Present'].iloc[0] / school_info['Teacher_Positions_Sanctioned'].iloc[0] * 100), 1) if school_info['Teacher_Positions_Sanctioned'].iloc[0] else 0

    parts = []

    # Students sentence
    if school_info["Total_student"].iloc[0]:
        stud_part = f"The school has total of {school_info['Total_student'].iloc[0]:,} students"
        extras = []
        if school_info['Total_girls'].iloc[0]:
            extras.append(f"including {school_info['Total_girls'].iloc[0]:,} girls ({girls_pct}% )")
        if school_info['Total_boys'].iloc[0]:
            extras.append(f"and {school_info['Total_boys'].iloc[0]:,} boys")
        if extras:
            stud_part += " — " + ", ".join(extras)
        stud_part += "."
        parts.append(stud_part)
    else:
        # still show girls/boys if total missing
        if school_info['Total_girls'].iloc[0] or school_info['Total_boys'].iloc[0]:
            gb = []
            if school_info['Total_girls'].iloc[0]:
                gb.append(f"{school_info['Total_girls'].iloc[0]:,} girls")
            if school_info['Total_boys'].iloc[0]:
                gb.append(f"{school_info['Total_boys'].iloc[0]:,} boys")
            parts.append("Student composition: " + ", ".join(gb) + ".")

    # Teachers sentence
    if school_info['Teacher_Positions_Sanctioned'].iloc[0]:
        teacher_part = f"There are {school_info['Teacher_Positions_Sanctioned'].iloc[0]:,} sanctioned teacher positions"
        subparts = []
        if school_info['Teachers_Present'].iloc[0]:
            if school_info['Teachers_Present'].iloc[0]>1:
                subparts.append(f"out of which {school_info['Teachers_Present'].iloc[0]:,} are currently filled ({teacher_fill_pct}% filled)")
            else: 
                subparts.append(f"out of which {school_info['Teachers_Present'].iloc[0]:,} is currently filled ({teacher_fill_pct}% filled)")
        if school_info['Teachers_Deputation'].iloc[0]:
            if school_info['Teachers_Deputation'].iloc[0]>1:
                subparts.append(f"{school_info['Teachers_Deputation'].iloc[0]:,} are on deputation")
            else:
                subparts.append(f"{school_info['Teachers_Deputation'].iloc[0]:,} is on deputation")
        if school_info['Teachers_New_Recruits'].iloc[0]:
            if school_info['Teachers_New_Recruits'].iloc[0]>1:
                subparts.append(f"{school_info['Teachers_New_Recruits'].iloc[0]:,} new teachers are recruited")
            else:
                subparts.append(f"{school_info['Teachers_New_Recruits'].iloc[0]:,} new teacher is recruited")
        if subparts:
            teacher_part += " , " + "; ".join(subparts)
        teacher_part += "."
        parts.append(teacher_part)
    else:
        # fallback: show any teacher counts available
        tpieces = []
        if school_info['Teachers_Present'].iloc[0]:
            tpieces.append(f"There are currently {school_info['Teachers_Present'].iloc[0]:,} teachers present")
        if school_info['Teachers_Deputation'].iloc[0]:
            if school_info['Teachers_Deputation'].iloc[0]>1:
                tpieces.append(f"{school_info['Teachers_Deputation'].iloc[0]:,} teachers are on deputation")
            else:
                tpieces.append(f"{school_info['Teachers_Deputation'].iloc[0]:,} teacher is on deputation")
        if school_info['Teachers_New_Recruits'].iloc[0]:
            if school_info['Teachers_New_Recruits'].iloc[0]>1:
                tpieces.append(f"{school_info['Teachers_New_Recruits'].iloc[0]:,} new teachers are recruited")
            else:
                tpieces.append(f"{school_info['Teachers_New_Recruits'].iloc[0]:,} new teacher is recruited")
        if tpieces:
            parts.append("Teachers: " + "; ".join(tpieces) + ".")

    # Children & anganwadi & Disabilities
    child_parts = []
    if school_info['Students_Disability_Count'].iloc[0]:
        if school_info['Students_Disability_Count'].iloc[0]>1:
            child_parts.append(f"There are {school_info['Students_Disability_Count'].iloc[0]:,} students with disabilities")
        else:
            child_parts.append(f"There is {school_info['Students_Disability_Count'].iloc[0]:,} student with disabilities")
    if school_info['Children_PrivateSchool'].iloc[0]:
        child_parts.append(f"{school_info['Children_PrivateSchool'].iloc[0]:,} children attend private schools")
    if school_info['Children_Anganwadi_0_3'].iloc[0]:
        if school_info['Children_Anganwadi_0_3'].iloc[0]>1:
            child_parts.append(f"{school_info['Children_Anganwadi_0_3'].iloc[0]:,} children aged 0-3 are enrolled in Anganwadi centers")
        else:
            child_parts.append(f"{school_info['Children_Anganwadi_0_3'].iloc[0]:,} children aged 0-3 is enrolled in Anganwadi centers")
    if child_parts:
        parts.append(", and ".join(child_parts) + ".")

    # Join into a single paragraph
    paragraph = " ".join(parts).strip()
    if not paragraph:
        paragraph = "No student/teacher details available."
    
    return paragraph

# ===================================
# 5. WRAPPER FUNCTIONS FOR CACHING
# =================================== 

# wrapper for translation to cache results
@st.cache_data(show_spinner=False)
def get_translation(text, dest='pa'):
    if not text or not isinstance(text, str):
        return text
    try:
        return GoogleTranslator(source='auto', target=dest).translate(text)
    except Exception:
        return text

@st.cache_data(show_spinner=False)
def get_translation_batch(text_list, dest='pa'):
    if not text_list:
        return []
    try:
        # Ensure list contains only strings
        clean_list = [str(t) for t in text_list]
        return GoogleTranslator(source='auto', target=dest).translate_batch(clean_list)
    except Exception:
        return text_list

# Cache the PDF generation
@st.cache_data(show_spinner="Generating Report PDF...")
def cached_generate_pdf(row, latest_date, school_info, overall_report, report, data, detected_lang):
    # This calls your original heavy function
    return generate_pdf(row, latest_date, school_info, overall_report, report, data, detected_lang)

# Cache LLM calls
@st.cache_data(show_spinner=False)
def cached_rephrase(val):
    return llm.rephrase_remark(val)

@st.cache_data(show_spinner=False)
def cached_suggestion(val):
    return llm.improvment_suggestion(val)

# ==========================================
# 6. CHARTS & PDF GENERATION
# ==========================================  

def bar_graph_pdf(x, y, ylabel, colour, title: str = None, ylim: int=None, font_path=None):
    """Return a PNG bytes buffer of a simple bar chart (matplotlib)."""

    custom_font = None
    if font_path:
        custom_font = FontProperties(fname=font_path)
        
    fig, ax = plt.subplots()
    ax.bar(x, y, color=colour)

    if custom_font:
        ax.set_ylabel(ylabel, fontproperties=custom_font, fontsize=10)
    else:
        ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(0, ylim)
    
    if title:
        if custom_font:
            ax.set_title(title, fontproperties=custom_font, fontsize=14)
        else:
            ax.set_title(title, fontsize=17, fontweight='normal')
    
    plt.xticks(rotation=45, ha='right')
    if custom_font:
        for label in ax.get_xticklabels():
            label.set_fontproperties(custom_font)

    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def teacher_pie_buf(present, sanctioned, title, font_path=None):

    custom_font = None
    if font_path:
        custom_font = FontProperties(fname=font_path)
    
    fig, ax = plt.subplots(figsize=(3.2, 1.8))
    absent = max(sanctioned - present, 0)

    sizes = [present, absent]
    colors_list = ["#2ca02c", (0, 0, 0, 0)]

    wedges, texts = ax.pie(
        sizes,
        colors=colors_list,
        pctdistance=0.75,
        labeldistance=1.15
    )

    centre_circle = plt.Circle((0, 0), 0.60, color='white', fc='white', linewidth=0)
    ax.add_artist(centre_circle)

    if custom_font:
        ax.set_title(title, fontproperties=custom_font, fontsize=11, pad=5)
    else:
        ax.set_title(title, fontsize=11, fontweight='normal', pad=5)

    # Remove default labels on the pie
    for t in texts:
        t.set_visible(False)

    # Add external legend 
    if custom_font:
        ax.legend(
        [wedges[0]],
        ["ਭਰਿਆ"],
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=8,
        prop = custom_font
    )
    else:
        ax.legend(
            [wedges[0]],
            ["Filled"],
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=8
        )

    ax.axis("equal")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)

    return buf

def bar_graph(x, y, label, colour, ylim=None):

    # Create DF
    g_df = pd.DataFrame({
        "Label": x,
        "Count": y
    })
    scale = alt.Scale(domain=[0, ylim]) if ylim else alt.Undefined

    chart = (
        alt.Chart(g_df)
        .mark_bar()
        .encode(
            x=alt.X("Label:N", title="", axis=alt.Axis(labelAngle=45)),
            y=alt.Y("Count:Q", title=label, scale=scale),
            color=alt.value(colour)
        )
        .properties(height=400)
    )
    return chart

def comparison_chart(first, latest, col1, col2, detected_lang, first_date, latest_date):

    first = first[[col1, col2]].dropna()
    latest = latest[[col1, col2]].dropna()
    first.set_index(col1, inplace = True)
    latest.set_index(col1, inplace = True)

    line1 = first[col2].rename(first_date)
    line2 = latest[col2].rename(latest_date)


    combined_df = pd.concat([line1,line2],axis = 1)
    combined_df = combined_df.reset_index()
    
    cols_to_fix = [first_date, latest_date]
    for col in cols_to_fix:
        if combined_df[col].dtype == 'object':
            combined_df[col] = combined_df[col].str.rstrip('%').astype(float)

    col1 = combined_df.columns[0]
    combined_df = combined_df.rename(columns={col1: 'Category'})
    if detected_lang == 'pa':
        combined_df['Category'] = get_translation_batch(combined_df['Category'].astype(str).tolist())
    source = combined_df.melt(
        id_vars=['Category'], 
        value_vars=[first_date, latest_date],
        var_name='Assessment_Date', 
        value_name='Score'
    )
    color_map = {
    first_date: '#94c0e1',  # Blue
    latest_date: '#144481'  # Orange
    }
    if detected_lang == 'pa':
        chart = alt.Chart(source).mark_bar().encode(
            x=alt.X('Category', axis=alt.Axis(labelAngle=90, title=get_translation(col1))),
            y=alt.Y('Score', scale=alt.Scale(domain=[0, 100]), title='ਸਕੋਰ (%)'),
            color=alt.Color(
            'Assessment_Date', 
            scale=alt.Scale(
                domain=list(color_map.keys()), 
                range=list(color_map.values())
            ),
            legend=None 
        ),
            xOffset='Assessment_Date',                        
            tooltip=['Category', 'Assessment_Date', 'Score']
        ).properties().interactive()
    else:
        chart = alt.Chart(source).mark_bar().encode(
            x=alt.X('Category', axis=alt.Axis(labelAngle=90, title=col1)),
            y=alt.Y('Score', scale=alt.Scale(domain=[0, 100]), title='Score (%)'),
            color=alt.Color(
            'Assessment_Date', 
            scale=alt.Scale(
                domain=list(color_map.keys()), 
                range=list(color_map.values())
            ),
            legend=None 
        ),
            xOffset='Assessment_Date',                        
            tooltip=['Category', 'Assessment_Date', 'Score']
        ).properties().interactive()

    return chart

def draw_paragraph_on_pdf(c, paragraph: str, y_position: float,
                          page_size=A4, left: int = 50, right: int = 50,
                          leading: int = 14, font_name: str = "Helvetica", font_size: int = 11):
    """
    Draw a wrapped paragraph on the given canvas `c`.
    Returns the new y_position after drawing.
    - c: reportlab.pdfgen.canvas.Canvas
    - paragraph: text
    - y_position: current vertical cursor (points from bottom)
    - left, right: horizontal margins (points)
    """
    width, height = page_size
    styles = getSampleStyleSheet()

    style_custom = ParagraphStyle(
        'CustomStyle',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=font_size,
        leading=leading
    )

    # ensure paragraph is a safe string
    text = paragraph or ""
    p = Paragraph(text, style_custom)

    max_width = width - left - right
    w, h = p.wrap(max_width, y_position) 

    if y_position - h < 72:  
        c.showPage()
        y_position = height - 50  

    p.drawOn(c, left, y_position - h)
    y_position = y_position - h - 10  
    return y_position

def render_domain_section(domain_no: int,
                        domain_title: str,
                          score_col: str,
                          perc_col: str,
                          remarks: list,
                          report: pd.DataFrame,
                          data: pd.DataFrame,
                          detected_lang: str,
                          font_name: str = None,
                          bold_name: str = None,
                          *,
                          pdf_canvas=None,
                          y_position: float = None,
                          page_size=A4,
                          st_container=None):
    """
    Render one domain to PDF and/or Streamlit.
    st_container may be:
      - None (skip Streamlit output)
      - a Streamlit module (the usual `st`) -> the function will create a container internally
      - a Streamlit container object (returned by st.container()) -> the function will use it directly
    """
    cleaned_report = report[[domain_title, score_col, perc_col]].dropna()
    rename_col = {score_col : "Score", perc_col: "Score Percentage"}
    clean_report = cleaned_report.rename(columns = rename_col)
    # --- STREAMLIT OUTPUT ---
    if st_container is not None:
        if hasattr(st_container, "container") and callable(getattr(st_container, "container")):
            container = st_container.container()
        else:
            container = st_container

        with container:
            if not clean_report.empty:
                st.dataframe(clean_report.reset_index(drop=True))
                if detected_lang == 'pa':
                    x = get_translation_batch(clean_report[domain_title].astype(str).tolist())
                    label = get_translation(score_col)
                else:
                    x = clean_report[domain_title].astype(str)
                    label =score_col
                y = clean_report["Score"]               
                chart_buf = bar_graph(x, y, label, colour="#4682B4", ylim = 4)
                st.altair_chart(chart_buf, use_container_width=True)
            else:
                st.write("No data available for this domain.")
            if detected_lang == 'pa':
                st.markdown("#### ਟਿੱਪਣੀਆਂ:")
            else:
                st.markdown("#### Remarks:")
            for remark_name, col in remarks:
                val = data[col].iloc[0] if col in data else ""
                if settings.ENABLE_REMARK_REPHRASE:
                    val = cached_rephrase(val)
                if detected_lang == 'pa':
                    val = get_translation(val)
                    remark_name = get_translation(remark_name)
                st.markdown(f"- **{remark_name}:** {val}")

    # --- PDF OUTPUT ---
    if pdf_canvas is not None and y_position is not None:
        c = pdf_canvas
        width, height = page_size
        styles = getSampleStyleSheet()
        style_custom = ParagraphStyle(
            'CustomStyle',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=11,
            leading=14
        )
        if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
            c.setFont(bold_name, 14)
            c.drawString(50, y_position, f"ਡੋਮੇਨ {domain_no}: {get_translation(domain_title)}")
        else:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, f"Domain {domain_no}: {domain_title}")
        y_position -= 25

        c.setFont(font_name, 10)
        for _, row_report in cleaned_report.iterrows():
            text = f"{row_report[domain_title]} | {score_col}: {row_report[score_col]} | {perc_col}: {row_report[perc_col]}"
            c.drawString(50, y_position, text)
            y_position -= 15
            if y_position < 120:
                c.showPage()
                y_position = height - 50

        y_position -= 10

        if not cleaned_report.empty:
            if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
                x = get_translation_batch(cleaned_report[domain_title].astype(str).tolist())
                label = get_translation(str(score_col))
            else:
                x = cleaned_report[domain_title].astype(str)
                label = str(score_col)
            y = cleaned_report[score_col]
            if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
                chart_buf = bar_graph_pdf(x, y, label, ylim=4, colour="#4682B4", font_path=settings.PUNJABI_REGULAR_FONT_PATH)
            else:
                chart_buf = bar_graph_pdf(x, y, label, ylim=4, colour="#4682B4")
            if y_position < 300:
                c.showPage()
                y_position = height - 50
            img = ImageReader(chart_buf)
            c.drawImage(img, 50, y_position - 250, width=500, height=270)
            y_position -= 300

        if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
            c.setFont(bold_name, 14)
            c.drawString(50, y_position, "ਟਿੱਪਣੀਆਂ:")
        else:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, "Remarks:")
        y_position -= 15

        max_width = page_size[0] - 100
        if y_position < 100:
            c.showPage()
            y_position = height - 50
        for remark_name, col in remarks:
            value = str(data[col].iloc[0]) if col in data else ""
            if settings.ENABLE_REMARK_REPHRASE:
                value = cached_rephrase(value)
            if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
                value = get_translation(value)
                remark_name = get_translation(remark_name)
            p = Paragraph(f"<b>{remark_name}:</b> {value}", style_custom)
            w, h = p.wrap(max_width, 1000)
            if y_position - h < 100:
                c.showPage()
                y_position = height - 50
            p.drawOn(c, 50, y_position - h)
            y_position -= h + 5

        y_position -= 20
        if y_position < 100:
            c.showPage()
            y_position = height - 50

        return y_position
    
def generate_pdf(row, latest_date, school_info, overall_report, report, data, detected_lang):
    """Encapsulates the heavy PDF generation logic."""
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4
    y_position = height - 50 

    regular_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
        regular_font = "Gurmukhi"
        bold_font = "Gurmukhi-Bold"
    
    # Header
    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
        raw_name = str(row['School_Name']).strip().upper()
        c.setFont(bold_font, 16)
        c.drawCentredString(width / 2, y_position, get_translation(raw_name))
    else:
        c.setFont(bold_font, 16)
        c.drawCentredString(width / 2, y_position, row['School_Name'].upper())
    
    y_position -= 30

    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
        c.setFont(bold_font, 16)
        c.drawString(50 , y_position, f"ਜ਼ਿਲ੍ਹਾ: {get_translation(row['District'])} - ਬਲਾਕ: {get_translation(row['Block'])}")
        y_position -= 20
        c.drawString(50, y_position, f"UDISE ਕੋਡ: {row['UDISE_Code']}")
        y_position -= 20
        c.drawString(50, y_position, f"ਮੁਲਾਂਕਣ ਤਾਰੀਖ: {pd.to_datetime(latest_date).date()}")
    else:
        c.setFont(bold_font, 12)
        c.drawString(50, y_position, f"District: {row['District']} - Block: {row['Block']}")
        y_position -= 20
        c.drawString(50, y_position, f"UDISE Code: {row['UDISE_Code']}")
        y_position -= 20
        c.drawString(50, y_position, f"Assessment Date: {pd.to_datetime(latest_date).date()}")
    y_position -= 20
    c.line(50, y_position, width - 50, y_position)
    y_position -= 30 

    # Info Paragraph
    paragraph = info_para(school_info)
    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
        paragraph = get_translation(paragraph)
    y_position = draw_paragraph_on_pdf(c, paragraph, y_position, page_size=A4, left=50, right=50, font_name = regular_font)
    # Charts
    if detected_lang =='pa' and settings.PUNJABI_FONT_LOADED:
        gender_buf = bar_graph_pdf(["ਕੁੜੀਆਂ", "ਮੁੰਡੇ"], [school_info['Total_girls'].iloc[0], school_info['Total_boys'].iloc[0]], "ਕੁੱਲ ਗਿਣਤੀ", colour="#4682B4", title="ਲਿੰਗ ਅਨੁਪਾਤ ਬਾਰ", font_path=settings.PUNJABI_REGULAR_FONT_PATH)
        teacher_buf = teacher_pie_buf(school_info['Teachers_Present'].iloc[0], school_info['Teacher_Positions_Sanctioned'].iloc[0], "ਮਨਜ਼ੂਰਸ਼ੁਦਾ ਬਨਾਮ ਭਰੇ ਹੋਏ ਅਧਿਆਪਕ ਅਹੁਦੇ", font_path= settings.PUNJABI_REGULAR_FONT_PATH)
    else:
        gender_buf = bar_graph_pdf(["Girls", "Boys"], [school_info['Total_girls'].iloc[0], school_info['Total_boys'].iloc[0]], "Count", colour="#4682B4", title="Gender split bar")
        teacher_buf = teacher_pie_buf(school_info['Teachers_Present'].iloc[0], school_info['Teacher_Positions_Sanctioned'].iloc[0], "Sanctioned vs. Filled Teachers Positions" )
    
    img_w, img_h = 220, 120
    c.drawImage(ImageReader(gender_buf), 50, y_position - img_h, width=img_w, height=img_h, mask='auto')
    c.drawImage(ImageReader(teacher_buf), 50 + img_w + 20, y_position - img_h, width=img_w, height=img_h, mask='auto')
    y_position = y_position - img_h - 20

    # Domains
    for _, domain_title, score_col, perc_col, remarks in constants.domains:
        y_position = render_domain_section(_, domain_title, score_col, perc_col, remarks, report, data, detected_lang, font_name = regular_font, bold_name=bold_font, pdf_canvas=c, y_position=y_position, page_size=A4)

    # Overall Score
    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
        c.setFont(bold_font, 14)
        c.drawString(50, y_position, "ਕੁੱਲ ਸਕੋਰ:")
    else:
        c.setFont(bold_font, 14)
        c.drawString(50, y_position, "Overall Score:")
    y_position -= 25
    for i, row_overall in overall_report.iterrows():
        text = f"{row_overall['Domain']} | Percentage: {row_overall['Score Percentage']}"
        c.setFont("Helvetica", 10)
        c.drawString(50, y_position, text)
        y_position -= 15
        if y_position < 150:
            c.showPage(); y_position = height - 50

    y_position -= 25
    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
        x = get_translation_batch(overall_report['Domain'].astype(str).tolist())
        chart_buf = bar_graph_pdf(x, overall_report['Score Percentage'].str.rstrip('%').astype(float), "ਸਕੋਰ ਪ੍ਰਤੀਸ਼ਤ", colour="#3F51B5", ylim=100, font_path=settings.PUNJABI_REGULAR_FONT_PATH)
    else:
        chart_buf = bar_graph_pdf(overall_report['Domain'].astype(str), overall_report['Score Percentage'].str.rstrip('%').astype(float), "Score Percentage", colour="#3F51B5", ylim=100)

    if y_position < 300:
        c.showPage(); y_position = height - 50
    c.drawImage(ImageReader(chart_buf), 50, y_position - 250, width=500, height=270)
    
    c.showPage()
    c.save()
    
    pdf_buffer.seek(0)
    return pdf_buffer



# =============================
# 7. VIEW RENDERERS
# =============================  

@st.fragment
def render_latest_view(data, row, code, long_report, school_info, detected_lang):
    """Renders the Single Assessment View (Checkboxes, PDF, Charts)."""
    
    # 1. Process base data first
    latest_date = data["Assessment_Date"].max()
    report = pivot_table(code, long_report, latest_date)
    Overall = overall_score(report)
    
    # Safely filter school info without overwriting the parameter
    school_info_filtered = school_info[(school_info['UDISE_Code'] == code) & (school_info['Assessment_Date'] == latest_date)]

    st.markdown("---")

    # 2. Render Header & PDF unconditionally
    if detected_lang == 'pa':
        st.header(get_translation(row['School_Name'].upper()))   
    else: 
        st.header(row['School_Name'].upper())

    c1, c2 = st.columns(2)      
    if detected_lang == 'pa':
        c1.subheader(f"ਜ਼ਿਲ੍ਹਾ: {get_translation(row['District'])} "); c1.write(f"ਮੁਲਾਂਕਣ ਤਾਰੀਖ: {pd.to_datetime(latest_date).date()}")
        c2.subheader(f"ਬਲਾਕ: {get_translation(row['Block'])}"); c2.write(f"Udise ਕੋਡ: {row['UDISE_Code']}")
    else:
        c1.subheader(f"District: {row['District']} "); c1.write(f"Assessment Date: {pd.to_datetime(latest_date).date()}")
        c2.subheader(f"Block: {row['Block']}"); c2.write(f"Udise Code: {row['UDISE_Code']}")

    # PDF Download always available
    pdf_data = cached_generate_pdf(row, latest_date, school_info_filtered, Overall, report, data, detected_lang)
    if detected_lang == 'pa':
        st.download_button("ਰਿਪੋਰਟ PDF ਡਾਊਨਲੋਡ ਕਰੋ", pdf_data, f"{row['School_Name'].upper()}_report.pdf", "application/pdf", key=f"dl_{code}_{time.time()}")
    else:
        st.download_button("Download Report PDF", pdf_data, f"{row['School_Name'].upper()}_report.pdf", "application/pdf", key=f"dl_{code}_{time.time()}")
    
    st.markdown("---")
    
    # 3. Render School Info & Base Graphs unconditionally
    if detected_lang == 'pa':
        st.markdown(get_translation(info_para(school_info_filtered)))
    else:
        st.markdown(info_para(school_info_filtered))
    
    gc1, gc2 = st.columns([1,1])
    with gc1:
        if detected_lang == 'pa':
            g_df = pd.DataFrame({"ਲਿੰਗ":["ਕੁੜੀਆਂ","ਲੜਕੇ"], "ਗਿਣਤੀ":[school_info_filtered['Total_girls'].iloc[0], school_info_filtered['Total_boys'].iloc[0]]})
            if (school_info_filtered['Total_girls'].iloc[0] + school_info_filtered['Total_boys'].iloc[0]) > 0:
                st.altair_chart(alt.Chart(g_df).mark_bar().encode(x="ਲਿੰਗ:N", y="ਗਿਣਤੀ:Q", color=alt.Color("ਲਿੰਗ:N", legend=None)).properties(height=200), use_container_width=True)
        else:
            g_df = pd.DataFrame({"Gender":["Girls","Boys"], "Count":[school_info_filtered['Total_girls'].iloc[0], school_info_filtered['Total_boys'].iloc[0]]})
            if (school_info_filtered['Total_girls'].iloc[0] + school_info_filtered['Total_boys'].iloc[0]) > 0:
                st.altair_chart(alt.Chart(g_df).mark_bar().encode(x="Gender:N", y="Count:Q", color=alt.Color("Gender:N", legend=None)).properties(height=200), use_container_width=True)
    with gc2:
        tp, sanctioned = int(school_info_filtered['Teachers_Present'].iloc[0] or 0), int(school_info_filtered['Teacher_Positions_Sanctioned'].iloc[0] or 0)
        if sanctioned > 0:
            fill_pct = round((tp / sanctioned) * 100, 1)
            if detected_lang == 'pa':
                t_df = pd.DataFrame({"ਸਥਿਤੀ": ["ਭਰਿਆ", "ਬਾਕੀ"], "Count": [tp, max(sanctioned - tp, 0)], "FillPct": [fill_pct, fill_pct]})
                st.altair_chart(
                    alt.Chart(t_df).mark_arc(innerRadius=40).encode(
                        theta="Count:Q", 
                        color=alt.Color("ਸਥਿਤੀ:N", scale=alt.Scale(domain=["ਭਰਿਆ"], range=["#2ca02c"])), 
                        order=alt.Order("ਸਥਿਤੀ", sort="descending")
                    ).properties(height=200), use_container_width=True
                )
            else:
                t_df = pd.DataFrame({"Status": ["Filled", "Remainder"], "Count": [tp, max(sanctioned - tp, 0)], "FillPct": [fill_pct, fill_pct]})
                st.altair_chart(
                    alt.Chart(t_df).mark_arc(innerRadius=40).encode(
                        theta="Count:Q", 
                        color=alt.Color("Status:N", scale=alt.Scale(domain=["Filled"], range=["#2ca02c"])), 
                        order=alt.Order("Status", sort="descending")
                    ).properties(height=200), use_container_width=True
                )
        else:
            if detected_lang == 'pa':
                st.caption("ਕੋਈ ਅਧਿਆਪਕ ਡੇਟਾ ਨਹੀਂ।")
            else:
                st.caption("No teacher-sanctioned data.")

    st.markdown("---")

    # 4. NOW render the checkboxes for detailed domain drill-down
    options = ["Safety and Hygiene / ਸੁਰੱਖਿਆ ਅਤੇ ਸਫਾਈ ", "Stimulating School Environment / ਸਕੂਲ ਦੇ ਵਾਤਾਵਰਣ ਨੂੰ ਉਤੇਜਿਤ ਕਰਨਾ", "Physical Development Opportunities / ਸਰੀਰਕ ਵਿਕਾਸ ਦੇ ਮੌਕੇ", "Smart School Facilities / ਸਮਾਰਟ ਸਕੂਲ ਸਹੂਲਤਾਂ"]
    if detected_lang == 'pa':
        st.write("ਆਪਣਾ ਡੋਮੇਨ ਚੁਣੋ (ਵਿਸਤ੍ਰਿਤ ਰਿਪੋਰਟ ਲਈ):")
    else:
        st.write("Select your Domain(s) for detailed view:")

    selected_options = [opt for opt in options if st.checkbox(opt, key=f"{code}_{opt}")]

    # Only render detailed breakdown if boxes are checked
    if selected_options:
        for _, index in enumerate(selected_options):
            selected_domain_data = next((d for d in constants.domains if d[1] == index.split("/",1)[0].strip()), None)
            if selected_domain_data:
                dom_id, domain_title, score_col, perc_col, remarks = selected_domain_data
                if detected_lang == 'pa':
                    st.subheader(f"ਡੋਮੇਨ {_+1}: {domain_title} / {get_translation(domain_title)}")
                else:
                    st.subheader(f"Domain {_+1}: {domain_title}")
                render_domain_section(_, domain_title, score_col, perc_col, remarks, report, data, detected_lang, st_container=st, page_size=A4)

    # 5. Overall Score Chart unconditionally
    if detected_lang == 'pa':
        st.subheader("ਕੁੱਲ ਸਕੋਰ:")
    else:
        st.subheader("Overall Score:")
    st.dataframe(Overall.reset_index(drop=True))
    if detected_lang == 'pa':
        st.altair_chart(bar_graph(get_translation_batch(Overall['Domain'].astype(str).tolist()), Overall['Score Percentage'].str.rstrip('%').astype(float), "ਸਕੋਰ ਪ੍ਰਤੀਸ਼ਤ", ylim=100, colour="#3F51B5"), use_container_width=True)
    else:
        st.altair_chart(bar_graph(Overall['Domain'].astype(str), Overall['Score Percentage'].str.rstrip('%').astype(float), "Score Percentage", ylim=100, colour="#3F51B5"), use_container_width=True)

        
def render_comparative_view(data, row, code, long_report, School_Info, detected_lang):
    """Renders the Comparative Analysis View."""
    latest_date, earlier_date = data["Assessment_Date"].max(), data["Assessment_Date"].min()
    report_first, report_latest = pivot_table(code, long_report, earlier_date), pivot_table(code, long_report, latest_date)
    Overall_first, Overall_latest = overall_score(report_first), overall_score(report_latest)
    
    si_early = School_Info[(School_Info['UDISE_Code'] == code) & (School_Info['Assessment_Date'] == earlier_date)].reset_index()
    si_late = School_Info[(School_Info['UDISE_Code'] == code) & (School_Info['Assessment_Date'] == latest_date)].reset_index()

    if detected_lang == 'pa':
        st.header(get_translation(row['School_Name'].upper()))   
    else: 
        st.header(row['School_Name'].upper())

    c1, c2 = st.columns(2)
    if detected_lang == 'pa':
        c1.subheader(f"ਜ਼ਿਲ੍ਹਾ: {get_translation(row['District'])} "); c1.write(f"Udise ਕੋਡ: {row['UDISE_Code']}")
        c2.subheader(f"ਬਲਾਕ: {get_translation(row['Block'])}") 
    else:
        c1.subheader(f"District: {row['District']}"); c1.write(f"Udise Code: {row['UDISE_Code']}")
        c2.subheader(f"Block: {row['Block']}")
    st.markdown("---")

    first, second = str(pd.to_datetime(earlier_date).date()), str(pd.to_datetime(latest_date).date())
    
    # --- 1. KEY HIGHLIGHTS (METRICS) ---
    if detected_lang == 'pa':
        st.header("ਸਕੂਲ ਦੇ ਅੰਕੜੇ:")
        st.subheader("ਮੁੱਖ ਹਾਈਲਾਈਟਸ")
    else:
        st.header("School Stats:")
        st.subheader("Key Highlights")
    
    # Calculate values
    stu_curr = si_late.loc[0,"Total_student"]
    stu_prev = si_early.loc[0,"Total_student"]
    tch_curr = si_late.loc[0,"Teachers_Present"]
    tch_prev = si_early.loc[0,"Teachers_Present"]

    k1, k2 = st.columns(2)
    if detected_lang == 'pa':
        k1.metric("ਕੁੱਲ ਵਿਦਿਆਰਥੀ", stu_curr, f"{stu_curr - stu_prev} from {first}")
        k2.metric("ਅਧਿਆਪਕ", tch_curr, f"{tch_curr - tch_prev} from {first}")
    else:
        k1.metric("Total Students", stu_curr, f"{stu_curr - stu_prev} from {first}")
        k2.metric("Teachers", tch_curr, f"{tch_curr - tch_prev} from {first}")
    st.markdown("---")


    # --- 2. DETAILED COMPARISON ---
    if detected_lang == 'pa':
        st.subheader("ਵਿਸਤ੍ਰਿਤ ਤੁਲਨਾ")
    else:
        st.subheader("Detailed Comparison")
    
    comp_data = {
        "Category": ["Students", "Teachers"],
        first: [stu_prev, tch_prev],
        second: [stu_curr, tch_curr]
    }
    df_comp = pd.DataFrame(comp_data)
    df_melted = df_comp.melt("Category", var_name="Year", value_name="Count")

    # Base chart
    base = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('Year:N', axis=alt.Axis(title=None)),
        color=alt.Color('Year:N', legend=None),
        tooltip=['Category', 'Year', 'Count']
    ).properties(width=150)

    if detected_lang == 'pa':
        students_chart = base.transform_filter(alt.datum.Category == 'Students').encode(
            y=alt.Y('Count:Q', title='ਕੁੱਲ ਗਿਣਤੀ')
        ).properties(title="ਵਿਦਿਆਰਥੀ")

        teachers_chart = base.transform_filter(alt.datum.Category == 'Teachers').encode(
            y=alt.Y('Count:Q', title=None)
        ).properties(title="ਅਧਿਆਪਕ")
    else:
        students_chart = base.transform_filter(alt.datum.Category == 'Students').encode(
            y=alt.Y('Count:Q', title='Total Count')
        ).properties(title="Students")

        teachers_chart = base.transform_filter(alt.datum.Category == 'Teachers').encode(
            y=alt.Y('Count:Q', title=None)
        ).properties(title="Teachers")

    final_chart = (students_chart | teachers_chart).resolve_scale(y='independent')

    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.altair_chart(final_chart, use_container_width=False)
    
    st.markdown("---")

    # --- 3. DOMAIN COMPARISON CHARTS ---
    if detected_lang == 'pa':
        st.header("ਡੋਮੇਨ ਤੁਲਨਾ:")
    else:
        st.header("Domain Comparison:")
    # Helper for legend
    c1_hex, c2_hex = "#94c0e1", "#144481"
    if detected_lang == 'pa':
        legend_html = f'<div style="display: flex; align-items: center; margin-bottom: 15px;"><span style="font-weight: bold; margin-right: 15px;">ਮੁਲਾਂਕਣ ਦੀ ਤਾਰੀਖ:</span><span style="background-color: {c1_hex}; width: 12px; height: 12px; display: inline-block; margin-right: 5px; border-radius: 2px;"></span><span style="margin-right: 15px;">{first}</span><span style="background-color: {c2_hex}; width: 12px; height: 12px; display: inline-block; margin-right: 5px; border-radius: 2px;"></span><span>{second}</span></div>'
    else:
        legend_html = f'<div style="display: flex; align-items: center; margin-bottom: 15px;"><span style="font-weight: bold; margin-right: 15px;">Assessment date:</span><span style="background-color: {c1_hex}; width: 12px; height: 12px; display: inline-block; margin-right: 5px; border-radius: 2px;"></span><span style="margin-right: 15px;">{first}</span><span style="background-color: {c2_hex}; width: 12px; height: 12px; display: inline-block; margin-right: 5px; border-radius: 2px;"></span><span>{second}</span></div>'
        
    st.markdown(legend_html, unsafe_allow_html=True)
    
    cols = st.columns(2)
    if detected_lang == 'pa':
        with cols[0]: st.altair_chart(comparison_chart(report_first, report_latest, "Safety and Hygiene", "Domain1 Score Percentage", detected_lang= detected_lang, first_date= first, latest_date= second), use_container_width=True)
        with cols[1]: st.altair_chart(comparison_chart(report_first, report_latest, "Stimulating School Environment", "Domain2 Score Percentage", detected_lang=detected_lang, first_date= first, latest_date= second), use_container_width=True)
    else:
        with cols[0]: st.altair_chart(comparison_chart(report_first, report_latest, "Safety and Hygiene", "Domain1 Score Percentage", detected_lang= detected_lang, first_date= first, latest_date= second), use_container_width=True)
        with cols[1]: st.altair_chart(comparison_chart(report_first, report_latest, "Stimulating School Environment", "Domain2 Score Percentage", detected_lang= detected_lang, first_date= first, latest_date= second), use_container_width=True)

    cols2 = st.columns(2)
    with cols2[0]: st.altair_chart(comparison_chart(report_first, report_latest, "Physical Development Opportunities", "Domain3 Score Percentage", detected_lang= detected_lang, first_date= first, latest_date= second), use_container_width=True)
    with cols2[1]: st.altair_chart(comparison_chart(report_first, report_latest, "Smart School Facilities", "Domain4 Score Percentage", detected_lang= detected_lang, first_date= first, latest_date= second), use_container_width=True)

    if detected_lang == 'pa':
        st.header("ਸਮੁੱਚੀ ਡੋਮੇਨ ਤੁਲਨਾ:")
    else:
        st.header("Overall Domain Comparison:")
    st.altair_chart(comparison_chart(Overall_first, Overall_latest, "Domain", "Score Percentage", detected_lang= detected_lang, first_date= first, latest_date= second), use_container_width=True)

# =============================
# 8. VILLAGE REPORT RENDERERS
# ============================= 

def render_village_view(village_data: dict, detected_lang: str):
    """Renders the single assessment view for the Village logic in Streamlit."""
    st.markdown("---")
    
    if detected_lang == 'pa':
        v_name = get_translation(village_data.get('village_name', 'Unknown'))
        st.header(f"ਪਿੰਡ: {v_name}")
        st.subheader(f"ਗ੍ਰਾਮ ਪੰਚਾਇਤ: {get_translation(village_data.get('gp_name', ''))} | ਬਲਾਕ: {get_translation(village_data.get('block_name', ''))}")
    else:
        v_name = village_data.get('village_name', 'Unknown').upper()
        st.header(f"Village: {v_name}")
        st.subheader(f"Gram Panchayat: {village_data.get('gp_name', '')} | Block: {village_data.get('block_name', '')}")

    st.info("Village development metrics (Health, Sanitation, Livelihoods, etc.) will be displayed here once database integration is finalized.[cite: 1]")
    
    # Fallback to show raw dictionary items temporarily until you map the DB schema
    st.dataframe(pd.DataFrame([village_data]))


def generate_village_pdf(village_data: dict, detected_lang: str, insights: str):
    """Generates the Village Status Report PDF."""
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4
    y_position = height - 50 

    regular_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
        regular_font = "Gurmukhi"
        bold_font = "Gurmukhi-Bold"
    
    # 1. Header
    c.setFont(bold_font, 16)
    v_name = village_data.get('village_name', 'Unknown').upper()
    
    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
        c.drawCentredString(width / 2, y_position, f"{get_translation(v_name)} ਸਥਿਤੀ ਰਿਪੋਰਟ")
    else:
        c.drawCentredString(width / 2, y_position, f"{v_name} STATUS REPORT")
    
    y_position -= 40
    
    # 2. General Village Info
    c.setFont(regular_font, 12)
    info_text = f"Gram Panchayat: {village_data.get('gp_name', '')} | Block: {village_data.get('block_name', '')}"
    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
        info_text = get_translation(info_text)
    
    c.drawString(50, y_position, info_text)
    y_position -= 20
    c.line(50, y_position, width - 50, y_position)
    y_position -= 30

    # 3. AI Insights section
    c.setFont(bold_font, 14)
    header_text = "AI Insights & Recommendations"
    if detected_lang == 'pa' and settings.PUNJABI_FONT_LOADED:
        header_text = "AI-ਦੁਆਰਾ ਤਿਆਰ ਕੀਤੀਆਂ ਗਈਆਂ ਜਾਣਕਾਰੀਆਂ"
        
    c.drawString(50, y_position, header_text)
    y_position -= 20
    
    # Draw insights text paragraph
    y_position = draw_paragraph_on_pdf(
        c, 
        insights, 
        y_position, 
        page_size=A4, 
        left=50, 
        right=50, 
        font_name=regular_font
    )

    c.showPage()
    c.save()
    
    pdf_buffer.seek(0)
    return pdf_buffer