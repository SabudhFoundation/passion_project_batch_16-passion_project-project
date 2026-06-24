[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23178784&assignment_repo_type=AssignmentRepo)

# Village & School Intelligence Portal

A unified, LLM-powered Streamlit dashboard for generating interactive, bilingual status reports for schools and villages across Punjab. Leverages automated insights, natural language understanding, and smart data preprocessing — supporting both English and Punjabi (ਪੰਜਾਬੀ).

---

## Key Features

- **Unified Master Portal:** Interactive Streamlit interface (`main.py`) that routes between School and Village reporting workflows.
- **Bilingual Support (English & Punjabi):** Dynamic UI translation, automated data processing, and native Gurmukhi font rendering for PDF exports.
- **Conversational AI & Insights:** LLM integration via structured prompts for natural language queries and automated comparative insights.
- **Smart Fuzzy Matching:** Uses `difflib` to suggest closest matching village or school names when a query fails.
- **Government Scheme Data:** Scrapers for Jal Jeevan Mission (JJM), MNREGA, and Swachh Bharat Mission (SBM) data pipelines.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit, streamlit-aggrid |
| Backend | Python 3.9+, Pandas |
| AI / NLP | Google Gemini (LLM), deep-translator |
| Database | MongoDB |
| PDF Generation | fpdf / reportlab |
| Configuration | pydantic, pydantic-settings |

---

## 📂 Project Structure

```
├── main.py                     <- Streamlit app entry point (run this)
├── requirements.txt            <- Python dependencies
├── .env.sample                 <- Template for environment variables (copy to .env)
├── meetings.md                 <- Mentor alignment meeting logs
├── weekly_report.md            <- Internal weekly progress tracker
│
├── screens/                    <- Streamlit UI screen modules
│   ├── school_report.py        <- School data reporting UI and controls
│   └── village_report.py       <- Village data reporting UI and controls
│
├── src/                        <- Core backend package
│   ├── __init__.py
│   ├── config.py               <- App configuration and Pydantic settings
│   ├── constants.py            <- Global constants and environment variables
│   ├── database.py             <- MongoDB client and connection handlers
│   ├── llm.py                  <- Google Gemini LLM integration layer
│   ├── prompts.py              <- System prompts for LLM querying
│   ├── utils.py                <- Shared helper functions
│   │
│   ├── data/
│   │   ├── processed/          <- Cleaned and transformed datasets
│   │   └── raw/                <- Original immutable source data
│   │
│   └── punjabi_font/           <- Gurmukhi fonts for PDF generation
│       ├── NotoSansGurmukhi-Bold.ttf
│       └── NotoSansGurmukhi-Regular.ttf
│
├── notebooks/                  <- Data scraping and merging scripts by scheme
│   ├── Jal Jeevan Mission/
│   │   ├── scrape_data.py
│   │   ├── merge_csv.py
│   │   └── LastGoodCode.py
│   ├── MNREGA/
│   │   ├── scrape_data.py
│   │   └── merge_csv.py
│   └── Swacch Bharat Mission/
│       ├── scraped_data.py
│       └── merge_csv.py
│
└── reports/                    <- Target output directory for generated assets.
    ├── figures/                <- Generated graphics and visualization exports.
    ├── final_project_report/   <- Final definitive report PDFs and supporting documents.
    └── presentation/           <- Supporting presentation files.
```

---

## Getting Started

**Python version:** 3.9+

### 1. Clone the Repository

```bash
git clone https://github.com/SabudhFoundation/passion-project-village_status_report_generation_tool.git
cd passion-project-village_status_report_generation_tool
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Copy the provided sample file and fill in your values:

```bash
cp .env.sample .env
```

All required variables and their descriptions are documented in `.env.sample`. At minimum, update:

```env
API_KEY=your_google_gemini_api_key_here
MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/
```

### 4. Run the Application

Always run from the root directory so Python resolves sub-modules correctly:

```bash
streamlit run main.py
```

---

## Usage Guide

1. **Home Portal:** Select either **School Tool** or **Village Tool** from the landing page.
2. **Search:** Type a village or school name in English or Punjabi. Fuzzy matching will suggest close results if the exact name is not found.
3. **Select & Compare:**
   - Select **1 row** → generates a standard status report.
   - Select **2 rows** → unlocks the Comparative AI Insights dashboard.
4. **Download:** Export the report and AI insights as a PDF using the download button.
5. **Return:** Use the "Back to Home" button in the sidebar to reset session state and return to the main menu.

---

## Data Pipelines

Government scheme data is collected via scripts in `notebooks/`:

| Scheme | Script Location |
|---|---|
| Jal Jeevan Mission (JJM) | `notebooks/Jal Jeevan Mission/` |
| MNREGA | `notebooks/MNREGA/` |
| Swachh Bharat Mission (SBM) | `notebooks/Swacch Bharat Mission/` |

Each scheme folder contains a `scrape_data.py` to collect raw data and a `merge_csv.py` to consolidate it.
