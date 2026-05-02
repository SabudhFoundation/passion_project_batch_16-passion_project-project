from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra = "ignore" 
    )

    # --- Global App Settings ---
    PAGE_TITLE: str 
    PAGE_LAYOUT: str

    # --- School Report Settings ---
    SCHOOL_FILENAME: str 
    SCHOOL_LIST_PATH: str
    FUZZY_SCORE_THRESHOLD: int

    # --- API & AI Settings ---
    API_KEY: str
    GEMINI_MODEL_NAME: str
    MODEL_NAME: str
    MODEL_TEMP: float
    ENABLE_REMARK_REPHRASE: bool
    MAX_REPHRASE_RETRIES: int

    # --- Village Report Settings ---

    MONGO_URI: str 

    # --- Font Settings ---
    PUNJABI_REGULAR_FONT_PATH: str
    PUNJABI_BOLD_FONT_PATH: str
    PUNJABI_FONT_LOADED: bool

settings = Settings()