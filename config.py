from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra = "ignore" 
    )

    PAGE_TITLE: str = "Village Status Report Generation" 
    PAGE_LAYOUT: str = "wide"
    

    # ⬅️ ADDED: Database connection string for MongoDB
    MONGO_URI: str 

    API_KEY: str
    GEMINI_MODEL_NAME: str
    MODEL_NAME: str = "gemini-2.5-flash" # Fallback if not in .env
    MODEL_TEMP: float = 0.3

    ENABLE_REMARK_REPHRASE: bool = True
    MAX_REPHRASE_RETRIES: int = 3

    FUZZY_SCORE_THRESHOLD: int = 85

    PUNJABI_REGULAR_FONT_PATH: str = "fonts/NotoSansGurmukhi-Regular.ttf"
    PUNJABI_BOLD_FONT_PATH: str = "fonts/NotoSansGurmukhi-Bold.ttf"
    PUNJABI_FONT_LOADED: bool = False

settings = Settings()