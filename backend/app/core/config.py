from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    app_title: Optional[str] = "Clinical Guideline RAG Service (CDC/WHO)"
    app_version: Optional[str] = "v0.1.0"

    openai_api_key: str = os.environ.get("OPENAI_API_KEY")

settings = Settings()
