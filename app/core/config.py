from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    strava_client_id: Optional[str] = None
    strava_client_secret: Optional[str] = None
    strava_redirect_uri: str = "http://127.0.0.1:8000/callback"
    openai_api_key: Optional[str] = None
    withings_client_id: Optional[str] = None
    withings_client_secret: Optional[str] = None
    withings_redirect_uri: str = "http://127.0.0.1:8000/callback"
    
    # App Settings
    debug: bool = False
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
