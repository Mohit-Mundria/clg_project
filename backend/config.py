"""
KisanAI - Application Configuration
Pydantic Settings for environment variable management
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
import os


class Settings(BaseSettings):
    # App
    app_name: str = "KisanAI"
    app_version: str = "1.0.0"
    app_host: str = Field(default="0.0.0.0", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")
    app_env: str = Field(default="development", env="APP_ENV")
    secret_key: str = Field(default="change-me-in-production", env="SECRET_KEY")

    # Groq & OpenAI LLM
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", env="GROQ_MODEL")

    # Weather
    weather_api_key: str = Field(default="e4d84587f15c4801a6c174344260405", env="WEATHER_API_KEY")

    # Defaults
    default_city: str = Field(default="New Delhi", env="DEFAULT_CITY")
    default_state: str = Field(default="Delhi", env="DEFAULT_STATE")
    default_country: str = Field(default="India", env="DEFAULT_COUNTRY")

    # CORS
    cors_origins: str = Field(
        default="http://localhost:8000,http://localhost:3000", env="CORS_ORIGINS"
    )

    # Paths
    base_dir: str = Field(default=os.path.dirname(os.path.abspath(__file__)))

    @property
    def model_dir(self) -> str:
        return os.path.join(self.base_dir, "models")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
