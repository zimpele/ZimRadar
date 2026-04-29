from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://zimradar:password@localhost:5432/zimradar"
    redis_url: str = "redis://localhost:6379"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma2:9b"
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"

    sentinelsat_user: str = ""
    sentinelsat_pass: str = ""
    noaa_api_key: str = ""

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "us-east-1"
    s3_bucket_tiles: str = "zimradar-tiles"
    s3_bucket_pdfs: str = "zimradar-pdfs"

    langsmith_api_key: str = ""
    langsmith_project: str = "zimradar"
    api_key: str = ""

    risk_score_weights: str = "0.6,0.2,0.2"

    @property
    def risk_weights(self) -> tuple[float, float, float]:
        w = [float(x) for x in self.risk_score_weights.split(",")]
        return w[0], w[1], w[2]


@lru_cache
def get_settings() -> Settings:
    return Settings()
