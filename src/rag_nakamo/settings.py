from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, Field
from functools import lru_cache
from typing import Optional, Literal

class Settings(BaseSettings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"Loading API from .env file: {bool(self.openai_api_key)}")
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # LLM
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    model_provider: Literal["openai", "huggingface", "ollama"] = "openai"
    orchestrator_model: str = "gpt-4o-mini" # now default
    response_model: str = "gpt-4o-mini"
    validation_model: str = "gpt-4o-mini"

    # DB
    chroma_persist_dir: str = ".chroma"
    max_context_tokens: int = 10000 # limit ?
    retrieval_top_k: int = 5
    # if we use ensemble retrieval, we will rerank the top k results
    # enable_rerank: bool = True
    # rerank_top_k: int = 3

    # WEB
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    google_cx: Optional[str] = Field(default=None, env="GOOGLE_CX")

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache
def get_settings() -> Settings:
    return Settings()
