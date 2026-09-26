"""
Application settings with environment variable support.

Uses pydantic-settings for robust configuration management.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.

    All settings can be overridden via environment variables with the
    SPECTRAL_ prefix. For example, SPECTRAL_LOG_LEVEL=DEBUG.
    """

    model_config = SettingsConfigDict(
        env_prefix="SPECTRAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─────────────────────────────────────────────────────────────────
    # General Settings
    # ─────────────────────────────────────────────────────────────────
    app_name: str = "Spectral Index Agent"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    debug: bool = False

    # ─────────────────────────────────────────────────────────────────
    # Data Paths
    # ─────────────────────────────────────────────────────────────────
    data_dir: Path = Field(default=Path("data"))
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    cache_dir: Path = Field(default=Path("data/cache"))

    # ─────────────────────────────────────────────────────────────────
    # USGS M2M API (Landsat)
    # ─────────────────────────────────────────────────────────────────
    usgs_username: str = Field(default="", description="USGS EarthExplorer username")
    usgs_token: str = Field(default="", description="USGS M2M API token")
    usgs_m2m_endpoint: str = "https://m2m.cr.usgs.gov/api/api/json/stable/"
    landsat_collections: list[str] = Field(
        default=["landsat_ot_c2_l2", "landsat_etm_c2_l2", "landsat_tm_c2_l2"],
        description="Landsat collections to search. "
        "landsat_ot_c2_l2 = Landsat 8/9, "
        "landsat_etm_c2_l2 = Landsat 7, "
        "landsat_tm_c2_l2 = Landsat 4-5.",
    )

    # ─────────────────────────────────────────────────────────────────
    # Copernicus Data Space (Sentinel)
    # ─────────────────────────────────────────────────────────────────
    copernicus_client_id: str = Field(default="", description="Copernicus OAuth client ID")
    copernicus_client_secret: str = Field(default="", description="Copernicus OAuth client secret")
    copernicus_token_url: str = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )
    # Sentinel Hub APIs (for searching and processing)
    sentinel_hub_catalog_url: str = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0"
    sentinel_hub_process_url: str = "https://sh.dataspace.copernicus.eu/api/v1/process"
    # OData API (for direct product download)
    copernicus_catalog_url: str = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    copernicus_download_url: str = "https://zipper.dataspace.copernicus.eu/odata/v1"
    sentinel_collections: list[str] = Field(default=["sentinel-2-l2a"])

    # ─────────────────────────────────────────────────────────────────
    # LLM Configuration
    # ─────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    google_api_key: str = Field(
        default="",
        description="Google AI Studio API key (for Gemini)",
        validation_alias="GOOGLE_API_KEY",
    )
    default_llm_provider: Literal["openai", "anthropic", "google"] = "openai"
    default_llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, ge=1)

    # ─────────────────────────────────────────────────────────────────
    # Request Settings
    # ─────────────────────────────────────────────────────────────────
    request_timeout: int = Field(default=300, description="HTTP timeout in seconds")
    max_retries: int = Field(default=3, description="Max retry attempts for API calls")
    retry_delay: float = Field(default=1.0, description="Base delay between retries")
    download_chunk_size: int = Field(default=8192, description="Chunk size for downloads")
    max_concurrent_downloads: int = Field(default=4, description="Max parallel downloads")

    # ─────────────────────────────────────────────────────────────────
    # Observability
    # ─────────────────────────────────────────────────────────────────
    langsmith_api_key: str = Field(default="", description="LangSmith API key for tracing")
    langsmith_project: str = Field(default="spectral-agent", description="LangSmith project name")
    enable_tracing: bool = Field(default=False, description="Enable LangSmith tracing")

    @field_validator("data_dir", "raw_data_dir", "processed_data_dir", "cache_dir", mode="after")
    @classmethod
    def ensure_directories_exist(cls, v: Path) -> Path:
        """Create directories if they don't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def landsat_raw_dir(self) -> Path:
        """Directory for raw Landsat data."""
        path = self.raw_data_dir / "landsat"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def sentinel_raw_dir(self) -> Path:
        """Directory for raw Sentinel data."""
        path = self.raw_data_dir / "sentinel"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def has_usgs_credentials(self) -> bool:
        """Check if USGS credentials are configured."""
        return bool(self.usgs_username and self.usgs_token)

    def has_copernicus_credentials(self) -> bool:
        """Check if Copernicus credentials are configured."""
        return bool(self.copernicus_client_id and self.copernicus_client_secret)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings
    """
    return Settings()
