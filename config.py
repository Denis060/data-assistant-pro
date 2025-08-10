"""Configuration management for Data Assistant Pro."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AppConfig:
    """Application configuration settings."""

    # File handling
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))  # 100MB
    SUPPORTED_FORMATS: list = None

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "data_assistant.log")

    # Performance
    ENABLE_PROFILING: bool = os.getenv("ENABLE_PROFILING", "False").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", 3600))  # 1 hour

    # Data processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 10000))
    MAX_ROWS_WARNING: int = int(os.getenv("MAX_ROWS_WARNING", 50000))

    # ML settings
    DEFAULT_TEST_SIZE: float = float(os.getenv("DEFAULT_TEST_SIZE", 0.2))
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", 42))

    # UI settings
    THEME: str = os.getenv("THEME", "dark")
    SHOW_DEVELOPER_INFO: bool = (
        os.getenv("SHOW_DEVELOPER_INFO", "True").lower() == "true"
    )

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.SUPPORTED_FORMATS is None:
            self.SUPPORTED_FORMATS = ["csv", "xlsx", "json", "parquet"]

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        return cls()

    def get_max_file_size_mb(self) -> float:
        """Get maximum file size in MB."""
        return self.MAX_FILE_SIZE / (1024 * 1024)

    def is_file_size_valid(self, file_size: int) -> bool:
        """Check if file size is within limits."""
        return file_size <= self.MAX_FILE_SIZE


# Global configuration instance
config = AppConfig.from_env()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> None:
    """Update configuration values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")


# Environment-specific configurations
class DevelopmentConfig(AppConfig):
    """Development environment configuration."""

    LOG_LEVEL: str = "DEBUG"
    ENABLE_PROFILING: bool = True


class ProductionConfig(AppConfig):
    """Production environment configuration."""

    LOG_LEVEL: str = "WARNING"
    ENABLE_PROFILING: bool = False
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB for production


def get_config_for_env(env: Optional[str] = None) -> AppConfig:
    """Get configuration for specific environment."""
    env = env or os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionConfig()
    elif env == "development":
        return DevelopmentConfig()
    else:
        return AppConfig()
