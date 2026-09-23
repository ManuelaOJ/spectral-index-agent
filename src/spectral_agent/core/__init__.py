"""Core components and base classes."""

from .base_tool import BaseIngestionTool
from .exceptions import (
    SpectralAgentError,
    IngestionError,
    AuthenticationError,
    DownloadError,
    SpectralValidationError,
    ValidationError,  # backward-compat alias
    RateLimitError,
    ProcessingError,
    ConfigurationError,
)

__all__ = [
    "BaseIngestionTool",
    "SpectralAgentError",
    "IngestionError",
    "AuthenticationError",
    "DownloadError",
    "SpectralValidationError",
    "ValidationError",
    "RateLimitError",
    "ProcessingError",
    "ConfigurationError",
]
