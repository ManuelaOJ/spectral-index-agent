"""Core components and base classes."""

from .base_tool import BaseIngestionTool
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    DownloadError,
    IngestionError,
    ProcessingError,
    RateLimitError,
    SpectralAgentError,
    SpectralValidationError,
    ValidationError,  # backward-compat alias
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
