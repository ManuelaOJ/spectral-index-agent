"""
Custom exceptions for the Spectral Agent Platform.

Provides a hierarchy of specific exceptions for better error handling.
"""


class SpectralAgentError(Exception):
    """Base exception for all Spectral Agent errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class IngestionError(SpectralAgentError):
    """Error during data ingestion process."""

    pass


class AuthenticationError(SpectralAgentError):
    """Authentication failed with external service."""

    pass


class DownloadError(SpectralAgentError):
    """Error downloading data from external source."""

    def __init__(
        self, message: str, scene_id: str | None = None, details: dict | None = None
    ):
        self.scene_id = scene_id
        super().__init__(message, details)


class SpectralValidationError(SpectralAgentError):
    """Input validation failed."""

    def __init__(
        self, message: str, field: str | None = None, details: dict | None = None
    ):
        self.field = field
        super().__init__(message, details)


# Backward-compatible alias
ValidationError = SpectralValidationError


class RateLimitError(SpectralAgentError):
    """API rate limit exceeded."""

    def __init__(
        self, message: str, retry_after: int | None = None, details: dict | None = None
    ):
        self.retry_after = retry_after
        super().__init__(message, details)


class ProcessingError(SpectralAgentError):
    """Error during data processing."""

    pass


class ConfigurationError(SpectralAgentError):
    """Configuration is invalid or missing."""

    pass
