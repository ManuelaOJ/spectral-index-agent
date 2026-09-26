"""Agent definitions for the Spectral Agent Platform."""

from .spectral_agent import (
    SpectralAgentConfig,
    create_spectral_agent,
    run_agent_query,
)

__all__ = [
    "create_spectral_agent",
    "run_agent_query",
    "SpectralAgentConfig",
]
