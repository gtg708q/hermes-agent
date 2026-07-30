class SSLConfigurationError(Exception):
    """Raised when SSL/TLS certificate bundle configuration fails."""
    pass


class EmptyStreamError(RuntimeError):
    """Raised when a provider closes a stream without yielding a response."""

    pass


class TurnWallClockExceeded(TimeoutError):
    """The configured whole-turn monotonic deadline was exhausted."""


class MoAPresetNotFoundError(ValueError):
    """Raised when a persisted MoA preset no longer exists in config."""
