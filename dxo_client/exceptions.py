"""
KnowledgeClient Exceptions

Custom exceptions for the knowledge client.
"""


class KnowledgeError(Exception):
    """Base exception for knowledge client errors."""
    pass


class KnowledgeConnectionError(KnowledgeError):
    """Raised when connection to knowledge service fails."""
    pass


class KnowledgeTimeoutError(KnowledgeError):
    """Raised when request to knowledge service times out."""
    pass


class KnowledgeAPIError(KnowledgeError):
    """Raised when knowledge service returns an error response."""

    def __init__(self, message: str, status_code: int = None, response_body: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class KnowledgeNotFoundError(KnowledgeError):
    """Raised when requested resource is not found."""
    pass


class KnowledgeValidationError(KnowledgeError):
    """Raised when request validation fails."""
    pass


class KnowledgeCircuitBreakerError(KnowledgeError):
    """Raised when circuit breaker is open."""
    pass


# Backwards compatibility aliases (deprecated)
KnowledgeBaseError = KnowledgeError
KnowledgeBaseConnectionError = KnowledgeConnectionError
KnowledgeBaseTimeoutError = KnowledgeTimeoutError
KnowledgeBaseAPIError = KnowledgeAPIError
KnowledgeBaseNotFoundError = KnowledgeNotFoundError
KnowledgeBaseValidationError = KnowledgeValidationError
KnowledgeBaseCircuitBreakerError = KnowledgeCircuitBreakerError
