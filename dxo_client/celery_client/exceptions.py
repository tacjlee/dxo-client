"""
Celery Client Exceptions.
"""


class CeleryClientError(Exception):
    """Base exception for Celery client errors."""
    pass


class CeleryTimeoutError(CeleryClientError):
    """Task execution timed out."""

    def __init__(self, message: str, task_id: str = None, timeout: int = None):
        super().__init__(message)
        self.task_id = task_id
        self.timeout = timeout


class CeleryTaskError(CeleryClientError):
    """Task execution failed."""

    def __init__(self, message: str, task_id: str = None, original_error: str = None):
        super().__init__(message)
        self.task_id = task_id
        self.original_error = original_error


class CeleryConnectionError(CeleryClientError):
    """Failed to connect to Celery broker."""
    pass
