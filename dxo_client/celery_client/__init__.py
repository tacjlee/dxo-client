"""
Celery Client SDK - Generic Celery task client.

Provides FeignClient-like interface for Celery task communication.

Requires: pip install dxo-client[celery]

Usage:
    # Direct call
    from dxo_client.celery_client import CeleryClient

    client = CeleryClient()
    result = client.call_sync("app.tasks.process", arg1, arg2, key="value")
    async_result = client.call_async("app.tasks.process", arg1, key="value")

    # Subclass for typed client
    from dxo_client.celery_client import CeleryClient

    class MyServiceClient(CeleryClient):
        TASK_PREFIX = "app.tasks.my_service"
        DEFAULT_QUEUE = "my_queue"

        def process(self, data, **options):
            return self.call_sync("process_task", data=data, **options)

    # Decorator-based (FeignClient-style)
    from dxo_client.celery_client import celery_client, task_method

    @celery_client(task_prefix="app.tasks.my_service", queue="my_queue")
    class MyServiceClient:

        @task_method(task="process_task", timeout=60)
        def process(self, data, **options):
            pass  # Implementation auto-generated
"""

try:
    import celery as _celery_check
except ImportError:
    raise ImportError(
        "Celery is required for dxo_client.celery_client module. "
        "Install with: pip install dxo-client[celery]"
    )

from .config import CeleryClientConfig
from .client import (
    CeleryClient,
    get_shared_celery_app,
    reset_shared_celery_app,
    initialize_celery_client,
)
from .decorators import celery_client, task_method
from .exceptions import (
    CeleryClientError,
    CeleryTimeoutError,
    CeleryTaskError,
    CeleryConnectionError,
)

__all__ = [
    # Client
    "CeleryClient",
    "CeleryClientConfig",
    "get_shared_celery_app",
    "reset_shared_celery_app",
    "initialize_celery_client",
    # Decorators
    "celery_client",
    "task_method",
    # Exceptions
    "CeleryClientError",
    "CeleryTimeoutError",
    "CeleryTaskError",
    "CeleryConnectionError",
]
