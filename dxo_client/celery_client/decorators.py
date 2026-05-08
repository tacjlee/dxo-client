"""
Celery Client Decorators - FeignClient-style declarative client definition.

Usage:
    from dxo_client.celery_client import celery_client, task_method

    @celery_client(task_prefix="app.tasks.my_service", queue="my_queue")
    class MyServiceClient:

        @task_method(task="process_data", timeout=60)
        def process(self, data, option=None, **kwargs):
            pass  # Implementation auto-generated

        @task_method(task="process_data", async_mode=True)
        def process_async(self, data, **kwargs):
            pass
"""

from functools import wraps
from typing import Optional, Union

from kombu import Queue

from .client import CeleryClient, get_shared_celery_app
from .config import CeleryClientConfig


# Default queue arguments for PEV architecture (1 day TTL with DLQ)
DEFAULT_QUEUE_TTL = 86400000  # 24 hours in milliseconds


def _get_queue_from_app(celery_app, queue_name: str) -> Union[Queue, str]:
    """
    Get Queue object from celery_app.conf.task_queues if available.

    This ensures queue arguments (like x-message-ttl) are preserved
    when sending tasks to queues that have custom configurations.

    Args:
        celery_app: Celery application instance
        queue_name: Name of the queue to look up

    Returns:
        Queue object if found in task_queues, otherwise the queue name string
    """
    if celery_app and hasattr(celery_app.conf, 'task_queues'):
        task_queues = celery_app.conf.task_queues
        if task_queues:
            for q in task_queues:
                if hasattr(q, 'name') and q.name == queue_name:
                    return q
    return queue_name


def celery_client(
    celery_app=None,
    service_name: str = "celery-service",
    task_prefix: str = "",
    queue: str = "celery",
    timeout: int = 300,
    config: Optional[CeleryClientConfig] = None,
):
    """
    Class decorator to create a Celery client.

    Injects CeleryClient functionality into the decorated class.

    Args:
        celery_app: Existing Celery app instance to use (recommended)
        service_name: Name of the service (for logging)
        task_prefix: Prefix for task names
        queue: Default queue
        timeout: Default timeout in seconds
        config: Optional CeleryClientConfig

    Example:
        # With existing Celery app (recommended)
        from your_worker import celery_app

        @celery_client(
            celery_app,
            task_prefix="app.tasks.executor",
            queue="executor_queue"
        )
        class ExecutorClient:

            @task_method(task="execute_plan")
            def execute(self, plan, **options):
                pass

        # Without Celery app (creates new shared app)
        @celery_client(
            task_prefix="app.tasks.executor",
            queue="executor_queue"
        )
        class ExecutorClient:

            @task_method(task="execute_plan")
            def execute(self, plan, **options):
                pass
    """
    def decorator(cls):
        # Store configuration on class
        cls._service_name = service_name
        cls._task_prefix = task_prefix
        cls._default_queue = queue
        cls._default_timeout = timeout
        cls._config = config or CeleryClientConfig()
        cls._celery = celery_app  # Use provided app or None

        # Store original __init__ if exists
        original_init = cls.__init__ if hasattr(cls, '__init__') else None

        def new_init(self, *args, **kwargs):
            # Initialize Celery app (only if not already provided)
            if cls._celery is None:
                cls._celery = get_shared_celery_app(cls._config)
            self._celery = cls._celery
            self._service_name = cls._service_name
            self._task_prefix = cls._task_prefix
            self._default_queue = cls._default_queue
            self._default_timeout = cls._default_timeout

            # Call original __init__ if it existed and wasn't the default object.__init__
            if original_init is not None and original_init is not object.__init__:
                original_init(self, *args, **kwargs)

        cls.__init__ = new_init

        # Add helper methods
        def _get_full_task_name(self, task: str) -> str:
            if "." in task or not self._task_prefix:
                return task
            return f"{self._task_prefix}.{task}"

        cls._get_full_task_name = _get_full_task_name

        return cls

    return decorator


def task_method(
    task: str,
    timeout: int = None,
    queue: str = None,
    async_mode: bool = False,
    response_type=None,
):
    """
    Method decorator to define a Celery task call.

    The decorated method's arguments are forwarded to the Celery task.
    The method body is ignored - implementation is auto-generated.

    Args:
        task: Task name (without prefix)
        timeout: Timeout in seconds (default: class default)
        queue: Target queue (default: class default)
        async_mode: If True, returns AsyncResult instead of waiting
        response_type: Optional Pydantic model for response deserialization

    Example:
        @task_method(task="process", timeout=60)
        def process(self, data, option=None, **kwargs):
            pass

        @task_method(task="process", async_mode=True)
        def process_async(self, data, **kwargs):
            pass
    """
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            full_task = self._get_full_task_name(task)
            queue_name = queue or self._default_queue
            target_timeout = timeout or self._default_timeout

            # Look up Queue object from celery_app to preserve queue_arguments (TTL, DLQ, etc.)
            target_queue = _get_queue_from_app(self._celery, queue_name)

            if async_mode:
                # Async call - return immediately
                return self._celery.send_task(
                    full_task,
                    args=args,
                    kwargs=kwargs,
                    queue=target_queue,
                )
            else:
                # Sync call - wait for result
                task_result = self._celery.send_task(
                    full_task,
                    args=args,
                    kwargs=kwargs,
                    queue=target_queue,
                )

                data = task_result.get(timeout=target_timeout, propagate=True)

                if response_type is not None:
                    return response_type.model_validate(data)
                return data

        return wrapper

    return decorator
