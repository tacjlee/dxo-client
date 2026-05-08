"""
CeleryClient - Generic Celery task client.

Provides FeignClient-like interface for Celery task communication.

Usage:
    # Direct call
    from dxo_client.celery_client import CeleryClient

    client = CeleryClient()
    response = client.call_sync("app.tasks.process", arg1, arg2, key="value")

    # Subclass for typed client
    class MyServiceClient(CeleryClient):
        TASK_PREFIX = "app.tasks.my_service"
        DEFAULT_QUEUE = "my_queue"

        def process(self, data, **options):
            return self.call_sync("process_task", data=data, **options)
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar

from celery import Celery
from celery.result import AsyncResult
from celery.exceptions import TimeoutError as CeleryTimeoutException

from .config import CeleryClientConfig
from .exceptions import CeleryClientError, CeleryTimeoutError, CeleryTaskError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Shared Celery app singleton
_celery_app: Optional[Celery] = None


def get_shared_celery_app(config: CeleryClientConfig = None) -> Celery:
    """
    Get or create shared Celery app instance.

    All CeleryClient instances share the same Celery app to avoid
    multiple connections to Redis.

    Args:
        config: Optional configuration override

    Returns:
        Shared Celery app instance
    """
    global _celery_app

    if _celery_app is None:
        config = config or CeleryClientConfig()
        _celery_app = Celery(
            "celery_client",
            broker=config.broker_url,
            backend=config.result_backend,
        )
        _celery_app.conf.update(
            task_serializer=config.task_serializer,
            result_serializer=config.result_serializer,
            accept_content=config.accept_content,
            task_track_started=config.task_track_started,
            result_expires=config.result_expires,
        )
        logger.info(f"Initialized shared Celery app with broker: {config.broker_url}")

    return _celery_app


def reset_shared_celery_app():
    """Reset shared Celery app (useful for testing)."""
    global _celery_app
    _celery_app = None


def initialize_celery_client(celery_app: Celery) -> None:
    """
    Initialize the shared Celery client with an existing Celery app.

    Use this to inject your microservice's Celery app instead of
    creating a new one. Call this at application startup before
    creating any CeleryClient instances.

    Args:
        celery_app: Existing Celery app instance

    Example:
        # In your microservice startup
        from your_worker import celery_app
        from dxo_client.celery_client import initialize_celery_client

        initialize_celery_client(celery_app)

        # Now all clients use your existing app
        client = MyServiceClient()
    """
    global _celery_app
    _celery_app = celery_app
    logger.info("Initialized shared Celery client with existing app")


class CeleryClient:
    """
    Generic Celery task client.

    Provides a simple interface for calling Celery tasks with full
    *args/**kwargs support.

    Class Attributes (override in subclass):
        SERVICE_NAME: Name of the service (for logging)
        TASK_PREFIX: Prefix for task names (e.g., "app.tasks.my_service")
        DEFAULT_QUEUE: Default queue to send tasks to
        DEFAULT_TIMEOUT: Default timeout in seconds

    Example:
        # Direct usage
        client = CeleryClient()
        result = client.call_sync("app.tasks.process", data, key="value")

        # Subclass
        class ExecutorClient(CeleryClient):
            TASK_PREFIX = "app.tasks.executor"
            DEFAULT_QUEUE = "executor_queue"

            def execute(self, plan, **options):
                return self.call_sync("execute_plan", plan=plan, **options)
    """

    SERVICE_NAME: str = "celery-service"
    TASK_PREFIX: str = ""
    DEFAULT_QUEUE: str = "celery"
    DEFAULT_TIMEOUT: int = 300

    def __init__(self, config: Optional[CeleryClientConfig] = None):
        """
        Initialize CeleryClient.

        Args:
            config: Optional configuration override
        """
        self._config = config or CeleryClientConfig()
        self._celery = get_shared_celery_app(self._config)

    def _get_full_task_name(self, task: str) -> str:
        """
        Get fully qualified task name.

        If task already contains a dot, it's assumed to be fully qualified.
        Otherwise, TASK_PREFIX is prepended.

        Args:
            task: Task name (short or full)

        Returns:
            Fully qualified task name
        """
        if "." in task or not self.TASK_PREFIX:
            return task
        return f"{self.TASK_PREFIX}.{task}"

    def call_sync(
        self,
        task: str,
        *args,
        response_type: Type[T] = None,
        timeout: int = None,
        queue: str = None,
        **kwargs
    ) -> T | Dict[str, Any] | Any:
        """
        Call a Celery task synchronously (blocks until result).

        Args:
            task: Task name (with or without prefix)
            *args: Positional arguments for the task
            response_type: Optional Pydantic model for response deserialization
            timeout: Timeout in seconds (default: DEFAULT_TIMEOUT)
            queue: Target queue (default: DEFAULT_QUEUE)
            **kwargs: Keyword arguments for the task

        Returns:
            Task result (deserialized to response_type if provided)

        Raises:
            CeleryTimeoutError: If task times out
            CeleryTaskError: If task execution fails

        Example:
            # Simple call
            result = client.call_sync("process", data, debug=True)

            # With timeout and queue
            result = client.call_sync("process", data, timeout=60, queue="fast")

            # With response type
            result = client.call_sync("process", data, response_type=MyResponse)
        """
        full_task = self._get_full_task_name(task)
        queue = queue or self.DEFAULT_QUEUE
        timeout = timeout or self.DEFAULT_TIMEOUT

        logger.info(
            f"[{self.SERVICE_NAME}] Calling {full_task} "
            f"(sync, queue={queue}, timeout={timeout}s)"
        )

        task_result = None
        try:
            # Send task
            task_result = self._celery.send_task(
                full_task,
                args=args,
                kwargs=kwargs,
                queue=queue,
            )

            logger.debug(f"[{self.SERVICE_NAME}] Task {task_result.id} submitted")

            # Wait for result
            data = task_result.get(timeout=timeout, propagate=True)

            logger.info(f"[{self.SERVICE_NAME}] Task {task_result.id} completed")

            # Deserialize if response_type provided
            if response_type is not None:
                return response_type.model_validate(data)
            return data

        except CeleryTimeoutException:
            task_id = task_result.id if task_result else None
            logger.error(
                f"[{self.SERVICE_NAME}] Task {task_id} timed out after {timeout}s"
            )
            raise CeleryTimeoutError(
                f"Task {full_task} timed out after {timeout}s",
                task_id=task_id,
                timeout=timeout
            )
        except CeleryTimeoutError:
            raise
        except Exception as e:
            task_id = task_result.id if task_result else None
            logger.error(f"[{self.SERVICE_NAME}] Task {task_id} failed: {e}")
            raise CeleryTaskError(
                f"Task {full_task} failed: {e}",
                task_id=task_id,
                original_error=str(e)
            )

    def call_async(
        self,
        task: str,
        *args,
        queue: str = None,
        **kwargs
    ) -> AsyncResult:
        """
        Call a Celery task asynchronously (returns immediately).

        Args:
            task: Task name (with or without prefix)
            *args: Positional arguments for the task
            queue: Target queue (default: DEFAULT_QUEUE)
            **kwargs: Keyword arguments for the task

        Returns:
            AsyncResult for tracking task status

        Example:
            # Start async task
            async_result = client.call_async("process", data)

            # Do other work...

            # Get result later
            result = client.get_result(async_result)
        """
        full_task = self._get_full_task_name(task)
        queue = queue or self.DEFAULT_QUEUE

        logger.info(f"[{self.SERVICE_NAME}] Calling {full_task} (async, queue={queue})")

        task_result = self._celery.send_task(
            full_task,
            args=args,
            kwargs=kwargs,
            queue=queue,
        )

        logger.debug(f"[{self.SERVICE_NAME}] Task {task_result.id} submitted (async)")

        return task_result

    def get_result(
        self,
        async_result: AsyncResult,
        response_type: Type[T] = None,
        timeout: int = None,
    ) -> T | Dict[str, Any] | Any:
        """
        Get result from an async task.

        Args:
            async_result: AsyncResult from call_async
            response_type: Optional Pydantic model for response deserialization
            timeout: Timeout in seconds (default: DEFAULT_TIMEOUT)

        Returns:
            Task result (deserialized to response_type if provided)

        Raises:
            CeleryTimeoutError: If task times out
            CeleryTaskError: If task execution fails
        """
        timeout = timeout or self.DEFAULT_TIMEOUT

        try:
            data = async_result.get(timeout=timeout, propagate=True)

            if response_type is not None:
                return response_type.model_validate(data)
            return data

        except CeleryTimeoutException:
            raise CeleryTimeoutError(
                f"Task {async_result.id} timed out after {timeout}s",
                task_id=async_result.id,
                timeout=timeout
            )
        except CeleryTimeoutError:
            raise
        except Exception as e:
            raise CeleryTaskError(
                f"Task {async_result.id} failed: {e}",
                task_id=async_result.id,
                original_error=str(e)
            )

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a task by ID.

        Args:
            task_id: Celery task ID

        Returns:
            Dict with task status information
        """
        result = AsyncResult(task_id, app=self._celery)
        return {
            "task_id": task_id,
            "status": result.status,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else None,
            "failed": result.failed() if result.ready() else None,
            "result": result.result if result.ready() and result.successful() else None,
            "error": str(result.result) if result.ready() and result.failed() else None,
        }

    def revoke_task(self, task_id: str, terminate: bool = False) -> None:
        """
        Revoke (cancel) a task.

        Args:
            task_id: Celery task ID
            terminate: If True, terminate the task even if already running
        """
        self._celery.control.revoke(task_id, terminate=terminate)
        logger.info(f"[{self.SERVICE_NAME}] Revoked task {task_id}")
