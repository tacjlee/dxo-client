"""
Tests for CeleryClient.

These tests mock the Celery app to test client behavior without
requiring a real Redis/Celery setup.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pydantic import BaseModel

from dxo_client.celery_client.config import CeleryClientConfig
from dxo_client.celery_client.exceptions import (
    CeleryClientError,
    CeleryTimeoutError,
    CeleryTaskError,
)


class MockAsyncResult:
    """Mock Celery AsyncResult."""

    def __init__(self, task_id: str, result=None, status="SUCCESS", failed=False):
        self.id = task_id
        self._result = result
        self._status = status
        self._failed = failed

    def get(self, timeout=None, propagate=True):
        if self._failed and propagate:
            raise Exception("Task failed")
        return self._result

    @property
    def status(self):
        return self._status

    def ready(self):
        return self._status in ("SUCCESS", "FAILURE")

    def successful(self):
        return self._status == "SUCCESS"

    def failed(self):
        return self._failed


class MockCeleryApp:
    """Mock Celery app."""

    def __init__(self):
        self.conf = MagicMock()
        self._tasks = {}
        self._last_task_call = None

    def send_task(self, name, args=None, kwargs=None, queue=None):
        self._last_task_call = {
            "name": name,
            "args": args or (),
            "kwargs": kwargs or {},
            "queue": queue,
        }
        # Return mock result
        result = {"status": "success", "data": "test"}
        return MockAsyncResult(task_id="test-task-id-123", result=result)

    @property
    def control(self):
        return MagicMock()


class ResponseModel(BaseModel):
    """Test response model."""
    status: str
    data: str


class TestCeleryClientConfig:
    """Tests for CeleryClientConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CeleryClientConfig()

        assert config.broker_url == "redis://localhost:6379/0"
        assert config.result_backend == "redis://localhost:6379/0"
        assert config.default_timeout == 300
        assert config.task_serializer == "json"
        assert config.result_serializer == "json"
        assert config.accept_content == ["json"]

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CeleryClientConfig(
            broker_url="redis://custom:6379/1",
            result_backend="redis://custom:6379/2",
            default_timeout=60,
        )

        assert config.broker_url == "redis://custom:6379/1"
        assert config.result_backend == "redis://custom:6379/2"
        assert config.default_timeout == 60

    def test_environment_variables(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("CELERY_BROKER_URL", "redis://env-broker:6379/0")
        monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://env-backend:6379/0")

        config = CeleryClientConfig()

        assert config.broker_url == "redis://env-broker:6379/0"
        assert config.result_backend == "redis://env-backend:6379/0"


class TestCeleryClient:
    """Tests for CeleryClient."""

    @pytest.fixture
    def mock_celery_app(self):
        """Create mock Celery app."""
        return MockCeleryApp()

    @pytest.fixture
    def client(self, mock_celery_app):
        """Create CeleryClient with mocked Celery app."""
        from dxo_client.celery_client.client import CeleryClient, reset_shared_celery_app

        # Reset singleton
        reset_shared_celery_app()

        # Create client and inject mock
        client = CeleryClient.__new__(CeleryClient)
        client._config = CeleryClientConfig()
        client._celery = mock_celery_app
        client.SERVICE_NAME = "test-service"
        client.TASK_PREFIX = "app.tasks.test"
        client.DEFAULT_QUEUE = "test_queue"
        client.DEFAULT_TIMEOUT = 300

        return client

    def test_get_full_task_name_with_prefix(self, client):
        """Test task name resolution with prefix."""
        full_name = client._get_full_task_name("my_task")
        assert full_name == "app.tasks.test.my_task"

    def test_get_full_task_name_already_qualified(self, client):
        """Test task name resolution for already qualified names."""
        full_name = client._get_full_task_name("other.module.my_task")
        assert full_name == "other.module.my_task"

    def test_call_sync_with_args(self, client, mock_celery_app):
        """Test synchronous call with positional arguments."""
        result = client.call_sync("my_task", "arg1", "arg2")

        assert result == {"status": "success", "data": "test"}
        assert mock_celery_app._last_task_call["name"] == "app.tasks.test.my_task"
        assert mock_celery_app._last_task_call["args"] == ("arg1", "arg2")
        assert mock_celery_app._last_task_call["queue"] == "test_queue"

    def test_call_sync_with_kwargs(self, client, mock_celery_app):
        """Test synchronous call with keyword arguments."""
        result = client.call_sync("my_task", name="test", value=123)

        assert mock_celery_app._last_task_call["kwargs"] == {"name": "test", "value": 123}

    def test_call_sync_with_mixed_args(self, client, mock_celery_app):
        """Test synchronous call with mixed arguments."""
        result = client.call_sync("my_task", "arg1", "arg2", name="test", value=123)

        assert mock_celery_app._last_task_call["args"] == ("arg1", "arg2")
        assert mock_celery_app._last_task_call["kwargs"] == {"name": "test", "value": 123}

    def test_call_sync_with_custom_queue(self, client, mock_celery_app):
        """Test synchronous call with custom queue."""
        result = client.call_sync("my_task", "data", queue="custom_queue")

        assert mock_celery_app._last_task_call["queue"] == "custom_queue"

    def test_call_sync_with_response_type(self, client):
        """Test synchronous call with response type deserialization."""
        result = client.call_sync("my_task", response_type=ResponseModel)

        assert isinstance(result, ResponseModel)
        assert result.status == "success"
        assert result.data == "test"

    def test_call_async(self, client, mock_celery_app):
        """Test asynchronous call."""
        result = client.call_async("my_task", "arg1", name="test")

        assert isinstance(result, MockAsyncResult)
        assert result.id == "test-task-id-123"
        assert mock_celery_app._last_task_call["name"] == "app.tasks.test.my_task"

    def test_get_result(self, client):
        """Test getting result from async task."""
        async_result = MockAsyncResult("task-123", result={"status": "done"})

        result = client.get_result(async_result)

        assert result == {"status": "done"}

    def test_get_result_with_response_type(self, client):
        """Test getting result with response type."""
        async_result = MockAsyncResult(
            "task-123",
            result={"status": "success", "data": "test"}
        )

        result = client.get_result(async_result, response_type=ResponseModel)

        assert isinstance(result, ResponseModel)
        assert result.status == "success"


class TestCeleryClientSubclass:
    """Tests for CeleryClient subclass pattern."""

    def test_subclass_with_custom_settings(self):
        """Test creating a subclass with custom settings."""
        from dxo_client.celery_client.client import CeleryClient, reset_shared_celery_app

        reset_shared_celery_app()

        class MyServiceClient(CeleryClient):
            SERVICE_NAME = "my-service"
            TASK_PREFIX = "app.tasks.my_service"
            DEFAULT_QUEUE = "my_queue"
            DEFAULT_TIMEOUT = 60

            def process(self, data, **options):
                return self.call_sync("process_task", data=data, **options)

        # Verify class attributes
        assert MyServiceClient.SERVICE_NAME == "my-service"
        assert MyServiceClient.TASK_PREFIX == "app.tasks.my_service"
        assert MyServiceClient.DEFAULT_QUEUE == "my_queue"
        assert MyServiceClient.DEFAULT_TIMEOUT == 60


class TestInitializeCeleryClient:
    """Tests for initialize_celery_client."""

    def test_initialize_with_existing_app(self):
        """Test injecting an existing Celery app."""
        from dxo_client.celery_client.client import (
            initialize_celery_client,
            get_shared_celery_app,
            reset_shared_celery_app,
        )

        # Reset first
        reset_shared_celery_app()

        # Create a mock Celery app
        mock_app = MockCeleryApp()

        # Initialize with our mock
        initialize_celery_client(mock_app)

        # Verify the shared app is our mock
        shared_app = get_shared_celery_app()
        assert shared_app is mock_app

        # Clean up
        reset_shared_celery_app()

    def test_clients_use_initialized_app(self):
        """Test that CeleryClient uses the initialized app."""
        from dxo_client.celery_client.client import (
            CeleryClient,
            initialize_celery_client,
            reset_shared_celery_app,
        )

        # Reset first
        reset_shared_celery_app()

        # Create and initialize with a mock app
        mock_app = MockCeleryApp()
        initialize_celery_client(mock_app)

        # Create a client - should use the initialized app
        client = CeleryClient()
        assert client._celery is mock_app

        # Clean up
        reset_shared_celery_app()


class TestCeleryExceptions:
    """Tests for Celery client exceptions."""

    def test_celery_timeout_error(self):
        """Test CeleryTimeoutError attributes."""
        error = CeleryTimeoutError(
            "Task timed out",
            task_id="task-123",
            timeout=60
        )

        assert str(error) == "Task timed out"
        assert error.task_id == "task-123"
        assert error.timeout == 60

    def test_celery_task_error(self):
        """Test CeleryTaskError attributes."""
        error = CeleryTaskError(
            "Task failed",
            task_id="task-123",
            original_error="Connection refused"
        )

        assert str(error) == "Task failed"
        assert error.task_id == "task-123"
        assert error.original_error == "Connection refused"

    def test_celery_client_error_hierarchy(self):
        """Test exception hierarchy."""
        assert issubclass(CeleryTimeoutError, CeleryClientError)
        assert issubclass(CeleryTaskError, CeleryClientError)
