"""
Tests for Celery client decorators.

Tests the @celery_client and @task_method decorators.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pydantic import BaseModel


class MockAsyncResult:
    """Mock Celery AsyncResult."""

    def __init__(self, task_id: str, result=None):
        self.id = task_id
        self._result = result

    def get(self, timeout=None, propagate=True):
        return self._result


class MockCeleryApp:
    """Mock Celery app."""

    def __init__(self):
        self.conf = MagicMock()
        self._last_task_call = None
        self._result = {"status": "success", "value": 42}

    def send_task(self, name, args=None, kwargs=None, queue=None):
        self._last_task_call = {
            "name": name,
            "args": args or (),
            "kwargs": kwargs or {},
            "queue": queue,
        }
        return MockAsyncResult(task_id="test-task-id", result=self._result)


class TestResponse(BaseModel):
    """Test response model."""
    status: str
    value: int


class TestCeleryClientDecorator:
    """Tests for @celery_client decorator."""

    @pytest.fixture
    def mock_celery_app(self):
        """Create mock Celery app."""
        return MockCeleryApp()

    def test_decorator_injects_attributes(self, mock_celery_app):
        """Test that decorator injects required attributes."""
        with patch.dict('sys.modules', {'celery': MagicMock(), 'celery.result': MagicMock()}):
            with patch('dxo_client.celery_client.decorators.get_shared_celery_app', return_value=mock_celery_app):
                from dxo_client.celery_client.decorators import celery_client

                @celery_client(
                    service_name="test-service",
                    task_prefix="app.tasks.test",
                    queue="test_queue",
                    timeout=120
                )
                class TestClient:
                    pass

                # Create instance
                client = TestClient()

                # Verify attributes
                assert client._service_name == "test-service"
                assert client._task_prefix == "app.tasks.test"
                assert client._default_queue == "test_queue"
                assert client._default_timeout == 120
                assert client._celery is mock_celery_app

    def test_decorator_preserves_original_init(self, mock_celery_app):
        """Test that decorator preserves original __init__."""
        with patch.dict('sys.modules', {'celery': MagicMock(), 'celery.result': MagicMock()}):
            with patch('dxo_client.celery_client.decorators.get_shared_celery_app', return_value=mock_celery_app):
                from dxo_client.celery_client.decorators import celery_client

                @celery_client(task_prefix="app.tasks.test")
                class TestClient:
                    def __init__(self, custom_value):
                        self.custom_value = custom_value

                client = TestClient("my-value")

                assert client.custom_value == "my-value"
                assert client._task_prefix == "app.tasks.test"

    def test_get_full_task_name_method(self, mock_celery_app):
        """Test that _get_full_task_name is injected."""
        with patch.dict('sys.modules', {'celery': MagicMock(), 'celery.result': MagicMock()}):
            with patch('dxo_client.celery_client.decorators.get_shared_celery_app', return_value=mock_celery_app):
                from dxo_client.celery_client.decorators import celery_client

                @celery_client(task_prefix="app.tasks.test")
                class TestClient:
                    pass

                client = TestClient()

                # Test task name resolution
                assert client._get_full_task_name("my_task") == "app.tasks.test.my_task"
                assert client._get_full_task_name("other.module.task") == "other.module.task"


class TestTaskMethodDecorator:
    """Tests for @task_method decorator."""

    @pytest.fixture
    def mock_celery_app(self):
        """Create mock Celery app."""
        return MockCeleryApp()

    def test_sync_task_method(self, mock_celery_app):
        """Test synchronous task method."""
        with patch.dict('sys.modules', {'celery': MagicMock(), 'celery.result': MagicMock()}):
            with patch('dxo_client.celery_client.decorators.get_shared_celery_app', return_value=mock_celery_app):
                from dxo_client.celery_client.decorators import celery_client, task_method

                @celery_client(task_prefix="app.tasks.test", queue="test_queue")
                class TestClient:

                    @task_method(task="process_data", timeout=60)
                    def process(self, data, option=None):
                        pass

                client = TestClient()
                result = client.process("my-data", option="fast")

                # Verify task was called correctly
                assert mock_celery_app._last_task_call["name"] == "app.tasks.test.process_data"
                assert mock_celery_app._last_task_call["args"] == ("my-data",)
                assert mock_celery_app._last_task_call["kwargs"] == {"option": "fast"}
                assert mock_celery_app._last_task_call["queue"] == "test_queue"

                # Verify result
                assert result == {"status": "success", "value": 42}

    def test_async_task_method(self, mock_celery_app):
        """Test asynchronous task method."""
        with patch.dict('sys.modules', {'celery': MagicMock(), 'celery.result': MagicMock()}):
            with patch('dxo_client.celery_client.decorators.get_shared_celery_app', return_value=mock_celery_app):
                from dxo_client.celery_client.decorators import celery_client, task_method

                @celery_client(task_prefix="app.tasks.test", queue="test_queue")
                class TestClient:

                    @task_method(task="process_data", async_mode=True)
                    def process_async(self, data):
                        pass

                client = TestClient()
                result = client.process_async("my-data")

                # Verify returns AsyncResult
                assert isinstance(result, MockAsyncResult)
                assert result.id == "test-task-id"

    def test_task_method_with_custom_queue(self, mock_celery_app):
        """Test task method with custom queue."""
        with patch.dict('sys.modules', {'celery': MagicMock(), 'celery.result': MagicMock()}):
            with patch('dxo_client.celery_client.decorators.get_shared_celery_app', return_value=mock_celery_app):
                from dxo_client.celery_client.decorators import celery_client, task_method

                @celery_client(task_prefix="app.tasks.test", queue="default_queue")
                class TestClient:

                    @task_method(task="process_data", queue="priority_queue")
                    def process(self, data):
                        pass

                client = TestClient()
                client.process("data")

                # Verify custom queue was used
                assert mock_celery_app._last_task_call["queue"] == "priority_queue"

    def test_task_method_with_response_type(self, mock_celery_app):
        """Test task method with response type deserialization."""
        with patch.dict('sys.modules', {'celery': MagicMock(), 'celery.result': MagicMock()}):
            with patch('dxo_client.celery_client.decorators.get_shared_celery_app', return_value=mock_celery_app):
                from dxo_client.celery_client.decorators import celery_client, task_method

                @celery_client(task_prefix="app.tasks.test")
                class TestClient:

                    @task_method(task="process_data", response_type=TestResponse)
                    def process(self, data):
                        pass

                client = TestClient()
                result = client.process("data")

                # Verify response is deserialized
                assert isinstance(result, TestResponse)
                assert result.status == "success"
                assert result.value == 42

    def test_task_method_with_only_kwargs(self, mock_celery_app):
        """Test task method with only keyword arguments."""
        with patch.dict('sys.modules', {'celery': MagicMock(), 'celery.result': MagicMock()}):
            with patch('dxo_client.celery_client.decorators.get_shared_celery_app', return_value=mock_celery_app):
                from dxo_client.celery_client.decorators import celery_client, task_method

                @celery_client(task_prefix="app.tasks.test")
                class TestClient:

                    @task_method(task="process_data")
                    def process(self, name=None, value=None, **extra):
                        pass

                client = TestClient()
                client.process(name="test", value=123, debug=True)

                # Verify kwargs
                assert mock_celery_app._last_task_call["args"] == ()
                assert mock_celery_app._last_task_call["kwargs"] == {
                    "name": "test",
                    "value": 123,
                    "debug": True
                }

    def test_task_method_preserves_function_name(self, mock_celery_app):
        """Test that decorator preserves function metadata."""
        with patch.dict('sys.modules', {'celery': MagicMock(), 'celery.result': MagicMock()}):
            with patch('dxo_client.celery_client.decorators.get_shared_celery_app', return_value=mock_celery_app):
                from dxo_client.celery_client.decorators import celery_client, task_method

                @celery_client(task_prefix="app.tasks.test")
                class TestClient:

                    @task_method(task="process_data")
                    def my_custom_method(self, data):
                        """My docstring."""
                        pass

                client = TestClient()

                # Verify function metadata is preserved
                assert client.my_custom_method.__name__ == "my_custom_method"


class TestDecoratorWithCeleryApp:
    """Tests for @celery_client decorator with celery_app parameter."""

    @pytest.fixture
    def mock_celery_app(self):
        """Create mock Celery app."""
        return MockCeleryApp()

    def test_decorator_with_celery_app_parameter(self, mock_celery_app):
        """Test that decorator uses provided celery_app."""
        from dxo_client.celery_client.decorators import celery_client, task_method

        @celery_client(
            mock_celery_app,
            task_prefix="app.tasks.test"
        )
        class TestClient:

            @task_method(task="process_data")
            def process(self, data):
                pass

        client = TestClient()

        # Verify the client uses the provided app
        assert client._celery is mock_celery_app

        # Verify task calls work
        client.process("test-data")
        assert mock_celery_app._last_task_call["name"] == "app.tasks.test.process_data"

    def test_decorator_celery_app_shared_across_instances(self, mock_celery_app):
        """Test that celery_app is shared across all instances."""
        from dxo_client.celery_client.decorators import celery_client, task_method

        @celery_client(
            mock_celery_app,
            task_prefix="app.tasks.test"
        )
        class TestClient:

            @task_method(task="process_data")
            def process(self, data):
                pass

        client1 = TestClient()
        client2 = TestClient()

        # Both instances should share the same app
        assert client1._celery is mock_celery_app
        assert client2._celery is mock_celery_app
        assert client1._celery is client2._celery


class TestDecoratorIntegration:
    """Integration tests for decorators working together."""

    @pytest.fixture
    def mock_celery_app(self):
        """Create mock Celery app."""
        return MockCeleryApp()

    def test_full_client_example(self, mock_celery_app):
        """Test a complete client example with multiple methods."""
        with patch.dict('sys.modules', {'celery': MagicMock(), 'celery.result': MagicMock()}):
            with patch('dxo_client.celery_client.decorators.get_shared_celery_app', return_value=mock_celery_app):
                from dxo_client.celery_client.decorators import celery_client, task_method

                @celery_client(
                    service_name="executor",
                    task_prefix="app.tasks.executor",
                    queue="executor_queue",
                    timeout=300
                )
                class ExecutorClient:
                    """Client for executor service."""

                    @task_method(task="execute_plan", timeout=600)
                    def execute(self, plan, options=None):
                        """Execute a plan synchronously."""
                        pass

                    @task_method(task="execute_plan", async_mode=True)
                    def execute_async(self, plan, options=None):
                        """Execute a plan asynchronously."""
                        pass

                    @task_method(task="get_status")
                    def get_status(self, task_id):
                        """Get execution status."""
                        pass

                # Test the client
                client = ExecutorClient()

                # Test sync method
                result = client.execute({"steps": [1, 2, 3]}, options={"fast": True})
                assert mock_celery_app._last_task_call["name"] == "app.tasks.executor.execute_plan"
                # plan is passed as positional arg, options as kwarg
                assert mock_celery_app._last_task_call["args"] == ({"steps": [1, 2, 3]},)
                assert mock_celery_app._last_task_call["kwargs"] == {"options": {"fast": True}}

                # Test async method
                async_result = client.execute_async({"steps": [4, 5]})
                assert isinstance(async_result, MockAsyncResult)

                # Test status method
                client.get_status("task-123")
                assert mock_celery_app._last_task_call["name"] == "app.tasks.executor.get_status"
                # task_id is passed as positional arg
                assert mock_celery_app._last_task_call["args"] == ("task-123",)
