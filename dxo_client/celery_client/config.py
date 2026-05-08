"""
Celery Client Configuration.

Configuration priority:
1. Explicit values passed to constructor
2. Consul (if available)
3. Environment variables
4. Default values
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _get_from_consul(key: str) -> Optional[str]:
    """Get config value from Consul if available."""
    try:
        from dxo_client import consul_client
        if consul_client is not None and consul_client.is_available():
            value = consul_client.get(key, None)
            if value is not None:
                return value
    except (ImportError, Exception):
        pass
    return None


def _get_config(key: str, default: str) -> str:
    """
    Get config value with fallback hierarchy:
    1. Consul (if available)
    2. Environment variable
    3. Default value
    """
    # Try Consul first
    value = _get_from_consul(key)
    if value is not None:
        return value

    # Fallback to environment variable
    return os.getenv(key, default)


@dataclass
class CeleryClientConfig:
    """
    Configuration for CeleryClient.

    Configuration priority:
    1. Explicit values passed to constructor
    2. Consul (if available)
    3. Environment variables
    4. Default values

    Environment Variables / Consul Keys:
        CELERY_BROKER_URL: Message broker URL (default: redis://localhost:6379/0)
        CELERY_RESULT_BACKEND: Result backend URL (default: redis://localhost:6379/0)

    Example:
        # Use defaults from Consul/environment
        config = CeleryClientConfig()

        # Override specific settings
        config = CeleryClientConfig(
            broker_url="amqp://guest:guest@rabbitmq:5672/",
            default_timeout=60
        )
    """

    broker_url: Optional[str] = None
    result_backend: Optional[str] = None
    default_timeout: int = 300
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: List[str] = field(default_factory=lambda: ["json"])
    task_track_started: bool = True
    result_expires: int = 3600

    def __post_init__(self):
        """Resolve config from Consul/environment after initialization."""
        if self.broker_url is None:
            self.broker_url = _get_config(
                "CELERY_BROKER_URL", "redis://localhost:6379/0"
            )
        if self.result_backend is None:
            self.result_backend = _get_config(
                "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
            )
