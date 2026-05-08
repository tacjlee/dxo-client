"""
Service Discovery

Consul-based service discovery with environment variable fallback.
"""

import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ServiceDiscovery:
    """
    Service discovery for workflow-knowledge.

    Hierarchy:
    1. Consul service catalog (if available)
    2. KNOWLEDGE_SERVICE_URL environment variable
    3. Default URL
    """

    def __init__(self):
        self._consul_enabled = os.getenv("CONSUL_ENABLED", "true").lower() not in ("false", "0", "no")
        self._consul_host = os.getenv("CONSUL_HOST", "localhost")
        self._consul_port = int(os.getenv("CONSUL_PORT", "8500"))
        self._consul = None
        self._cached_url: Optional[str] = None
        self._cache_timestamp: float = 0
        self._cache_ttl: int = 60  # seconds

        if self._consul_enabled:
            self._init_consul()

    def _init_consul(self):
        """Initialize Consul client."""
        try:
            import consul
            self._consul = consul.Consul(host=self._consul_host, port=self._consul_port)
            self._consul.agent.self()  # Test connection
            logger.info(f"Consul connected at {self._consul_host}:{self._consul_port}")
        except ImportError:
            logger.warning("python-consul not installed, Consul service discovery disabled")
            self._consul = None
        except Exception as e:
            logger.warning(f"Failed to connect to Consul: {e}")
            self._consul = None

    def _get_from_consul(self, service_name: str) -> Optional[str]:
        """Get service URL from Consul service catalog."""
        if not self._consul:
            return None

        try:
            # Query service from Consul catalog
            _, services = self._consul.catalog.service(service_name)
            if services:
                service = services[0]
                address = service.get("ServiceAddress") or service.get("Address")
                port = service.get("ServicePort")
                return f"http://{address}:{port}"

            # Fallback to KV store
            _, data = self._consul.kv.get(f"config/dev/services/{service_name}/url")
            if data and data.get("Value"):
                return data["Value"].decode("utf-8")
        except Exception as e:
            logger.warning(f"Consul lookup failed for {service_name}: {e}")

        return None

    def get_knowledge_service_url(self) -> str:
        """
        Get workflow-knowledge service URL.

        Fallback hierarchy:
        1. Consul service catalog
        2. KNOWLEDGE_SERVICE_URL environment variable
        3. Default URL
        """
        # Check cache
        if self._cached_url and (time.time() - self._cache_timestamp) < self._cache_ttl:
            return self._cached_url

        # Try Consul
        url = self._get_from_consul("workflow-knowledge")
        if url:
            self._cached_url = url
            self._cache_timestamp = time.time()
            logger.debug(f"Knowledge service URL from Consul: {url}")
            return url

        # Try environment variable
        env_url = os.getenv("KNOWLEDGE_SERVICE_URL")
        if env_url:
            self._cached_url = env_url
            self._cache_timestamp = time.time()
            logger.debug(f"Knowledge service URL from env: {env_url}")
            return env_url

        # Default
        default_url = "http://workflow-knowledge:8000"
        logger.debug(f"Using default knowledge service URL: {default_url}")
        return default_url

    def invalidate_cache(self):
        """Force cache refresh on next lookup."""
        self._cached_url = None
        self._cache_timestamp = 0
