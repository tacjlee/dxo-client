"""
Consul Configuration Client

Provides centralized configuration management via Consul KV store with fallback
to environment variables. Generic client that can be used by any microservice.

Usage:
    from dxo_client import consul_client

    # Get from base_path (2 args)
    value = consul_client.get("REDIS_HOST", "localhost")
    port = consul_client.get_int("REDIS_PORT", 6379)

    # Get from custom path (3 args)
    value = consul_client.get("config/dev/system/configs", "MY_KEY", "default")
    port = consul_client.get_int("config/dev/tenant/configs", "MAX_USERS", 100)

Installation:
    pip install dxo-client[consul]
"""

import os
import json
import time
import logging
import threading
from typing import Optional, Any, Dict, Union, overload

logger = logging.getLogger(__name__)


class ConsulClient:
    """
    Singleton Consul client with TTL-based caching and graceful fallback.

    Configuration hierarchy:
    1. Consul KV store (if enabled and available)
    2. Environment variables
    3. Default values

    Environment variables:
    - CONSUL_ENABLED: Enable/disable Consul (default: true)
    - CONSUL_HOST: Consul server host (default: localhost)
    - CONSUL_PORT: Consul server port (default: 8500)
    - CONSUL_BASE_PATH: Base path for settings (default: config/dev/settings)
    - CONSUL_CACHE_TTL: Cache TTL in seconds (default: 60)
    """

    _instance: Optional["ConsulClient"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ConsulClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._consul = None
        self._available = False
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._cache_lock = threading.RLock()

        # Load Consul connection settings from environment
        self._enabled = os.getenv("CONSUL_ENABLED", "true").lower() not in ("false", "0", "no", "off")
        self._host = os.getenv("CONSUL_HOST", "localhost")
        self._port = int(os.getenv("CONSUL_PORT", "8500"))
        self._base_path = os.getenv("CONSUL_BASE_PATH", "config/dev/settings")
        self._cache_ttl = int(os.getenv("CONSUL_CACHE_TTL", "60"))

        if self._enabled:
            self._connect()

    def _connect(self) -> bool:
        """Attempt to connect to Consul server."""
        try:
            import consul
            self._consul = consul.Consul(host=self._host, port=self._port)
            # Test connection by getting agent info
            self._consul.agent.self()
            self._available = True
            logger.info(f"Connected to Consul at {self._host}:{self._port}")
            return True
        except ImportError:
            logger.warning("python-consul package not installed. Install with: pip install dxo-client[consul]")
            self._available = False
            return False
        except Exception as e:
            logger.warning(f"Failed to connect to Consul at {self._host}:{self._port}: {e}")
            self._available = False
            return False

    @property
    def base_path(self) -> str:
        """Get the base path for Consul KV lookups."""
        return self._base_path

    @property
    def host(self) -> str:
        """Get the Consul host."""
        return self._host

    @property
    def port(self) -> int:
        """Get the Consul port."""
        return self._port

    def is_available(self) -> bool:
        """Check if Consul is available and connected."""
        return self._enabled and self._available

    def is_enabled(self) -> bool:
        """Check if Consul integration is enabled."""
        return self._enabled

    def _get_cached(self, cache_key: str) -> Optional[tuple[str, bool]]:
        """
        Get a value from cache if not expired.
        Returns (value, hit) tuple where hit indicates if cache was valid.
        """
        with self._cache_lock:
            if cache_key in self._cache:
                value, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    return (value, True)
        return (None, False)

    def _set_cached(self, cache_key: str, value: str) -> None:
        """Store a value in cache with current timestamp."""
        with self._cache_lock:
            self._cache[cache_key] = (value, time.time())

    def _get_from_consul(self, path: str, key: str) -> Optional[str]:
        """Get a value directly from Consul KV store."""
        if not self.is_available():
            return None

        try:
            full_key = f"{path}/{key}"
            _, data = self._consul.kv.get(full_key)
            if data is not None and data.get("Value") is not None:
                return data["Value"].decode("utf-8")
        except Exception as e:
            logger.debug(f"Failed to get key '{key}' from Consul path '{path}': {e}")
            if "Connection" in str(e) or "refused" in str(e).lower():
                self._available = False

        return None

    def _fetch(self, path: str, key: str, default: str) -> str:
        """Internal fetch with explicit path, key, default."""
        cache_key = f"{path}:{key}"

        # Check cache first
        cached_value, cache_hit = self._get_cached(cache_key)
        if cache_hit and cached_value is not None:
            return cached_value

        # Try Consul if available
        if self.is_available():
            consul_value = self._get_from_consul(path, key)
            if consul_value is not None:
                self._set_cached(cache_key, consul_value)
                return consul_value

        # Fall back to environment variable
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value

        return default

    # =========================================================================
    # CORE METHODS - Overloaded: 2 args (key, default) or 3 args (path, key, default)
    # =========================================================================

    @overload
    def get(self, key: str, default: str = "") -> str: ...
    @overload
    def get(self, path: str, key: str, default: str = "") -> str: ...

    def get(self, *args) -> str:
        """
        Get a configuration value with fallback hierarchy:
        1. Local cache (if not expired)
        2. Consul KV store (if available)
        3. Environment variable
        4. Default value

        Usage:
            get(key, default)           -> uses base_path
            get(path, key, default)     -> uses custom path
        """
        if len(args) == 1:
            # get(key)
            return self._fetch(self._base_path, args[0], "")
        elif len(args) == 2:
            # get(key, default) or get(path, key)
            # Assume get(key, default) - most common case
            return self._fetch(self._base_path, args[0], args[1])
        elif len(args) == 3:
            # get(path, key, default)
            return self._fetch(args[0], args[1], args[2])
        else:
            raise TypeError(f"get() takes 1 to 3 arguments ({len(args)} given)")

    @overload
    def get_int(self, key: str, default: int = 0) -> int: ...
    @overload
    def get_int(self, path: str, key: str, default: int = 0) -> int: ...

    def get_int(self, *args) -> int:
        """
        Get a configuration value as an integer.

        Usage:
            get_int(key, default)           -> uses base_path
            get_int(path, key, default)     -> uses custom path
        """
        if len(args) == 1:
            value = self._fetch(self._base_path, args[0], "0")
            default = 0
        elif len(args) == 2:
            value = self._fetch(self._base_path, args[0], str(args[1]))
            default = args[1]
        elif len(args) == 3:
            value = self._fetch(args[0], args[1], str(args[2]))
            default = args[2]
        else:
            raise TypeError(f"get_int() takes 1 to 3 arguments ({len(args)} given)")

        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid integer value: {value}, using default: {default}")
            return default

    @overload
    def get_bool(self, key: str, default: bool = False) -> bool: ...
    @overload
    def get_bool(self, path: str, key: str, default: bool = False) -> bool: ...

    def get_bool(self, *args) -> bool:
        """
        Get a configuration value as a boolean.
        Recognizes: true/True/TRUE/1/yes/Yes/YES/on as True

        Usage:
            get_bool(key, default)           -> uses base_path
            get_bool(path, key, default)     -> uses custom path
        """
        if len(args) == 1:
            value = self._fetch(self._base_path, args[0], "False")
        elif len(args) == 2:
            value = self._fetch(self._base_path, args[0], str(args[1]))
        elif len(args) == 3:
            value = self._fetch(args[0], args[1], str(args[2]))
        else:
            raise TypeError(f"get_bool() takes 1 to 3 arguments ({len(args)} given)")

        return value.lower() in ("true", "1", "yes", "on")

    @overload
    def get_float(self, key: str, default: float = 0.0) -> float: ...
    @overload
    def get_float(self, path: str, key: str, default: float = 0.0) -> float: ...

    def get_float(self, *args) -> float:
        """
        Get a configuration value as a float.

        Usage:
            get_float(key, default)           -> uses base_path
            get_float(path, key, default)     -> uses custom path
        """
        if len(args) == 1:
            value = self._fetch(self._base_path, args[0], "0.0")
            default = 0.0
        elif len(args) == 2:
            value = self._fetch(self._base_path, args[0], str(args[1]))
            default = args[1]
        elif len(args) == 3:
            value = self._fetch(args[0], args[1], str(args[2]))
            default = args[2]
        else:
            raise TypeError(f"get_float() takes 1 to 3 arguments ({len(args)} given)")

        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid float value: {value}, using default: {default}")
            return default

    # =========================================================================
    # JSON CONFIG METHODS - For JSON-structured configs {"value": "..."}
    # =========================================================================

    def _fetch_json(self, path: str, key: str, default: str) -> str:
        """Internal fetch for JSON configs with explicit path, key, default."""
        cache_key = f"json:{path}:{key}"

        # Check cache first
        cached_value, cache_hit = self._get_cached(cache_key)
        if cache_hit and cached_value is not None:
            return cached_value

        if self.is_available():
            raw_value = self._get_from_consul(path, key)
            if raw_value is not None:
                value = self._parse_json_value(raw_value)
                if value is not None:
                    self._set_cached(cache_key, value)
                    return value

        # Fall back to environment variable
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value

        return default

    @overload
    def get_json(self, key: str, default: str = "") -> str: ...
    @overload
    def get_json(self, path: str, key: str, default: str = "") -> str: ...

    def get_json(self, *args) -> str:
        """
        Get a configuration value from a JSON-structured config.
        Extracts the 'value' field from JSON: {"value": "...", "type": "...", ...}

        Usage:
            get_json(key, default)           -> uses base_path
            get_json(path, key, default)     -> uses custom path
        """
        if len(args) == 1:
            return self._fetch_json(self._base_path, args[0], "")
        elif len(args) == 2:
            return self._fetch_json(self._base_path, args[0], args[1])
        elif len(args) == 3:
            return self._fetch_json(args[0], args[1], args[2])
        else:
            raise TypeError(f"get_json() takes 1 to 3 arguments ({len(args)} given)")

    @overload
    def get_json_int(self, key: str, default: int = 0) -> int: ...
    @overload
    def get_json_int(self, path: str, key: str, default: int = 0) -> int: ...

    def get_json_int(self, *args) -> int:
        """Get a JSON config value as an integer."""
        if len(args) == 1:
            value = self._fetch_json(self._base_path, args[0], "0")
            default = 0
        elif len(args) == 2:
            value = self._fetch_json(self._base_path, args[0], str(args[1]))
            default = args[1]
        elif len(args) == 3:
            value = self._fetch_json(args[0], args[1], str(args[2]))
            default = args[2]
        else:
            raise TypeError(f"get_json_int() takes 1 to 3 arguments ({len(args)} given)")

        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid integer value: {value}, using default: {default}")
            return default

    @overload
    def get_json_bool(self, key: str, default: bool = False) -> bool: ...
    @overload
    def get_json_bool(self, path: str, key: str, default: bool = False) -> bool: ...

    def get_json_bool(self, *args) -> bool:
        """Get a JSON config value as a boolean."""
        if len(args) == 1:
            value = self._fetch_json(self._base_path, args[0], "False")
        elif len(args) == 2:
            value = self._fetch_json(self._base_path, args[0], str(args[1]))
        elif len(args) == 3:
            value = self._fetch_json(args[0], args[1], str(args[2]))
        else:
            raise TypeError(f"get_json_bool() takes 1 to 3 arguments ({len(args)} given)")

        return value.lower() in ("true", "1", "yes", "on")

    @overload
    def get_json_float(self, key: str, default: float = 0.0) -> float: ...
    @overload
    def get_json_float(self, path: str, key: str, default: float = 0.0) -> float: ...

    def get_json_float(self, *args) -> float:
        """Get a JSON config value as a float."""
        if len(args) == 1:
            value = self._fetch_json(self._base_path, args[0], "0.0")
            default = 0.0
        elif len(args) == 2:
            value = self._fetch_json(self._base_path, args[0], str(args[1]))
            default = args[1]
        elif len(args) == 3:
            value = self._fetch_json(args[0], args[1], str(args[2]))
            default = args[2]
        else:
            raise TypeError(f"get_json_float() takes 1 to 3 arguments ({len(args)} given)")

        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid float value: {value}, using default: {default}")
            return default

    @overload
    def get_json_with_metadata(self, key: str) -> Optional[Dict[str, Any]]: ...
    @overload
    def get_json_with_metadata(self, path: str, key: str) -> Optional[Dict[str, Any]]: ...

    def get_json_with_metadata(self, *args) -> Optional[Dict[str, Any]]:
        """
        Get the full JSON config including metadata.

        Usage:
            get_json_with_metadata(key)           -> uses base_path
            get_json_with_metadata(path, key)     -> uses custom path
        """
        if len(args) == 1:
            path, key = self._base_path, args[0]
        elif len(args) == 2:
            path, key = args[0], args[1]
        else:
            raise TypeError(f"get_json_with_metadata() takes 1 to 2 arguments ({len(args)} given)")

        if not self.is_available():
            return None

        raw_value = self._get_from_consul(path, key)
        if raw_value is None:
            return None

        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            return None

    def _parse_json_value(self, raw_value: str) -> Optional[str]:
        """
        Parse a JSON config and extract the 'value' field.
        If not JSON or no 'value' field, returns the raw value.
        """
        try:
            config = json.loads(raw_value)
            if isinstance(config, dict) and "value" in config:
                return str(config["value"])
            return raw_value
        except json.JSONDecodeError:
            return raw_value

    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================

    @overload
    def get_all(self) -> Dict[str, str]: ...
    @overload
    def get_all(self, path: str) -> Dict[str, str]: ...
    @overload
    def get_all(self, path: str, prefix: str) -> Dict[str, str]: ...

    def get_all(self, *args) -> Dict[str, str]:
        """
        Get all settings from a Consul path (optionally filtered by prefix).

        Usage:
            get_all()                -> all from base_path
            get_all(path)            -> all from custom path
            get_all(path, prefix)    -> filtered by prefix
        """
        if len(args) == 0:
            path, prefix = self._base_path, ""
        elif len(args) == 1:
            path, prefix = args[0], ""
        elif len(args) == 2:
            path, prefix = args[0], args[1]
        else:
            raise TypeError(f"get_all() takes 0 to 2 arguments ({len(args)} given)")

        if not self.is_available():
            return {}

        try:
            full_prefix = f"{path}/{prefix}" if prefix else path
            _, data = self._consul.kv.get(full_prefix, recurse=True)

            if data is None:
                return {}

            result = {}
            base_len = len(path) + 1  # +1 for trailing slash
            for item in data:
                key = item["Key"][base_len:] if len(item["Key"]) > base_len else ""
                if key and item.get("Value") is not None:
                    result[key] = item["Value"].decode("utf-8")

            return result
        except Exception as e:
            logger.warning(f"Failed to get all settings from Consul path '{path}': {e}")
            return {}

    # =========================================================================
    # CACHE AND CONNECTION MANAGEMENT
    # =========================================================================

    def refresh_cache(self) -> None:
        """Clear the local cache to force fresh reads from Consul."""
        with self._cache_lock:
            self._cache.clear()
        logger.debug("Consul cache cleared")

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to Consul if previously unavailable.

        Returns:
            True if reconnection successful, False otherwise
        """
        if not self._enabled:
            return False

        if self._available:
            return True

        logger.info("Attempting to reconnect to Consul...")
        return self._connect()


# Global singleton instance
consul_client = ConsulClient()


def get_consul_client() -> ConsulClient:
    """Get the global Consul client instance."""
    return consul_client
