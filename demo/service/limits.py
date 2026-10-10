"""
Cost/abuse guards for the free-trial demo.

Deliberately simple and in-process: Render's free plan runs a single instance,
so an in-memory limiter and cache are enough. They reset on restart, which is
acceptable for a demo. When the trial budget is exhausted the API returns a
clear message and the page falls back to bring-your-own-key.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import deque
from typing import Any


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class DemoGuard:
    """Per-IP rate limit + global daily cap + response cache."""

    def __init__(
        self,
        *,
        per_ip_limit: int | None = None,
        per_ip_window_s: int = 600,
        daily_cap: int | None = None,
        cache_ttl_s: int = 86_400,
        cache_max: int = 500,
    ) -> None:
        self.per_ip_limit = per_ip_limit if per_ip_limit is not None else _int_env("DEMO_PER_IP_LIMIT", 3)
        self.per_ip_window_s = per_ip_window_s
        self.daily_cap = daily_cap if daily_cap is not None else _int_env("DEMO_DAILY_CAP", 200)
        self.cache_ttl_s = cache_ttl_s
        self.cache_max = cache_max
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}
        self._day = ""
        self._day_count = 0
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    # -- rate limiting ------------------------------------------------------

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def check(self, ip: str) -> tuple[bool, str | None]:
        """Return (allowed, reason). Records the hit when allowed."""
        now = time.time()
        with self._lock:
            day = self._today()
            if day != self._day:
                self._day = day
                self._day_count = 0
            if self._day_count >= self.daily_cap:
                return False, "daily"

            hits = self._hits.setdefault(ip, deque())
            cutoff = now - self.per_ip_window_s
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self.per_ip_limit:
                return False, "rate"

            hits.append(now)
            self._day_count += 1
            return True, None

    @property
    def daily_remaining(self) -> int:
        with self._lock:
            if self._today() != self._day:
                return self.daily_cap
            return max(0, self.daily_cap - self._day_count)

    # -- cache --------------------------------------------------------------

    @staticmethod
    def key(message: str, provider: str, model: str) -> str:
        raw = f"{provider}:{model}:{message}".encode()
        return hashlib.sha256(raw).hexdigest()

    def cache_get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.time() - ts > self.cache_ttl_s:
                self._cache.pop(key, None)
                return None
            return value

    def cache_set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            if len(self._cache) >= self.cache_max:
                oldest = min(self._cache.items(), key=lambda kv: kv[1][0])[0]
                self._cache.pop(oldest, None)
            self._cache[key] = (time.time(), value)
