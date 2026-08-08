"""In-process rate limiter used for call-session initiation."""

from collections import defaultdict, deque
from time import monotonic

from app.config import settings

_requests: dict[str, deque[float]] = defaultdict(deque)


def enforce_rate_limit(key: str) -> bool:
    """Return whether `key` is under the configured one-minute limit."""
    now = monotonic()
    bucket = _requests[key]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if not bucket:
        _requests.pop(key, None)
        bucket = _requests[key]
    if len(bucket) >= settings.rate_limit_per_minute:
        return False
    bucket.append(now)
    return True
