"""Protects the AI route on a public demo: per-IP rate limit plus a global daily cap.

In-memory, so it resets on restart and is per process. Fine for a single-instance demo.
Env: WON_RATE_PER_MIN (default 8 per IP), WON_DAILY_CAP (default 300 requests in total).
"""
import os
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

RATE_PER_MIN = int(os.getenv("WON_RATE_PER_MIN", "8"))
DAILY_CAP = int(os.getenv("WON_DAILY_CAP", "300"))


class Limiter:
    def __init__(self, per_min: int, daily: int, clock=time.time):
        self.per_min, self.daily, self.clock = per_min, daily, clock
        self.hits: dict[str, deque] = defaultdict(deque)
        self.day, self.day_count = self._today(), 0

    def _today(self) -> int:
        return int(self.clock() // 86400)

    def check(self, ip: str) -> str | None:
        """Returns None if allowed, else the reason it was refused."""
        now = self.clock()
        if self._today() != self.day:
            self.day, self.day_count = self._today(), 0
        if self.day_count >= self.daily:
            return "The demo has reached its daily limit. Please try again tomorrow."
        q = self.hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_min:
            return "Too many requests. Please wait a minute."
        q.append(now)
        self.day_count += 1
        return None


limiter = Limiter(RATE_PER_MIN, DAILY_CAP)


async def guard(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    reason = limiter.check(ip)
    if reason:
        raise HTTPException(429, reason)
