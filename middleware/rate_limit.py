from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict
from fastapi import Response

RATE_LIMIT = 10
WINDOW_SECONDS = 60

request_log: dict[str, list[float]] = defaultdict(list)

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        client_ip = request.client.host
        now = time.time()
        window_start = now - WINDOW_SECONDS
        timestamps = [t for t in request_log[client_ip] if t > window_start]
        if len(timestamps) >= RATE_LIMIT:
            return Response(status_code=429)
        timestamps.append(now)
        request_log[client_ip] = timestamps
        return await call_next(request)

