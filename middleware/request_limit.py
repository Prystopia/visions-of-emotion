from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Response

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

class LimitUploadSize(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length is None:
                return Response(status_code=411)  # Length Required
            if int(content_length) > MAX_UPLOAD_BYTES:
                return Response(status_code=413)  # Content Too Large
        return await call_next(request)

