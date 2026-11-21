import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_chat import router as ai_router

app = FastAPI(title="WON API")

# Comma-separated list of UI origins allowed to call this API, e.g. https://won-demo.netlify.app
# Leave unset in local dev (the Vite proxy makes CORS unnecessary).
origins = [o.strip() for o in os.getenv("WON_ALLOWED_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_methods=["POST", "GET"], allow_headers=["Content-Type"]
    )

app.include_router(ai_router)


@app.get("/health")
def health():
    return {"ok": True}
