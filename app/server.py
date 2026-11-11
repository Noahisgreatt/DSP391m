"""The whole product in ONE process on ONE port: the Docker app image and the Hugging Face Space.
    /                 the web page (app/gradio_app.py)
    /docs             the API (api/main.py): /analyze, /models, /health, /results/..., /metrics

    python -m app.server            -> http://localhost:7860   (PORT to change it)
The page talks to the API over HTTP exactly as in the two-container setup (docker-compose.yml),
here simply on the same address.
"""
import os

PORT = int(os.environ.get("PORT", 7860))
os.environ.setdefault("API_URL", f"http://127.0.0.1:{PORT}")

import gradio as gr                                    # noqa: E402
import uvicorn                                         # noqa: E402

from api.main import app as api                        # noqa: E402
from app.gradio_app import OUT, build                  # noqa: E402

app = gr.mount_gradio_app(api, build().queue(default_concurrency_limit=1), path="/", allowed_paths=[str(OUT)])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
