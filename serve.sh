#!/usr/bin/env bash
# Start the product WITHOUT Docker: API (:8000) + web page (:7860). Ctrl+C stops both.
#   bash serve.sh            then open http://localhost:7860
#   bash serve.sh --share    also prints a public link (https://....gradio.live, 72 h) anyone can open
# (With Docker instead: docker compose up --build -d, see docs/mlops.md.)
cd "$(dirname "$0")"
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate dsp391m 2>/dev/null || source .venv/bin/activate 2>/dev/null || true
python -c "import gradio" 2>/dev/null || pip install -q -r requirements-ui.txt
python -c "import imageio_ffmpeg" 2>/dev/null || pip install -q imageio-ffmpeg
mkdir -p outputs/api
uvicorn api.main:app --host 0.0.0.0 --port 8000 > outputs/api/api.log 2>&1 &
API=$!
trap 'kill $API 2>/dev/null' EXIT
echo "starting the API (loads the models, about 1 minute) ... log: outputs/api/api.log"
until curl -s localhost:8000/health >/dev/null; do
    if ! kill -0 $API 2>/dev/null; then echo "the API stopped with an error:"; tail -30 outputs/api/api.log; exit 1; fi
    sleep 2
done
echo "API ready: http://localhost:8000/docs"
echo "WEB PAGE:  http://localhost:7860   (Ctrl+C here stops both)"
[[ "$1" == "--share" ]] && export UI_SHARE=1 && echo "public link: wait for 'Running on public URL' below"
python -m app.gradio_app
