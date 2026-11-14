#!/usr/bin/env bash
# Start the product WITHOUT Docker: web page + API on one port (app/server.py). Ctrl+C stops everything.
#   bash serve.sh            page http://localhost:7860, API http://localhost:7860/docs
#   bash serve.sh --share    + ONE public https link for the page AND the API (Cloudflare quick tunnel:
#                            free, no account, a new link every start, works while this runs)
# (With Docker instead: docker compose up --build -d, see docs/mlops.md.)
cd "$(dirname "$0")"
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate dsp391m 2>/dev/null || source .venv/bin/activate 2>/dev/null || true
python -c "import gradio" 2>/dev/null || pip install -q -r requirements-ui.txt
python -c "import imageio_ffmpeg" 2>/dev/null || pip install -q imageio-ffmpeg
export PORT=${PORT:-7860}
mkdir -p outputs/api
python -m app.server > outputs/api/server.log 2>&1 &
SERVER=$!
TUNNEL=
trap 'kill $SERVER $TUNNEL 2>/dev/null' EXIT
echo "starting (loads the models, about 1 minute) ... log: outputs/api/server.log"
until curl -s "localhost:$PORT/health" >/dev/null; do
    if ! kill -0 $SERVER 2>/dev/null; then echo "the server stopped with an error:"; tail -30 outputs/api/server.log; exit 1; fi
    sleep 2
done
echo "WEB PAGE: http://localhost:$PORT      API: http://localhost:$PORT/docs"

if [[ "$1" == "--share" ]]; then
    CF=$(command -v cloudflared || echo "$HOME/.local/bin/cloudflared")
    if [[ ! -x "$CF" ]]; then                            # one-time download of Cloudflare's tunnel program
        echo "downloading cloudflared (once) ..."
        ARCH=$([[ "$(uname -m)" == aarch64 ]] && echo arm64 || echo amd64)
        mkdir -p "$(dirname "$CF")"
        curl -fsSL -o "$CF" "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$ARCH" && chmod +x "$CF"
    fi
    "$CF" tunnel --no-autoupdate --url "http://localhost:$PORT" > outputs/api/tunnel.log 2>&1 &
    TUNNEL=$!
    URL=
    for _ in $(seq 60); do
        URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' outputs/api/tunnel.log | head -1)
        [[ -n "$URL" ]] && break
        sleep 1
    done
    if [[ -n "$URL" ]]; then
        echo
        echo "PUBLIC LINK (anyone, while this runs):"
        echo "   web page:  $URL"
        echo "   API:       $URL/docs     e.g.  curl -F file=@clip.mp4 $URL/analyze"
        echo "   (uploads up to 100 MB through the tunnel; the laptop must stay on and awake)"
    else
        echo "no public link yet: see outputs/api/tunnel.log"
    fi
fi
echo "Ctrl+C stops everything"
wait $SERVER
