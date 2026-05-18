#!/bin/bash
# Server management script for web_demo

PID_FILE="server.pid"
LOG_FILE="app.log"
CONFIG_FILE="${2:-../configs/stream.yaml}"
PORT="${PORT:-5000}"

# Allow overriding GPU via env, otherwise keep user's setting.
# (Do not hardcode a single GPU index here.)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES
fi

try_activate_conda() {
  # Optional: activate conda env if conda exists and FLOODDIFFUSION_CONDA_ENV is set (default: flooddiffusion)
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh" >/dev/null 2>&1 || true
    CONDA_ENV_NAME="${FLOODDIFFUSION_CONDA_ENV:-flooddiffusion}"
    conda activate "$CONDA_ENV_NAME" >/dev/null 2>&1 || true
  fi
}

case "$1" in
    start)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo "Server is already running (PID: $PID)"
                exit 1
            else
                echo "Removing stale PID file"
                rm -f "$PID_FILE"
            fi
        fi
        
        echo "Starting server..."
        echo "Config file: $CONFIG_FILE"
        echo "Port: $PORT"
        try_activate_conda
        PY_BIN="$(command -v python || true)"
        if [ -z "$PY_BIN" ]; then
          PY_BIN="$(command -v python3 || true)"
        fi
        if [ -z "$PY_BIN" ]; then
          echo "Error: neither python nor python3 found in PATH"
          exit 1
        fi
        # Run from web_demo/ so relative paths work as expected.
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        cd "$SCRIPT_DIR" || exit 1
        nohup "$PY_BIN" app.py --config "$CONFIG_FILE" --port "$PORT" > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        sleep 3
        
        if ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
            echo "Server started successfully (PID: $(cat $PID_FILE))"
            curl -s "http://localhost:${PORT}/api/status"
        else
            echo "Failed to start server"
            rm -f "$PID_FILE"
            exit 1
        fi
        ;;
        
    stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "No PID file found. Killing all python app.py processes..."
            pkill -9 -f "python app.py"
            exit 0
        fi
        
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Stopping server (PID: $PID)..."
            kill -9 $PID
            rm -f "$PID_FILE"
            echo "Server stopped"
        else
            echo "Server is not running"
            rm -f "$PID_FILE"
        fi
        ;;
        
    restart)
        $0 stop
        sleep 2
        $0 start "$2"
        ;;
        
    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo "Server is running (PID: $PID)"
                curl -s "http://localhost:${PORT}/api/status"
            else
                echo "PID file exists but process is not running"
                rm -f "$PID_FILE"
                exit 1
            fi
        else
            echo "Server is not running"
            exit 1
        fi
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status} [config_file]"
        echo ""
        echo "Commands:"
        echo "  start [config_file]  - Start the server with optional config file"
        echo "  stop                 - Stop the server"
        echo "  restart [config_file]- Restart the server with optional config file"
        echo "  status               - Check server status"
        echo ""
        echo "Examples:"
        echo "  $0 start                              # Use default config (../configs/stream.yaml)"
        echo "  $0 start ../configs/stream_tiny.yaml  # Use custom config"
        echo "  $0 restart ../configs/stream.yaml"
        exit 1
        ;;
esac
