#!/bin/bash
# scripts/manage_backend.sh

# Function to start the backend
start_backend() {
    echo "Starting backend..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "Activated .venv"
    fi
    if [ -f .backend.pid ]; then
        PID=$(cat .backend.pid)
        if ps -p $PID > /dev/null; then
            echo "Backend is already running (PID: $PID)"
            return
        fi
    fi
    uv run book-app > backend.log 2>&1 &
    UV_PID=$!
    echo $UV_PID > .backend.pid
    echo "Backend started (UV PID: $UV_PID)"
}

# Function to stop the backend
stop_backend() {
    echo "Stopping backend..."
    if [ -f .backend.pid ]; then
        UV_PID=$(cat .backend.pid)
        # Kill the uv process and its children
        if ps -p $UV_PID > /dev/null; then
            pkill -P $UV_PID
            kill $UV_PID
            rm .backend.pid
            echo "Backend stopped (UV PID: $UV_PID)"
        else
            echo "Backend UV process is not running"
            rm .backend.pid
        fi
    fi
    # Cleanup any stray uvicorn processes just in case
    pkill -f "uvicorn.*book_app.backend.main:app" && echo "Cleaned up stray uvicorn processes"
}

# Function to restart the backend
restart_backend() {
    stop_backend
    sleep 2
    start_backend
}

case "$1" in
    start)
        start_backend
        ;;
    stop)
        stop_backend
        ;;
    restart)
        restart_backend
        ;;
    dev)
        echo "Starting backend in dev mode (with reload)..."
        if [ -d ".venv" ]; then
            source .venv/bin/activate
            echo "Activated .venv"
        fi
        # uv run book-app handles uvicorn.run(..., reload=True) already in main.py
        uv run book-app
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|dev}"
        exit 1
esac
