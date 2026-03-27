#!/bin/bash
# scripts/manage_frontend.sh

FRONTEND_DIR="/home/mu6mula/work/Quant-Guild-Library/book_app/frontend"

# Function to start the frontend
start_frontend() {
    echo "Starting frontend..."
    if [ -f .frontend.pid ]; then
        PID=$(cat .frontend.pid)
        if ps -p $PID > /dev/null; then
            echo "Frontend is already running (PID: $PID)"
            return
        fi
    fi
    cd "$FRONTEND_DIR" && npm run dev:no-turbo > frontend.log 2>&1 &
    echo $! > /home/mu6mula/work/Quant-Guild-Library/scripts/.frontend.pid
    echo "Frontend started (PID: $!)"
}

# Function to stop the frontend
stop_frontend() {
    echo "Stopping frontend..."
    PID_FILE="/home/mu6mula/work/Quant-Guild-Library/scripts/.frontend.pid"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null; then
            kill $PID
            rm "$PID_FILE"
            echo "Frontend stopped (PID: $PID)"
        else
            echo "Frontend is not running"
            rm "$PID_FILE"
        fi
    else
        echo "Frontend is not running"
    fi
}

# Function to restart the frontend
restart_frontend() {
    stop_frontend
    sleep 2
    start_frontend
}

case "$1" in
    start)
        start_frontend
        ;;
    stop)
        stop_frontend
        ;;
    restart)
        restart_frontend
        ;;
    dev)
        echo "Starting frontend in dev mode..."
        cd "$FRONTEND_DIR" && npm run dev:no-turbo
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|dev}"
        exit 1
esac
