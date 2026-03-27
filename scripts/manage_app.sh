#!/bin/bash
# scripts/manage_app.sh

SCRIPTS_DIR="/home/mu6mula/work/Quant-Guild-Library/scripts"

case "$1" in
    start)
        "$SCRIPTS_DIR/manage_backend.sh" start
        "$SCRIPTS_DIR/manage_frontend.sh" start
        ;;
    stop)
        "$SCRIPTS_DIR/manage_backend.sh" stop
        "$SCRIPTS_DIR/manage_frontend.sh" stop
        ;;
    restart)
        "$SCRIPTS_DIR/manage_backend.sh" restart
        "$SCRIPTS_DIR/manage_frontend.sh" restart
        ;;
    dev)
        echo "Starting both backend and frontend in dev mode..."
        # Start both in dev mode. Since dev mode usually stays in foreground,
        # we might want to run them in parallel or just mention that individual ones are better.
        # Running both in parallel for dev:
        "$SCRIPTS_DIR/manage_backend.sh" dev &
        "$SCRIPTS_DIR/manage_frontend.sh" dev &
        wait
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|dev}"
        exit 1
esac
