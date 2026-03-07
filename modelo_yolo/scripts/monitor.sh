#!/bin/bash
echo "📊 Iniciando Monitor MQTT..."
echo "   (Pressiona Ctrl+C para sair)"
echo ""
.venv/bin/python3 scripts/mqtt_monitor.py "$@"
