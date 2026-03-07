#!/bin/bash

# ═════════════════════════════════════════════════════════════════════
# 🎥 FAN APP - Camera MQTT System
# Script de inicialização unificado
# ═════════════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "════════════════════════════════════════════════════"
echo "🚀 FAN APP - Inicialização do Sistema"
echo "════════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────────────────────────────────────
# 1. VERIFICAR DEPENDÊNCIAS
# ─────────────────────────────────────────────────────────────────────
echo "📦 Verificando dependências..."

# Configurar PYTHONPATH para incluir src/
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Verificar Python no venv
if [ ! -f ".venv/bin/python3" ]; then
    echo "❌ Virtual environment não encontrado (.venv/bin/python3)"
    echo "   Por favor crie o venv: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Verificar Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker não encontrado"
    exit 1
fi

# Verificar modelo
if [ ! -f "model/zip_n_model_quant.onnx" ]; then
    echo "❌ Modelo não encontrado: model/zip_n_model_quant.onnx"
    exit 1
fi

# Verificar módulo Python
if ! .venv/bin/python3 -c "from camera_mqtt_publisher import CameraMQTTPublisher" 2>/dev/null; then
    echo "❌ Módulo camera_mqtt_publisher com problemas"
    exit 1
fi   


echo "✅ Dependências OK"
echo ""

# ─────────────────────────────────────────────────────────────────────
# 2. INICIAR SERVIÇOS (Mosquitto + Monitor)
# ─────────────────────────────────────────────────────────────────────
echo "🐳 Iniciando serviços Docker (Broker + Monitor)..."

# Parar containers antigos (manuais) se existirem
docker stop fan-app-mosquitto fan-app-monitor 2>/dev/null || true
docker rm fan-app-mosquitto fan-app-monitor 2>/dev/null || true

# Iniciar via Compose
docker compose up -d

echo "✅ Serviços iniciados!"
echo "   Aguardando broker ficar pronto..."
sleep 3

echo ""

# ─────────────────────────────────────────────────────────────────────
# 3. CONFIGURAR X11 (para OpenCV GUI)
# ─────────────────────────────────────────────────────────────────────
echo "🖥️  Configurando display..."
xhost +local: > /dev/null 2>&1 || true
echo "✅ Display configurado"
echo ""

# ─────────────────────────────────────────────────────────────────────
# 4. PARÂMETROS DA CÂMERA
# ─────────────────────────────────────────────────────────────────────
CAMERA_ID=${1:-CAM_001}
LEVEL=${2:-0}
PUBLISH_INTERVAL=${3:-10}

echo "════════════════════════════════════════════════════"
echo "📷 Configuração da Câmera"
echo "════════════════════════════════════════════════════"
echo "Camera ID: $CAMERA_ID"
echo "Level: $LEVEL"
echo "Publish Interval: ${PUBLISH_INTERVAL}s"
echo "MQTT Broker: localhost:1883"
echo "════════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────────────────────────────────────
# 5. INSTRUÇÕES PARA MONITORIZAR
# ─────────────────────────────────────────────────────────────────────
echo "💡 Para monitorizar eventos MQTT (noutro terminal):"
echo "   mosquitto_sub -h localhost -t 'stadium/events/#' -v"
echo ""
echo "📝 Para ver LOGS COM DADOS (JSON):"
echo "   docker compose logs -f monitor"
echo ""
echo "🔌 Para ver logs do Broker (conexões):"
echo "   docker compose logs -f mosquitto"
echo ""
echo "📊 Para ver apenas crowd density:"
echo "   mosquitto_sub -h localhost -t 'stadium/events/congestion'"
echo ""
echo "════════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────────────────────────────────────
# 6. INICIAR APLICAÇÃO
# ─────────────────────────────────────────────────────────────────────
echo "🚀 Starting Camera App..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
.venv/bin/python3 src/main.py "$@" \
  --mode yolo \
  --mqtt-broker localhost \
  --mqtt-port 1883 \
  --camera-id "$CAMERA_ID" \
  --level "$LEVEL" \
  --publish-interval "$PUBLISH_INTERVAL"

# ─────────────────────────────────────────────────────────────────────
# 7. CLEANUP (quando utilizador sai)
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "✅ Aplicação encerrada"
echo ""
echo "💡 Para parar tudo:"
echo "   docker compose down"
echo "════════════════════════════════════════════════════"
