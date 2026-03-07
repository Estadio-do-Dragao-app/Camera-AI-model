# 🎥 Fan App - Camera System com MQTT e Calibração

Sistema de crowd counting em tempo real com deep learning (modelo ZIP), publicação MQTT e calibração para coordenadas reais.

---

## 🚀 Quick Start

```bash
# Um único comando:
bash run.sh
```

**O script faz tudo**:
1. Verifica dependências
2. Inicia Mosquitto (Docker)
3. Executa câmera (Python local)
4. Mostra instruções

**Para sair**: Pressiona `q` na janela OpenCV

---

## 📋 Estrutura

```
fan_app/
├── run.sh                       # ⭐ Script principal
├── main.py                      # Aplicação com GUI
├── camera_mqtt_publisher.py     # Módulo MQTT
├── camera_calibration.py        # Calibração píxel→metros
├── camera_config.json           # Config de câmeras
├── model/
│   └── zip_n_model_quant.onnx  # Modelo ZIP
└── mosquitto/
    └── config/
        └── mosquitto.conf       # Config Mosquitto
```

---

## ⚙️ Uso Avançado

### Múltiplas Câmeras

```bash
# Câmera Norte
bash run.sh CAM_NORTE_L0 0 10

# Câmera Sul (noutro terminal)
bash run.sh CAM_SUL_L0 0 10
```

### Parâmetros

```bash
bash run.sh [CAMERA_ID] [LEVEL] [INTERVAL]
#           └─ Default: CAM_001
#                      └─ Default: 0
#                                └─ Default: 10 (segundos)
```

---

## 📊 Eventos MQTT

**Tópicos**:
- `stadium/events/all` - Todos os eventos
- `stadium/events/congestion` - Crowd density

**Formato**:
```json
{
  "event_type": "crowd_density",
  "level": 0,
  "grid_data": [
    {"x": -3.69, "y": 7.15, "count": 4}
  ],
  "total_people": 42,
  "metadata": {
    "camera_id": "CAM_001",
    "coordinate_unit": "meters"
  }
}
```

**Monitorizar**:
```bash
mosquitto_sub -h localhost -t "stadium/events/#" -v
```

---

## 📐 Calibração

As coordenadas são automaticamente convertidas de **pixels para metros** se a câmera estiver configurada em `camera_config.json`.

**Ver**: `CALIBRATION.md` para detalhes

---

## 🔧 Gestão do Mosquitto

```bash
# Verificar
sudo docker ps | grep mosquitto

# Parar
sudo docker stop fan-app-mosquitto

# Reiniciar
bash run.sh
```

---

## 📝 Dependências

- Python 3.12+
- OpenCV (`python3-opencv`)
- ONNX Runtime
- Paho MQTT
- Docker (para Mosquitto)

**Instalação**:
```bash
sudo apt install python3-opencv python3-numpy python3-paho-mqtt
pip3 install --break-system-packages onnxruntime
```

---

## 🎯 Compatibilidade

✅ **100% compatível com Stadium-Event-Generator**
- Mesmo formato de eventos
- Mesmos tópicos MQTT
- Coordenadas reais em metros
- Integração plug-and-play

---

## 📚 Documentação Adicional

- `CALIBRATION.md` - Detalhes da calibração
- `README.md` - Este ficheiro (original)

**Sistema pronto para produção!** 🚀
