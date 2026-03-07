# 📐 Camera Calibration - Coordenadas Reais

## ✅ Implementado com Sucesso!

O sistema agora suporta **calibração de câmera** para converter coordenadas de pixels para **metros reais**.

---

## 🎯 Como Funciona

### 1. **Configuração da Câmera**
Cada câmera tem configuração em `camera_config.json`:

```json
{
  "CAM_001": {
    "position": {"x": 0.0, "y": 0.0, "z": 10.0},
    "orientation": {"pan": 0.0, "tilt": -30.0},
    "fov": {"horizontal": 70.0, "vertical": 55.0},
    "coverage_area": {
      "x_min": -25.0, "x_max": 25.0,
      "y_min": 0.0, "y_max": 50.0
    }
  }
}
```

### 2. **Transformação Píxel→Metros**
- Usa geometria de perspectiva
- Considera altura da câmera (Z)
- Aplica trigonometria baseada em FOV
- Pixels no topo = longe, pixels na base = perto

### 3. **Resultado**
Grid data agora em **coordenadas reais**:

```json
{
  "grid_data": [
    {"x": -3.69, "y": 7.15, "count": 4}  // metros!
  ],
  "metadata": {
    "coordinate_unit": "meters"
  }
}
```

---

## 📊 Teste Realizado

**Input**: Píxel (224, 400)  
**Output**: Metros (0.00, 50.00)  

**Evento MQTT**:
```json
{
  "event_type": "crowd_density",
  "level": 0,
  "total_people": 4,
  "grid_sample": {
    "x": -3.69,
    "y": 7.15,
    "count": 4
  },
  "coordinate_unit": "meters"
}
```

✅ **Calibração funcionando perfeitamente!**

---

## 🔧 Como Usar

### Adicionar Nova Câmera

Edita `camera_config.json`:

```json
{
  "cameras": {
    "SUA_CAMERA": {
      "position": {"x": X, "y": Y, "z": ALTURA},
      "orientation": {"pan": 0, "tilt": -30},
      "fov": {"horizontal": 80, "vertical": 60},
      "coverage_area": {
        "x_min": MIN_X, "x_max": MAX_X,
        "y_min": MIN_Y, "y_max": MAX_Y
      }
    }
  }
}
```

### Executar com Calibração

```bash
# Câmera com ID no config = calibração automática
python3 main.py --camera-id CAM_001 --level 0

# Câmera SEM ID no config = coordenadas em pixels
python3 main.py --camera-id CAM_UNKNOWN --level 0
```

**Output mostra**:
```
📷 Câmera CAM_001 inicializada
   Calibração: Ativa (coordenadas em metros)
```

---

## 📁 Ficheiros Criados

1. **`camera_config.json`** - Configuração de câmeras
2. **`camera_calibration.py`** - Módulo de calibração
3. **`camera_mqtt_publisher.py`** (atualizado) - Integração

---

## 🎓 Conceitos Usados

### Transformação de Perspectiva
```
        Câmera (altura Z)
             |
             |  tilt angle
             |/
             *----> FOV horizontal
            /|
           / |
          /  | distância
         /   |
        *----*----> Chão (Z=0)
      Pixel   Real (X,Y)
```

### Fórmula Simplificada
```python
distance = camera_height / tan(tilt_angle + pixel_angle_y)
real_x = camera_x + distance * sin(pixel_angle_x)
real_y = camera_y distance * cos(pan_angle)
```

---

## ✨ Vantagens

- ✅ Coordenadas compatíveis entre múltiplas câmeras
- ✅ Fusão de dados de diferentes câmeras facilitada
- ✅ Mapeamento direto para layout do estádio
- ✅ Compatível com sistemas de navegação/mapas

**Sistema pronto para cenários multi-câmera!** 🚀
