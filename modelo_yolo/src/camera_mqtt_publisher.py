"""
Cliente MQTT para publicar eventos de crowd counting das câmeras.
Compatível com o formato do Stadium-Event-Generator.
"""
import json
import uuid
import numpy as np
from datetime import datetime

print("🚀 LOADED UPDATED MODULE: camera_mqtt_publisher.py (LIGHTWEIGHT VERSION)")

# Importar calibração de câmera
try:
    from camera_calibration import CameraCalibration
    CALIBRATION_AVAILABLE = True
except ImportError:
    print("⚠️  camera_calibration não disponível - coordenadas em pixels")
    CALIBRATION_AVAILABLE = False

# Importar MQTT client
try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("AVISO: paho-mqtt não instalado. Execute: pip install paho-mqtt")
    mqtt = None

# Tópicos MQTT (compatíveis com Stadium-Event-Generator)
MQTT_TOPIC_ALL_EVENTS = "stadium/events/all"
MQTT_TOPIC_HEATMAP = "stadium/events/congestion"


class CameraMQTTPublisher:
    """Publica eventos de crowd density para o broker MQTT"""
    
    def __init__(self, camera_id, level=0, mqtt_broker="localhost", mqtt_port=1883):
        """
        Inicializa o publisher MQTT.
        
        Args:
            camera_id: Identificador da câmera (ex: "CAM_NORTE_L0")
            level: Nível do estádio (0 ou 1)
            mqtt_broker: Host do broker MQTT
            mqtt_port: Porta do broker MQTT
        """
        self.camera_id = camera_id
        self.level = level
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_connected = False
        
        print(f"📷 Publisher iniciado para {camera_id} (Level {level})")
        
        # Carregar calibração da câmera
        if CALIBRATION_AVAILABLE:
            try:
                self.calibration = CameraCalibration(camera_id)
                print(f"   Calibração: Ativa (coordenadas em metros)")
            except Exception as e:
                print(f"   ⚠️  Calibração falhou: {e}")
                print(f"   Usando coordenadas em pixels")
                self.calibration = None
        else:
            self.calibration = None
            print(f"   Calibração: Desativada (coordenadas em pixels)")
        
        # Configurar MQTT
        self.mqtt_client = self._setup_mqtt()

    def _setup_mqtt(self):
        """Configura e conecta ao broker MQTT"""
        if mqtt is None:
            print("⚠️  MQTT desativado (paho-mqtt não instalado)")
            return None
        
        try:
            # Criar cliente MQTT
            try:
                # Para versões recentes do paho-mqtt (>= 2.0.0)
                client = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION2,
                    client_id=f"{self.camera_id}_{int(datetime.now().timestamp())}",
                    clean_session=True
                )
            except AttributeError:
                # Para versões antigas
                client = mqtt.Client(
                    client_id=f"{self.camera_id}_{int(datetime.now().timestamp())}",
                    clean_session=True
                )
            
            # Callbacks
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            
            # Conectar
            print(f"🔌 Conectando ao broker MQTT: {self.mqtt_broker}:{self.mqtt_port}")
            client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
            client.loop_start()
            
            return client
            
        except Exception as e:
            print(f"⚠️  Falha ao conectar MQTT: {e}")
            print(f"   Continuando sem publicação MQTT...")
            return None
    
    def _on_connect(self, client, userdata, flags, rc, *args):
        """Callback quando conecta ao broker"""
        if rc == 0:
            self.mqtt_connected = True
            print(f"✅ Conectado ao broker MQTT")
        else:
            print(f"❌ Falha na conexão MQTT (código {rc})")
    
    def _on_disconnect(self, client, userdata, rc, *args):
        """Callback quando desconecta"""
        self.mqtt_connected = False
        if rc != 0:
            print(f"⚠️  Conexão MQTT perdida (código {rc})")
    
    def density_map_to_grid_data(self, density_map, grid_resolution=10):
        """
        Converte density map para formato de grid_data compatível com o simulador.
        """
        height, width = density_map.shape
        grid_data = []
        
        # Criar grid
        for y in range(0, height, grid_resolution):
            for x in range(0, width, grid_resolution):
                # Extrair célula
                y_end = min(y + grid_resolution, height)
                x_end = min(x + grid_resolution, width)
                cell = density_map[y:y_end, x:x_end]
                
                # Somar densidade na célula
                cell_count = np.sum(cell)
                
                # Apenas células com densidade significativa (Lowered threshold to capture single person)
                if cell_count > 0.10:
                    grid_data.append({
                        "x": int(x + grid_resolution / 2),
                        "y": int(y + grid_resolution / 2),
                        "count": int(round(cell_count))
                    })
        
        return grid_data
    
    def generate_crowd_density_event(self, density_map, total_people, grid_resolution=10):
        """
        Gera evento de crowd density.
        """
        # Gerar grid_data em pixels
        pixel_grid_data = self.density_map_to_grid_data(density_map, grid_resolution)
        
        # Transformar para metros se calibração disponível
        if self.calibration:
            grid_data = self.calibration.transform_grid_data(pixel_grid_data)
            coordinate_unit = "meters"
            
            # Clustering simples para agrupar pontos muito próximos
            merged_grid = []
            while grid_data:
                grid_data.sort(key=lambda k: k['count'], reverse=True)
                current = grid_data.pop(0)
                
                neighbors = []
                remaining = []
                
                for other in grid_data:
                    dx = current['x'] - other['x']
                    dy = current['y'] - other['y']
                    dist = (dx*dx + dy*dy) ** 0.5
                    
                    if dist < 0.8: 
                        neighbors.append(other)
                    else:
                        remaining.append(other)
                
                merged_grid.append(current)
                grid_data = remaining
            
            grid_data = merged_grid

        else:
            grid_data = pixel_grid_data
            coordinate_unit = "pixels"
        
        # FIX: Use the ACTUAL count passed from the detector, not the number of grid cells
        # final_count = len(grid_data) 
        final_count = int(round(total_people))
        
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "crowd_density",
            "timestamp": datetime.now().isoformat() + "Z",
            "level": self.level,
            "grid_data": grid_data,
            "total_people": final_count,
            "metadata": {
                "grid_resolution": grid_resolution,
                "update_interval": 10,
                "camera_id": self.camera_id,
                "coordinate_unit": coordinate_unit
            }
        }
        
        return event
    
    def publish_event_data(self, density_map, count, grid_resolution=10):
        """
        Publica os dados de densidade processados externamente.
        
        Args:
            density_map: Mapa de densidade já calculado
            count: Contagem total já calculada
            grid_resolution: Resolução para o grid
        """
        if not self.mqtt_client or not self.mqtt_connected:
            return False
        
        try:
            if density_map is None:
                return False
                
            event = self.generate_crowd_density_event(density_map, count, grid_resolution)
            
            payload = json.dumps(event, ensure_ascii=False)
            
            self.mqtt_client.publish(MQTT_TOPIC_ALL_EVENTS, payload, qos=0)
            self.mqtt_client.publish(MQTT_TOPIC_HEATMAP, payload, qos=0)
            
            print(f"📤 Evento publicado: {int(event['total_people'])} pessoas (camera: {self.camera_id})")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao publicar evento: {e}")
            return False
    
    def disconnect(self):
        """Desconecta do broker MQTT"""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            print(f"🔌 Desconectado do broker MQTT")
