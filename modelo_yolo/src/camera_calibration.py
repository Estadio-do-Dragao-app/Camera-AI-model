"""
Calibração de câmera e transformação de perspectiva.
Converte coordenadas de pixels para coordenadas reais em metros.
"""
import json
import numpy as np
from pathlib import Path


class CameraCalibration:
    """
    Calibração de câmera para transformação píxel→metros.
    
    Usa parâmetros da câmera (posição, FOV, orientação) para converter
    coordenadas de pixels em coordenadas do mundo real.
    """
    
    def __init__(self, camera_id, config_path="config/camera_config.json"):
        """
        Inicializa calibração da câmera.
        
        Args:
            camera_id: ID da câmera (deve existir no config)
            config_path: Caminho para o ficheiro de configuração
        """
        self.camera_id = camera_id
        self.config = self._load_config(config_path)
        
        if camera_id not in self.config["cameras"]:
            print(f"⚠️  Câmera {camera_id} não encontrada no config")
            print(f"   Usando configuração padrão...")
            self.use_default_config()
        else:
            self.cam_config = self.config["cameras"][camera_id]
            print(f"✅ Calibração carregada para {camera_id}")
        
        # Extrair parâmetros
        self.position = self.cam_config["position"]
        self.orientation = self.cam_config["orientation"]
        self.fov = self.cam_config["fov"]
        self.resolution = self.cam_config["resolution"]
        self.coverage = self.cam_config["coverage_area"]
        
        # Pré-calcular transformação
        self._setup_transformation()
    
    def _load_config(self, config_path):
        """Carrega configuração das câmeras"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Ficheiro não encontrado: {config_path}")
            return {"cameras": {}}
        except json.JSONDecodeError as e:
            print(f"⚠️  Erro ao ler config: {e}")
            return {"cameras": {}}
    
    def use_default_config(self):
        """Configuração padrão quando câmera não está no config"""
        self.cam_config = {
            "position": {"x": 0.0, "y": 0.0, "z": 10.0},
            "orientation": {"pan": 0.0, "tilt": -30.0, "roll": 0.0},
            "fov": {"horizontal": 70.0, "vertical": 55.0},
            "resolution": {"width": 448, "height": 448},
            "coverage_area": {
                "x_min": -25.0, "x_max": 25.0,
                "y_min": 0.0, "y_max": 50.0
            }
        }
    
    def _setup_transformation(self):
        """Pré-calcula parâmetros de transformação"""
        # Posição da câmera
        self.cam_x = self.position["x"]
        self.cam_y = self.position["y"]
        self.cam_z = self.position["z"]
        
        # Orientação (converter para radianos)
        self.pan_rad = np.deg2rad(self.orientation["pan"])
        self.tilt_rad = np.deg2rad(self.orientation["tilt"])
        
        # FOV (converter para radianos)
        self.fov_h_rad = np.deg2rad(self.fov["horizontal"])
        self.fov_v_rad = np.deg2rad(self.fov["vertical"])
        
        # Resolução
        self.img_width = self.resolution["width"]
        self.img_height = self.resolution["height"]
        
        # Área de cobertura
        self.x_min = self.coverage["x_min"]
        self.x_max = self.coverage["x_max"]
        self.y_min = self.coverage["y_min"]
        self.y_max = self.coverage["y_max"]
    
    def pixel_to_meters(self, pixel_x, pixel_y):
        """
        Converte coordenadas de pixel para metros reais.
        
        Usa transformação de perspectiva simplificada baseada em:
        - Posição da câmera (altura Z)
        - Ângulo de tilt
        - Campo de visão (FOV)
        - Área de cobertura conhecida
        
        Args:
            pixel_x: Coordenada X em pixels (0 a img_width)
            pixel_y: Coordenada Y em pixels (0 a img_height)
        
        Returns:
            (real_x, real_y) em metros
        """
        # Normalizar coordenadas de pixel para [-1, 1]
        norm_x = (pixel_x / self.img_width) * 2 - 1   # -1 (esquerda) a +1 (direita)
        norm_y = (pixel_y / self.img_height) * 2 - 1  # -1 (topo) a +1 (base)
        
        # Calcular ângulos relativos para este pixel
        angle_x = norm_x * (self.fov_h_rad / 2)
        
        # CORREÇÃO:
        # Pinhole standard: Y cresce para baixo.
        # Topo da imagem (norm_y = -1) -> "Acima" do centro ótico -> Menos inclinado (mais perto do horizonte)
        # Base da imagem (norm_y = +1) -> "Abaixo" do centro ótico -> Mais inclinado (mais perto dos pés)
        
        # Se Tilt = -45 (olhando para baixo):
        # Topo deve somar ângulo positivo (subir em direção ao horizonte): -45 + 30 = -15
        # Base deve somar ângulo negativo (descer em direção aos pés): -45 - 30 = -75
        
        # Portanto, invertemos o sinal do norm_y para o cálculo do ângulo vertical
        angle_y = (-norm_y) * (self.fov_v_rad / 2) + self.tilt_rad
        
        # Debug visual (se necessário):
        # pixel_y=0 (topo) -> norm_y=-1 -> angle_y = (1)*(FOV/2) + Tilt. Ex: 30 + (-45) = -15 (Shallow/Far)
        # pixel_y=Max (base) -> norm_y=1 -> angle_y = (-1)*(FOV/2) + Tilt. Ex: -30 + (-45) = -75 (Steep/Close)
        
        # Calcular distância ao chão usando trigonometria
        # distance = height / tan(-angle_y)
        # angle_y é negativo (abaixo do horizonte). -angle_y é positivo.
        
        if angle_y > -0.01:  # Muito perto do horizonte ou acima dele
            # Assumir que é muito longe (fundo da sala) mas não descartar
            distance_ground = 100.0
        else:
            distance_ground = self.cam_z / np.tan(-angle_y)
          
        # Limitar distância máxima para evitar projeções absurdas
        distance_ground = min(distance_ground, 100.0)
        
        # Converter para coordenadas XY do mundo
        real_x = self.cam_x + distance_ground * np.sin(self.pan_rad + angle_x)
        real_y = self.cam_y + distance_ground * np.cos(self.pan_rad + angle_x)
        
        return real_x, real_y
    
    def transform_grid_data(self, pixel_grid_data):
        """
        Transforma grid_data de coordenadas de pixels para metros.
        
        Args:
            pixel_grid_data: Lista de dicts com {x, y, count} em pixels
        
        Returns:
            Lista de dicts com {x, y, count} em metros
        """
        real_grid_data = []
        
        for cell in pixel_grid_data:
            pixel_x = cell["x"]
            pixel_y = cell["y"]
            count = cell["count"]
            
            # Converter para metros
            result = self.pixel_to_meters(pixel_x, pixel_y)
            if result is None:
               continue
               
            real_x, real_y = result
            
            real_grid_data.append({
                "x": round(real_x, 2),
                "y": round(real_y, 2),
                "z": 0.0, # Assumindo chão plano (Z=0)
                "count": count
            })
        
        return real_grid_data
    
    def get_coverage_info(self):
        """Retorna informação sobre área de cobertura"""
        return {
            "camera_id": self.camera_id,
            "position": self.position,
            "coverage_area": self.coverage,
            "fov": self.fov
        }
