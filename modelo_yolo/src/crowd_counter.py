import cv2
import numpy as np
import time
import onnxruntime as ort
from ultralytics import YOLO

class CrowdCounter:
    def __init__(self, mode="yolo", model_path="model/zip_n_model_quant.onnx", yolo_model="yolov8n.pt"):
        """
        Initializes CrowdCounter in either 'yolo' or 'density' mode.
        """
        self.mode = mode
        print(f"🚀 Initializing CrowdCounter in [{self.mode.upper()}] mode...")
        
        if self.mode == "yolo":
            try:
                self.model = YOLO(yolo_model)
                print(f"✅ YOLO Model Loaded: {yolo_model}")
            except Exception as e:
                raise RuntimeError(f"Failed to load YOLO model: {e}")
                
        elif self.mode == "density":
            try:
                self.session = ort.InferenceSession(model_path)
                self.input_name = self.session.get_inputs()[0].name
                # Model expects 256x256
                self.model_width = 256
                self.model_height = 256
                print(f"✅ Density Model Loaded: {model_path} ({self.model_width}x{self.model_height})")
            except Exception as e:
                raise RuntimeError(f"Failed to load ONNX model: {e}")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def preprocess_density(self, frame):
        """Prepare image for density model"""
        img = cv2.resize(frame, (self.model_width, self.model_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # Standard ImageNet Normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def process_frame(self, frame):
        if self.mode == "yolo":
            return self._process_yolo(frame)
        elif self.mode == "density":
            return self._process_density(frame)
        return None, 0

    def _process_yolo(self, frame):
        """Fast inference using ONLY YOLO."""
        if frame is None: return None, 0
        
        # 1. Run YOLO
        results = self.model.predict(frame, classes=[0], verbose=False)
        
        # 2. Extract
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        count = len(boxes)
        
        # 3. Synthetic Density Map
        h, w = frame.shape[:2]
        density_map = np.zeros((h, w), dtype=np.float32)
        
        for box in boxes:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            sigma = max(5, (x2 - x1) // 6)
            radius = int(sigma * 2)
            cv2.circle(density_map, (cx, cy), radius, 1.0, -1)

        if count > 0:
            density_map = cv2.GaussianBlur(density_map, (31, 31), 0)
            cur_sum = np.sum(density_map)
            if cur_sum > 0:
                density_map = density_map * (count / cur_sum)
        
        return density_map, count

    def _process_density(self, frame):
        """Standard Density Model inference."""
        if frame is None: return None, 0

        # Preprocess
        blob = self.preprocess_density(frame)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: blob})
        dmap = outputs[0][0][0]
        
        # Simple noise filter
        # 1. Thresholding (User reported best visual results with Morphological + 0.01)
        dmap[dmap < 0.01] = 0
        
        # 2. Morphological Opening to remove small noise speckles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dmap = cv2.morphologyEx(dmap, cv2.MORPH_OPEN, kernel)
        
        count = np.sum(dmap)
        
        # Resize to original frame size for consistency?
        # Actually main.py handles resizing for viz. 
        # But we should return the raw map.
        
        return dmap, count
