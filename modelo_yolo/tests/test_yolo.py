import cv2
import numpy as np
import typer
import sys
from ultralytics import YOLO
from pathlib import Path

try:
    from camera_calibration import CameraCalibration
except ImportError as e:
    print(f"❌ Erro ao importar módulos: {e}")
    sys.exit(1)

def main(
    image_path: str = typer.Argument(..., help="Path to input image"),
    camera_id: str = typer.Option("CAM_001", "--camera-id", help="Camera ID in config"),
    output: str = typer.Option("yolo_result.jpg", "--output", help="Output image path"),
    imgsz: int = typer.Option(640, "--imgsz", help="Inference size")
):
    print(f"🚀 Iniciando Detecção YOLO para {image_path} (imgsz={imgsz})")
    
    # 1. Carregar Modelo YOLO (vai baixar automaticamente o yolov8n.pt se não existir)
    try:
        model = YOLO("yolov8n.pt") 
    except Exception as e:
        print(f"❌ Erro ao carregar YOLO: {e}")
        return

    # 2. Carregar Calibração
    try:
        calib = CameraCalibration(camera_id)
        print(f"✅ Calibração carregada: Camera em ({calib.cam_x}, {calib.cam_y}, {calib.cam_z}m)")
    except Exception as e:
        print(f"❌ Erro na calibração: {e}")
        return

    # 3. Carregar Imagem
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Erro ao ler imagem")
        return
        
    h, w = frame.shape[:2]
    calib.img_width = w
    calib.img_height = h

    # 4. Inferência
    print(f"🧠 Processando imagem com YOLOv8n (sz={imgsz})...")
    # classes=0 (person)
    results = model.predict(frame, classes=[0], imgsz=imgsz, verbose=False)
    
    result = results[0]
    boxes = result.boxes
    
    print(f"👥 Pessoas detectadas: {len(boxes)}")

    # 5. Desenhar e Listar
    output_img = frame.copy()
    
    print("\n--- Coordenadas Reais (Metros) ---")
    print("ID  | Pixel (X, Y) | Real (X, Y) | Conf")
    print("-" * 45)
    
    # Ordenar por coordenada Y do fundo da caixa (quem está mais perto/longe)
    # box.xyxy: [x1, y1, x2, y2]
    # Usaremos o ponto central inferior: ( (x1+x2)/2, y2 )
    
    detections = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        
        # Ponto de referência: centro da base da caixa (pés da pessoa)
        px_x = int((x1 + x2) / 2)
        px_y = int(y2)
        
        detections.append({
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "point": (px_x, px_y),
            "conf": conf
        })

    detections.sort(key=lambda d: d["point"][1]) # Ordenar por Y
    
    for i, det in enumerate(detections):
        px_x, px_y = det["point"]
        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]
        
        # Converter Pixel -> Metros
        real_coords = calib.pixel_to_meters(px_x, px_y)
        
        real_str = "N/A"
        if real_coords:
            real_x, real_y = real_coords
            real_str = f"({real_x:5.2f}, {real_y:5.2f})"
        
        if i < 20:
            print(f"{i:03d} | ({px_x:4d}, {px_y:4d}) | {real_str} | {conf:.2f}")
            
        # Desenhar Caixa
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Desenhar Ponto de Referência
        cv2.circle(output_img, (px_x, px_y), 4, (0, 0, 255), -1)
        
        # Escrever Distância
        if real_coords:
            label = f"{real_y:.1f}m"
            cv2.putText(output_img, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Info Box
    cv2.rectangle(output_img, (0, 0), (300, 80), (0,0,0), -1)
    cv2.putText(output_img, f"YOLOp Count: {len(boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(output_img, f"Model: YOLOv8n", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Salvar
    cv2.imwrite(output, output_img)
    print(f"\n✅ Imagem com detecções salva em: {output}")

if __name__ == "__main__":
    typer.run(main)
