import cv2
import numpy as np
import typer
import sys
from ultralytics import YOLO
from pathlib import Path

# Tentativa de importar modulos locais
try:
    from crowd_counter import CrowdCounter
    from camera_calibration import CameraCalibration
except ImportError as e:
    print(f"❌ Erro ao importar módulos locais: {e}")
    sys.exit(1)

def find_local_maxima_cv2(density_map, neighborhood_size=7, rel_threshold=0.1):
    """
    Encontra picos locais.
    threshold agora é relativo ao valor máximo do mapa (0.0 a 1.0).
    """
    # 1. Encontrar valor máximo para definir threshold absoluto
    max_val = density_map.max()
    abs_threshold = max_val * rel_threshold
    
    # 2. Dilatação para encontrar máximos locais
    kernel = np.ones((neighborhood_size, neighborhood_size), np.uint8)
    dilated = cv2.dilate(density_map, kernel)
    
    # 3. Verificar condições
    # - É igual ao dilatado (máximo local)
    # - É maior que o threshold (joga fora ruído de fundo)
    peaks = (density_map == dilated) & (density_map > abs_threshold)
    
    y_indices, x_indices = np.where(peaks)
    
    # Debug info
    # print(f"  [DEBUG] Max Density: {max_val:.5f} | Threshold: {abs_threshold:.5f} | Found: {len(y_indices)}")
    
    return list(zip(x_indices, y_indices))

def is_point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

def main(
    image_path: str = typer.Argument(..., help="Path to input image"),
    camera_id: str = typer.Option("CAM_001", "--camera-id", help="Camera ID in config"),
    output: str = typer.Option("hybrid_result.jpg", "--output", help="Output image path")
):
    print(f"🚀 Iniciando Detecção HÍBRIDA para {image_path}")

    # 1. Carregar Calibração
    try:
        calib = CameraCalibration(camera_id)
        print(f"✅ Calibração carregada: {calib.cam_z}m altura")
    except:
        print("⚠️ Calibração falhou/ignorada.")
        calib = None

    # 2. Carregar Imagem
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Erro imagem")
        return
    h, w = frame.shape[:2]
    if calib: 
        calib.img_width = w
        calib.img_height = h

    # 3. Executar YOLO (Foreground / Pessoas Perto)
    print("🧠 (1/2) Executando YOLOv8...")
    yolo_model = YOLO("yolov8n.pt")
    yolo_results = yolo_model.predict(frame, classes=[0], verbose=False)
    yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int) # [[x1,y1,x2,y2], ...]
    yolo_confs = yolo_results[0].boxes.conf.cpu().numpy()

    # 4. Executar Density Model (Background / Multidão)
    print("🧠 (2/2) Executando Density Model...")
    density_counter = CrowdCounter()
    density_map, _ = density_counter.process_frame(frame)
    
    if density_map is None:
        print("❌ Erro no Density Model")
        return

    # Encontrar picos no mapa ORIGINAL (Low Res) para evitar artefatos de resize
    # O modelo é 256x256 (ou similar), então neighborhood=3 já é significativo.
    raw_peaks_lowres = find_local_maxima_cv2(density_map, rel_threshold=0.1, neighborhood_size=5)
    
    # Debug Stats
    print(f"📊 Stats do Mapa de Densidade: Min={density_map.min():.5f}, Max={density_map.max():.5f}, Sum={density_map.sum():.2f}")

    # Converter picos para resolução original e guardar confiança
    # density_map.shape é (H_model, W_model)
    mh, mw = density_map.shape
    scale_x = w / mw
    scale_y = h / mh
    
    raw_peaks = []
    for px, py in raw_peaks_lowres:
        # Escalar para imagem original
        orig_x = int(px * scale_x)
        orig_y = int(py * scale_y)
        conf = density_map[py, px]
        raw_peaks.append((orig_x, orig_y, conf))
        
    print(f"📊 YOLO detetou {len(yolo_boxes)} caixas.")
    print(f"📊 Density detetou {len(raw_peaks)} picos (no mapa raw).")

    # 5. Lógica de Fusão (Hybrid Logic)
    # Se um pico cair dentro de uma caixa YOLO -> Ignorar (YOLO já pegou)
    # Se um pico estiver fora -> É uma pessoa que o YOLO falhou (provavelmente longe/cabeça)
    
    final_detections = [] # Lista de dicts
    
    # Adicionar todos YOLO
    for i, box in enumerate(yolo_boxes):
        x1, y1, x2, y2 = box
        # Ponto no chão (centro da base)
        px = int((x1+x2)/2)
        py = y2
        final_detections.append({
            "type": "yolo",
            "box": box,
            "point": (px, py),
            "conf": yolo_confs[i]
        })

    # Filtrar Picos
    valid_peaks = []
    skipped_peaks = 0
    
    for px, py, conf in raw_peaks:
        
        # Verificar colisão com QUALQUER caixa yolo
        collision = False
        for box in yolo_boxes:
            if is_point_in_box((px, py), box):
                collision = True
                break
        
        if not collision:
            valid_peaks.append((px, py))
            final_detections.append({
                "type": "density",
                "point": (px, py),
                "conf": conf * 10 # Scale for display
            })
        else:
            skipped_peaks += 1

    print(f"🔄 Fusão: {skipped_peaks} picos removidos (overlap). {len(valid_peaks)} picos mantidos (background).")
    print(f"👥 TOTAL FINAL: {len(final_detections)} pessoas.")

    # 6. Desenhar
    output_img = frame.copy()
    
    # Desenhar YOLO (Verde)
    for det in [d for d in final_detections if d['type'] == 'yolo']:
        x1, y1, x2, y2 = det['box']
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(output_img, "YOLO", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Desenhar Density (Vermelho)
    for det in [d for d in final_detections if d['type'] == 'density']:
        px, py = det['point']
        cv2.circle(output_img, (px, py), 4, (0, 0, 255), -1) # Ponto vermelho
        # cv2.circle(output_img, (px, py), 6, (0, 0, 255), 1)  # Circulo vazio
        
    # Listar Coordenadas Reais (Apenas para debug/exemplo)
    if calib:
        for det in final_detections:
            px, py = det['point']
            real = calib.pixel_to_meters(px, py)
            if real:
                label = f"{real[1]:.1f}m"
                color = (0, 255, 255) if det['type'] == 'yolo' else (0, 100, 255) # Amarelo vs Laranja
                cv2.putText(output_img, label, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Legenda
    cv2.rectangle(output_img, (0, 0), (250, 100), (0,0,0), -1)
    cv2.putText(output_img, f"Total: {len(final_detections)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(output_img, f"YOLO: {len(yolo_boxes)} (Green)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(output_img, f"Density: {len(valid_peaks)} (Red)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(output, output_img)
    print(f"✅ Salvo em: {output}")

if __name__ == "__main__":
    typer.run(main)
