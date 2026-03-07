import cv2
import numpy as np
import typer
import sys
from ultralytics import YOLO
from crowd_counter import CrowdCounter
from test_hybrid import find_local_maxima_cv2, is_point_in_box

def get_tiled_density_map(cc, img, tile_size=256):
    h, w = img.shape[:2]
    
    # Pad to multiple of tile_size
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
    ph, pw = padded_img.shape[:2]
    
    density_accum = np.zeros((ph, pw), dtype=np.float32)
    
    rows = ph // tile_size
    cols = pw // tile_size
    
    print(f"  > Executing Tiled Density: {rows}x{cols} grid...")
    
    for r in range(rows):
        for c in range(cols):
            y1 = r * tile_size
            x1 = c * tile_size
            y2 = y1 + tile_size
            x2 = x1 + tile_size
            
            tile = padded_img[y1:y2, x1:x2]
            dmap, _ = cc.process_frame(tile)
            
            if dmap is not None:
                # Output is 16x16 or similar (low res)
                # Resize to tile_size (256x256)
                dmap_resized = cv2.resize(dmap, (tile_size, tile_size))
                density_accum[y1:y2, x1:x2] = dmap_resized
                
    # Crop back
    return density_accum[0:h, 0:w]

def main(
    image_path: str = typer.Argument(..., help="Path to input image"),
    output: str = typer.Option("tiled_hybrid_result.jpg", help="Output image")
):
    print(f"🚀 Iniciando Detecção HÍBRIDA + TILING para {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        return

    # 1. YOLO (Foreground)
    print("🧠 (1/2) YOLOv8...")
    yolo = YOLO("yolov8n.pt")
    y_results = yolo.predict(img, classes=[0], verbose=False)
    yolo_boxes = y_results[0].boxes.xyxy.cpu().numpy().astype(int)
    
    # 2. Tiled Density (Background)
    print("🧠 (2/2) Tiled Density...")
    cc = CrowdCounter()
    density_map = get_tiled_density_map(cc, img)
    
    # 3. Peak Detection on FULL MAP
    # Blur to remove grid artifacts from tiling/resizing
    density_map = cv2.GaussianBlur(density_map, (15, 15), 0)
    
    # Use relative threshold on the full map
    peaks = find_local_maxima_cv2(density_map, rel_threshold=0.08, neighborhood_size=15)
    
    print(f"📊 Stats: YOLO={len(yolo_boxes)} | Density Peaks={len(peaks)}")
    
    # 4. Fusion
    final_detections = []
    
    # Add YOLO
    for box in yolo_boxes:
        final_detections.append({'type': 'yolo', 'box': box, 'point': (int((box[0]+box[2])/2), box[3])})
        
    # Add Non-Overlapping Peaks
    added_peaks = 0
    for px, py in peaks:
        collision = False
        for box in yolo_boxes:
            if is_point_in_box((px, py), box):
                collision = True
                break
        
        if not collision:
            final_detections.append({'type': 'density', 'point': (px, py)})
            added_peaks += 1
            
    print(f"👥 TOTAL: {len(final_detections)} (YOLO={len(yolo_boxes)} + Density={added_peaks})")
    
    # 5. Draw
    viz = img.copy()
    for d in final_detections:
        if d['type'] == 'yolo':
            x1, y1, x2, y2 = d['box']
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            px, py = d['point']
            cv2.circle(viz, (px, py), 3, (0, 0, 255), -1)
            
    cv2.putText(viz, f"Total: {len(final_detections)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(output, viz)
    print(f"Saved to {output}")

if __name__ == "__main__":
    typer.run(main)
