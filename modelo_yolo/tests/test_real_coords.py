import cv2
import numpy as np
import typer
import sys
from pathlib import Path

try:
    from crowd_counter import CrowdCounter
    from camera_calibration import CameraCalibration
except ImportError as e:
    print(f"❌ Erro ao importar módulos: {e}")
    sys.exit(1)

def find_local_maxima_cv2(density_map, threshold=0.01, neighborhood_size=5):
    """
    Encontra picos locais (pessoas) no mapa de densidade usando OpenCV.
    """
    # Kernel para dilatação (vizinhança)
    kernel = np.ones((neighborhood_size, neighborhood_size), np.uint8)
    
    # Dilatar: cada pixel recebe o valor máximo dos vizinhos
    dilated = cv2.dilate(density_map, kernel)
    
    # Picos são onde o valor original é igual ao dilatado (e maior que threshold)
    peaks = (density_map == dilated) & (density_map > threshold)
    
    # Obter coordenadas (y, x)
    y_indices, x_indices = np.where(peaks)
    
    return list(zip(x_indices, y_indices))

def main(
    image_path: str = typer.Argument(..., help="Path to input image"),
    camera_id: str = typer.Option("CAM_001", "--camera-id", help="Camera ID in config"),
    output: str = typer.Option("coords_result.jpg", "--output", help="Output image path")
):
    print(f"📏 Iniciando extração de coordenadas para {image_path}")
    
    # 1. Carregar Modelo
    try:
        counter = CrowdCounter()
    except Exception as e:
        print(f"❌ Erro ao carregar CrowdCounter: {e}")
        return

    # 2. Carregar Calibração
    try:
        calib = CameraCalibration(camera_id)
        print(f"✅ Calibração carregada: Camera em ({calib.cam_x}, {calib.cam_y}, {calib.cam_z}m)")
        print(f"   Tilt: {np.rad2deg(calib.tilt_rad):.1f}°, Pan: {np.rad2deg(calib.pan_rad):.1f}°")
    except Exception as e:
        print(f"❌ Erro na calibração: {e}")
        return

    # 3. Carregar Imagem
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Erro ao ler imagem")
        return
        
    h, w = frame.shape[:2]
    # Atualizar resolução na calibração
    calib.img_width = w
    calib.img_height = h

    # 4. Inferência
    print("🧠 Processando imagem...")
    density_map, count = counter.process_frame(frame)
    if density_map is None:
        return

    print(f"👥 Pessoas estimadas (Integral): {count:.1f}")

    # 5. Redimensionar mapa de densidade para tamanho original da imagem
    density_resized = cv2.resize(density_map, (w, h))
    
    # 6. Encontrar picos (pessoas individuais) usando CV2
    # Threshold baixo pois o density map tem valores pequenos
    peaks = find_local_maxima_cv2(density_resized, threshold=0.0001, neighborhood_size=15)
    print(f"📍 Picos encontrados (aprox. pessoas detectadas como pontos): {len(peaks)}")

    # 7. Desenhar e Listar
    output_img = frame.copy()
    
    print("\n--- Coordenadas Reais (Metros) ---")
    print("ID  | Pixel (X, Y) | Real (X, Y)")
    print("-" * 35)
    
    # Ordenar picos pela coordenada Y (fundo para frente) para o print ficar bonito
    peaks.sort(key=lambda p: p[1])
    
    for i, (px_x, px_y) in enumerate(peaks):
        # Converter Pixel -> Metros
        real_coords = calib.pixel_to_meters(px_x, px_y)
        
        if real_coords:
            real_x, real_y = real_coords
            
            # Printar alguns
            if i < 20: 
                print(f"{i:03d} | ({px_x:4d}, {px_y:4d}) | ({real_x:5.2f}, {real_y:5.2f})m")
            elif i == 20:
                print("... (lista truncada)")
            
            # Desenhar
            text = f"{real_y:.1f}m"
            cv2.circle(output_img, (px_x, px_y), 5, (0, 0, 255), -1) # Ponto vermelho
            cv2.circle(output_img, (px_x, px_y), 2, (0, 255, 255), -1) # Centro amarelo
            # cv2.putText(output_img, text, (px_x+5, px_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Info Box
    cv2.rectangle(output_img, (0, 0), (350, 80), (0,0,0), -1)
    cv2.putText(output_img, f"Est. Count: {int(round(count))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(output_img, f"Peaks Found: {len(peaks)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Salvar
    cv2.imwrite(output, output_img)
    print(f"\n✅ Imagem com coordenadas salva em: {output}")

if __name__ == "__main__":
    typer.run(main)
