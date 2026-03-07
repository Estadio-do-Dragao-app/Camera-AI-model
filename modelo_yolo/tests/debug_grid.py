import cv2
import numpy as np
import typer
import sys
# Adicionar src ao path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from camera_calibration import CameraCalibration

def main(
    image_path: str = typer.Argument(..., help="Path to input image"),
    camera_id: str = typer.Option("CAM_001", "--camera-id", help="Camera ID"),
    output: str = typer.Option("debug_grid.jpg", "--output", help="Output image")
):
    print(f"📏 Generating Calibration Grid for {image_path}...")
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error loading image")
        return
        
    h, w = img.shape[:2]
    
    # Init Calibration
    try:
        calib = CameraCalibration(camera_id, config_path="config/camera_config.json")
        calib.img_width = w
        calib.img_height = h
        print(f"✅ Loaded Calibration: Pos=({calib.cam_x}, {calib.cam_y}, {calib.cam_z}m), Tilt={np.rad2deg(calib.tilt_rad):.1f}")
    except Exception as e:
        print(f"❌ Error loading calibration: {e}")
        return

    # Draw Grid lines every meter
    # We will project world points back to pixels? 
    # Or simply iterate pixels and draw? 
    # Current calibration class only has pixel_to_meters. 
    # We need meters_to_pixel (projection) to draw a clean grid.
    
    # Since we don't have meters_to_pixel, we can sample the image sparsely and draw the coordinates.
    
    overlay = img.copy()
    
    step = 40 # pixel step
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            
            # Simple Masking (heuristic): Ignore top 20% of image (usually ceiling) if tilt is small
            # This is just for visualization clarity
            # if y < h * 0.2: continue

            result = calib.pixel_to_meters(x, y)
            if result is None:
                continue
            real_x, real_y = result
            
            # Draw point
            cv2.circle(overlay, (x, y), 2, (0, 0, 255), -1)
            
            # Draw coordinate text if within reasonable range
            dist_sq = (real_x - calib.cam_x)**2 + (real_y - calib.cam_y)**2
            if dist_sq < 30**2: # Only draw within 30m
                label = f"{real_x:.1f},{real_y:.1f}"
                cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    
    cv2.imwrite(output, img)
    print(f"✅ Projeção de grelha salva em: {output}")

if __name__ == "__main__":
    typer.run(main)
