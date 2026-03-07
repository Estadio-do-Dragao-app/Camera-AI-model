import cv2
import numpy as np
import typer
import sys
from crowd_counter import CrowdCounter

def main(
    image_path: str = typer.Argument(..., help="Path to input image"),
    overlap: bool = typer.Option(False, help="Use overlap (not implemented simple grid)"),
    output: str = typer.Option("tiled_result.jpg", help="Output image")
):
    print(f"🧩 Testing Tiling Logic on {image_path}")
    
    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error loading image")
        return

    h, w, c = img.shape
    print(f"Original Shape: {w}x{h}")

    # Initialize Model
    # Note: Model expects 256x256
    cc = CrowdCounter()
    input_size = 256
    
    # Calculate number of tiles
    # We will simply resize the image to be a multiple of 256 to avoid partial tiles for this test
    # If we want to detect 300 people, we probably need good resolution.
    # Let's try resizing the image to keeping aspect ratio but ensuring enough pixels per person.
    # If a head is 20px, and we have 256px input, we can fit ~12 heads across. 
    # 300 people -> sqrt(300) = 17x17 grid of people.
    # We might need 2x2 or 3x3 tiles of 256x256 to cover the image with sufficient detail.
    
    # Strategy: Resize image so that it is composed of N x M tiles of 256x256.
    # Let's try to maintain original resolution roughly.
    # 640x480 -> 2.5 tiles x 1.8 tiles.
    # If the image is high res (e.g. 1920x1080), we have many tiles.
    
    # Let's verify input image size first
    print(f"Processing...")
    
    # Pad image to be multiple of 256
    pad_h = (256 - h % 256) % 256
    pad_w = (256 - w % 256) % 256
    
    padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
    ph, pw = padded_img.shape[:2]
    
    print(f"Padded Shape: {pw}x{ph}")
    
    total_count = 0
    density_map_accum = np.zeros((ph, pw), dtype=np.float32)
    
    vis_img = padded_img.copy()
    
    rows = ph // 256
    cols = pw // 256
    
    print(f"Grid: {rows}x{cols} tiles ({rows*cols} inferences)")
    
    for r in range(rows):
        for c in range(cols):
            y1 = r * 256
            x1 = c * 256
            y2 = y1 + 256
            x2 = x1 + 256
            
            tile = padded_img[y1:y2, x1:x2]
            
            # Run inference
            dmap, count = cc.process_frame(tile)
            
            if dmap is not None:
                # Resize dmap (which is 16x16 output usually) back to 256x256 for visualization
                # Wait, examine_model said output is 16x16. 
                # Does process_frame return 256x256?
                # Let's check crowd_counter.py process_frame implementation output.
                # It returns `outputs[0][0][0]`. If output is 16x16, then dmap is 16x16.
                
                # We should accumulate the count
                total_count += count
                
                # For viz, resize dmap to 256x256
                dmap_resized = cv2.resize(dmap, (256, 256))
                
                # Place in accumulator
                density_map_accum[y1:y2, x1:x2] = dmap_resized
                
                # Draw grid
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(vis_img, f"{count:.1f}", (x1+10, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            else:
                print(f"❌ Failed inference on tile {r},{c}")

    print(f"✅ Total Count (Tiled): {total_count:.2f}")
    
    # Save result
    # Normalize density map for viz
    if density_map_accum.max() > 0:
        dm_norm = 255 * (density_map_accum / density_map_accum.max())
        dm_norm = dm_norm.astype(np.uint8)
        heatmap = cv2.applyColorMap(dm_norm, cv2.COLORMAP_JET)
        result = cv2.addWeighted(vis_img, 0.6, heatmap, 0.4, 0)
    else:
        result = vis_img
        
    # Crop back to original size
    final_result = result[0:h, 0:w]
    
    cv2.putText(final_result, f"Tiled Count: {total_count:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output, final_result)
    print(f"Saved to {output}")

if __name__ == "__main__":
    typer.run(main)
