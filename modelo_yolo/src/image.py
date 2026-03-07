import cv2
import numpy as np
import typer
import sys
from pathlib import Path
from typing import Optional

try:
    from crowd_counter import CrowdCounter
except ImportError as e:
    print(f"❌ Erro ao importar crowd_counter: {e}")
    sys.exit(1)

def main(
    image_path: str = typer.Argument(..., help="Path to input image"),
    model: str = typer.Option(
        "model/zip_n_model_quant.onnx", 
        "--model", 
        help="Path to ONNX model"
    ),
    output: Optional[str] = typer.Option(
        None, 
        "--output", 
        help="Path to save output image (optional)"
    )
):
    """
    Processa uma única imagem usando a arquitetura CrowdCounter.
    """
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found at {image_path}")
        raise typer.Exit(1)
    
    # 1. Initialize Counter
    try:
        counter = CrowdCounter(model)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise typer.Exit(1)

    # 2. Load Image
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Error: Could not load image")
        raise typer.Exit(1)
    
    original_height, original_width = frame.shape[:2]
    print(f"Image loaded: {original_width}x{original_height}")

    # 3. Process
    print("Running inference...")
    density_map, count = counter.process_frame(frame)
    
    if density_map is None:
        print("❌ Inference failed")
        raise typer.Exit(1)

    print(f"\nestimated Crowd Count: {int(round(count))}")

    # 4. Generate Visualization
    density_resized = cv2.resize(density_map, (original_width, original_height))
    
    if density_resized.max() > density_resized.min():
        vis_map = (density_resized - density_resized.min()) / (density_resized.max() - density_resized.min())
    else:
        vis_map = np.zeros_like(density_resized)
    
    vis_map = (vis_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
    result = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    # Text Overlay
    cv2.rectangle(result, (0, 0), (400, 100), (0, 0, 0), -1)
    cv2.putText(result, f"Count: {int(round(count))}", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(result, f"Model Input: {counter.model_width}x{counter.model_height}", (10, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # 5. Save/Show
    if output:
        output_path_obj = Path(output)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path_obj), result)
        print(f"✅ Result saved to: {output}")
    
    cv2.imshow('Crowd Analysis', result)
    cv2.imshow('Original', frame)
    cv2.imshow('Heatmap', heatmap)
    
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    typer.run(main)
