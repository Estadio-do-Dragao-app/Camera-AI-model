import cv2
import numpy as np
import onnxruntime as ort
import typer
import sys
from pathlib import Path
from typing import Optional

# EBC model normalization constants from the original implementation
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def preprocess_image(frame, target_size=(448, 448)):

    # Resize image to models expected input size
    image = cv2.resize(frame, target_size)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to float and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply EBC models normalization 
    image = (image - MEAN.reshape(1, 1, 3)) / STD.reshape(1, 1, 3)
    
    # Convert from HWC to CHW format
    image = image.transpose(2, 0, 1)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def process_image(image_path, model_path="model/zip_n_model_quant.onnx", output_path=None):
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return None
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return None
    
    print(f"Loading image: {image_path}")
    print(f"Using model: {model_path}")
    
    # 1. Load the image
    frame = cv2.imread(str(image_path))
    if frame is None:
        print("Error: Could not load image")
        return None
    
    original_height, original_width = frame.shape[:2]
    print(f"Original image size: {original_width}x{original_height}")
    
    # 2. Load the ONNX model
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Determine input size 
    if len(input_shape) == 4 and all(isinstance(dim, int) for dim in input_shape[2:]):
        h, w = input_shape[2], input_shape[3]
    else:
        h, w = 448, 448  # Default for EBC models on SHA dataset (ussually 448x448)
    
    print(f"Model input size: {w}x{h}")
    print(f"Model expects input: {input_shape}")

    # 3. Preprocess the image
    blob = preprocess_image(frame, target_size=(w, h))

    # 4. Run Inference
    try:
        print("Running inference...")
        outputs = session.run(None, {input_name: blob})
        density_map = outputs[0][0][0]  # Shape: [Height, Width]
    except Exception as e:
        print(f"Inference error: {e}")
        return None

    # 5. Calculate Crowd Count
    count = np.sum(density_map)
    print(f"Estimated crowd count: {int(round(count))}")

    # 6. Generate Heatmap Visualization
    # Resize density map to match original frame size for visualization
    density_resized = cv2.resize(density_map, (original_width, original_height))
    
    # Normalize density map for visualization (handle edge case of all zeros)
    if density_resized.max() > density_resized.min():
        vis_map = (density_resized - density_resized.min()) / (density_resized.max() - density_resized.min())
    else:
        vis_map = np.zeros_like(density_resized)
    
    vis_map = (vis_map * 255).astype(np.uint8)
    
    # Apply Jet colormap (Blue = Low, Red = High density)
    heatmap = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)

    # 7. Create result visualization
    result = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    # 8. Add text overlay
    # Background for text
    cv2.rectangle(result, (0, 0), (400, 100), (0, 0, 0), -1)
    
    # Count display
    cv2.putText(result, f"Count: {int(round(count))}", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Model info
    cv2.putText(result, f"Input: {w}x{h}", (10, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Image info
    cv2.putText(result, f"Image: {original_width}x{original_height}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # 9. Save or display result
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        print(f"Result saved to: {output_path}")
    
    # Display the result
    cv2.imshow('EBC Crowd Counting - Image Analysis', result)
    cv2.imshow('Original Image', frame)
    cv2.imshow('Density Heatmap', heatmap)
    
    print("Press any key to close the windows")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return int(round(count))

def main(
    image: str = typer.Argument(..., help="Path to input image"),
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
    # Process the image
    count = process_image(image, model, output)
    
    if count is not None:
        print(f"\nFinal result: {count} people detected")
    else:
        print("Processing failed")
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(main)
