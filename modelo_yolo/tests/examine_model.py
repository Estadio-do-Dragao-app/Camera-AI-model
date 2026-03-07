import onnxruntime as ort
import sys

model_path = "model/zip_n_model_quant.onnx"

try:
    session = ort.InferenceSession(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    print("\n--- Inputs ---")
    for i in session.get_inputs():
        print(f"Name: {i.name}")
        print(f"Shape: {i.shape}")
        print(f"Type: {i.type}")
        
    print("\n--- Outputs ---")
    for o in session.get_outputs():
        print(f"Name: {o.name}")
        print(f"Shape: {o.shape}")
        print(f"Type: {o.type}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
