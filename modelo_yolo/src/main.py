import cv2
import numpy as np
import argparse
import time
import sys
from pathlib import Path

# Importar módulos locais
try:
    from crowd_counter import CrowdCounter
    CROWD_COUNTER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Erro ao importar crowd_counter: {e}")
    sys.exit(1)

# Importar MQTT publisher
try:
    from camera_mqtt_publisher import CameraMQTTPublisher
    MQTT_AVAILABLE = True
except ImportError:
    print("⚠️  camera_mqtt_publisher não disponível")
    MQTT_AVAILABLE = False


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fan App - Real-time Crowd Counting')
    parser.add_argument('--mqtt-broker', default='localhost', help='MQTT broker host')
    parser.add_argument('--mqtt-port', type=int, default=1883, help='MQTT broker port')
    parser.add_argument('--camera-id', default='CAM_001', help='Camera identifier')
    parser.add_argument('--level', type=int, default=0, choices=[0, 1], help='Stadium level')
    parser.add_argument('--publish-interval', type=int, default=10, help='MQTT publish interval (seconds)')
    parser.add_argument('--no-mqtt', action='store_true', help='Disable MQTT publishing')
    parser.add_argument('--mode', type=str, default='yolo', choices=['yolo', 'density'], help='Detection mode: yolo or density')
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("🎥 FAN APP - Real-time Crowd Counting (Optimized Architecture)")
    print("=" * 60)
    print(f"Camera ID: {args.camera_id}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Level: {args.level}")
    
    if not args.no_mqtt and MQTT_AVAILABLE:
        print(f"MQTT Broker: {args.mqtt_broker}:{args.mqtt_port}")
        print(f"Publish Interval: {args.publish_interval}s")
        mqtt_enabled = True
    else:
        print("MQTT: Disabled")
        mqtt_enabled = False
    print("=" * 60)
    print()
    
    # 1. Initialize Crowd Counter (Loads model ONCE)
    model_path = "model/zip_n_model_quant.onnx"
    try:
        counter = CrowdCounter(mode=args.mode, model_path=model_path)
    except Exception as e:
        print(f"❌ Falha crítica ao iniciar CrowdCounter: {e}")
        return 1
    
    print()
    
    # 2. Initialize MQTT publisher if enabled
    publisher = None
    if mqtt_enabled:
        try:
            publisher = CameraMQTTPublisher(
                camera_id=args.camera_id,
                level=args.level,
                mqtt_broker=args.mqtt_broker,
                mqtt_port=args.mqtt_port
            )
            print()
        except Exception as e:
            print(f"⚠️  Falha ao inicializar MQTT: {e}")
            print("   Continuando sem MQTT...")
            publisher = None

    # 3. Start Camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Não foi possível abrir a webcam")
        return 1
    
    print("▶️  Câmera iniciada. Pressiona 'q' para sair")
    print()
    
    last_publish_time = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        # 4. SINGLE INFERENCE POINT
        # Process frame creates density map and count
        density_map, count = counter.process_frame(frame)

        if density_map is None:
            print("⚠️ Falha no processamento do frame")
            continue
        
        # 5. MQTT Publishing (Reuse results!)
        current_time = time.time()
        if publisher and (current_time - last_publish_time) >= args.publish_interval:
            try:
                # Passamos os resultados JÁ CALCULADOS para o publisher
                publisher.publish_event_data(density_map, count, grid_resolution=10)
                last_publish_time = current_time
            except Exception as e:
                print(f"❌ Erro ao publicar: {e}")

        # 6. Generate Heatmap Visualization
        # Resize density map to match original frame size for visualization
        density_resized = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))
        
        # Normalize density map for visualization
        if density_resized.max() > density_resized.min():
            vis_map = (density_resized - density_resized.min()) / (density_resized.max() - density_resized.min())
        else:
            vis_map = np.zeros_like(density_resized)
        
        vis_map = (vis_map * 255).astype(np.uint8)
        
        # Apply Jet colormap
        heatmap = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)

        # 7. Overlay Heatmap
        result = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # 8. Draw UI
        # Background for text
        cv2.rectangle(result, (0, 0), (350, 110), (0, 0, 0), -1)
        
        # Count display
        # Count display
        # Debug: Ensure count matches what is sent
        cv2.putText(result, f"People: {int(count)}", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Camera ID
        cv2.putText(result, f"Camera: {args.camera_id}", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # MQTT status
        mqtt_status = "MQTT: Connected" if (publisher and publisher.mqtt_connected) else "MQTT: Offline"
        color = (0, 255, 0) if (publisher and publisher.mqtt_connected) else (100, 100, 100)
        cv2.putText(result, mqtt_status, (10, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.imshow('EBC Crowd Counting - Real-time', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    
    # Disconnect MQTT
    if publisher:
        publisher.disconnect()
    
    print("\n✅ Aplicação encerrada")
    return 0

if __name__ == "__main__":
    sys.exit(main())