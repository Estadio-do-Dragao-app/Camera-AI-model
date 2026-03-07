import paho.mqtt.client as mqtt
import json
import datetime
from colorama import init, Fore, Style

# Inicializar cores
init(autoreset=True)

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"{Fore.GREEN}✅ Monitor conectado ao broker!{Style.RESET_ALL}")
    client.subscribe("stadium/events/#")

def on_message(client, userdata, msg):
    try:
        # Tentar fazer parse do JSON
        payload = json.loads(msg.payload.decode())
        
        # Extrair campos importantes
        event_type = payload.get("event_type", "unknown")
        timestamp = payload.get("timestamp", "").split("T")[-1][:8] # Só hora
        
        # Campos específicos de crowd density
        if event_type == "crowd_density":
            total = payload.get("total_people", 0)
            level = payload.get("level", "?")
            metadata = payload.get("metadata", {})
            camera = metadata.get("camera_id", "Unknown")
            unit = metadata.get("coordinate_unit", "")
            
            print(f"{Fore.CYAN}[{timestamp}] {Style.BRIGHT}{event_type.upper()}{Style.RESET_ALL} | "
                  f"📷 {camera} (L{level}) | "
                  f"👥 {Fore.YELLOW}{total} pessoas{Style.RESET_ALL} | "
                  f"📏 {unit}")
            
            # Mostrar coordenadas se houver pessoas
            grid = payload.get("grid_data", [])
            if grid:
                coords = [f"({item['x']:.1f}, {item['y']:.1f}, {item.get('z', 0):.1f})" for item in grid if item['count'] > 0]
                if coords:
                    print(f"   📍 Locations (x,y,z): {', '.join(coords)}")
                  
        else:
            # Outros eventos genéricos
            print(f"{Fore.BLUE}[{timestamp}] {event_type}{Style.RESET_ALL} | {msg.topic}")
            
    except json.JSONDecodeError:
        print(f"{Fore.RED}[ERR] Mensagem não-JSON em {msg.topic}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[ERR] Erro ao processar: {e}{Style.RESET_ALL}")

# Configurar cliente (Compatível com Paho MQTT 2.x)
# Usando VERSION2 que é o novo padrão
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "monitor_logger")

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"{Fore.GREEN}✅ Monitor conectado ao broker!{Style.RESET_ALL}")
        client.subscribe("stadium/events/#")
    else:
        print(f"{Fore.RED}❌ Falha na conexão: {reason_code}{Style.RESET_ALL}")

client.on_connect = on_connect
client.on_message = on_message

print(f"{Fore.YELLOW}🔍 Iniciando monitor de eventos...{Style.RESET_ALL}")
try:
    client.connect("mosquitto", 1883, 60) # Conecta ao hostname 'mosquitto' dentro do docker network
    client.loop_forever()
except KeyboardInterrupt:
    print("\n🛑 Monitor parado.")
except Exception as e:
    print(f"\n❌ Erro de conexão: {e}")
    # Fallback para localhost se falhar (caso corra fora do docker compose)
    try:
        print(f"{Fore.YELLOW}⚠️ Tentando localhost...{Style.RESET_ALL}")
        client.connect("localhost", 1883, 60)
        client.loop_forever()
    except Exception as e2:
         print(f"❌ Falha no fallback: {e2}")
