import os
import time
import json
import math
import random
import logging
import paho.mqtt.client as mqtt

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Variáveis do Broker MQTT via Ambiente
MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "teg/bancada/telemetria")
PUBLISH_INTERVAL = float(os.getenv("PUBLISH_INTERVAL", 1.0))

def generate_synthetic_teg_data(step):
    """
    Gera dados sintéticos simulando um ciclo térmico no TEG.
    A temperatura do lado quente flutua senoidalmente entre 45°C e 95°C.
    """
    # 1. Simulação Térmica
    t_frio = 25.0 + random.uniform(-0.2, 0.2) # Temperatura ambiente levemente ruidosa
    # Flutuação senoidal suave de temperatura
    t_quente = 70.0 + 25.0 * math.sin(step * 0.05) + random.uniform(-0.3, 0.3)
    
    delta_t = max(0.1, t_quente - t_frio)

    # 2. Resposta Elétrica Sintética Baseada na Física do TEG
    # Coeficiente Seebeck aproximado do módulo: ~0.085 V/K
    seebeck_alpha = 0.085
    r_interna = 1.6 # Ohms
    r_carga = 2.0   # Carga resistiva conectada em Ohms

    v_oc = seebeck_alpha * delta_t # Tensão em circuito aberto
    i_amp = v_oc / (r_interna + r_carga) # Corrente de circuito
    v_medida = i_amp * r_carga # Tensão sobre a carga

    # Adicionar ruído de medição dos sensores (ADC / MAX6675 / INA219)
    v_medida_ruido = max(0.0, v_medida + random.uniform(-0.02, 0.02))
    i_mA_ruido = max(0.0, (i_amp * 1000.0) + random.uniform(-2.0, 2.0))
    p_mW_ruido = v_medida_ruido * i_mA_ruido

    # Opcional: Injetar uma anomalia esporádica para testar os alertas do Gêmeo Digital
    # A cada 200 passos, simula uma degradação temporária na pasta térmica ou trinca
    if (step % 200) > 180:
        v_medida_ruido *= 0.75 # Queda de tensão por alta resistência interna
        p_mW_ruido = v_medida_ruido * i_mA_ruido

    payload = {
        "device_id": "ESP32_SIMULADO",
        "t_quente": round(t_quente, 2),
        "t_frio": round(t_frio, 2),
        "tensao_V": round(v_medida_ruido, 3),
        "corrente_mA": round(i_mA_ruido, 2),
        "potencia_mW": round(p_mW_ruido, 2)
    }

    return payload

def main():
    client = mqtt.Client(client_id="TEG_Hardware_Simulator")

    connected = False
    while not connected:
        try:
            logging.info(f"Conectando ao Broker MQTT em {MQTT_BROKER}:{MQTT_PORT}...")
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            connected = True
            logging.info("Conexão MQTT estabelecida com sucesso!")
        except Exception as e:
            logging.warning(f"Aguardando Mosquitto iniciar... Erro: {e}")
            time.sleep(2)

    step = 0
    client.loop_start()

    try:
        while True:
            telemetry = generate_synthetic_teg_data(step)
            json_payload = json.dumps(telemetry)
            
            client.publish(MQTT_TOPIC, json_payload)
            logging.info(f"Publicado no tópico '{MQTT_TOPIC}': {json_payload}")

            step += 1
            time.sleep(PUBLISH_INTERVAL)
    except KeyboardInterrupt:
        logging.info("Encerrando simulador...")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()