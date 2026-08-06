import os
import time
import json
import math
import random
import logging
import paho.mqtt.client as mqtt
from ekf import *

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurações do Broker MQTT via variáveis de ambiente
MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "teg/bancada/telemetria")
PUBLISH_INTERVAL = float(os.getenv("PUBLISH_INTERVAL", 1.0))

def generate_physical_data(step, ekf_filter):
    """
    Simula os sensores físicos ruidosos da bancada e aplica o EKF
    para estimar a resistência interna em tempo real.
    """
    # 1. Leitura bruta/ruidosa das temperaturas
    t_frio = 25.0 + random.uniform(-0.3, 0.3)
    t_quente = 70.0 + 25.0 * math.sin(step * 0.05) + random.uniform(-0.5, 0.5)
    delta_t = max(0.1, t_quente - t_frio)

    # Parametrização física base do TEG
    seebeck_alpha = 0.085
    r_interna_real = 1.6  # Resistência nominal
    r_carga = 2.0

    # Injeção de Anomalia Física (a cada 200 passos simula degradação por trinca/solda)
    if (step % 200) > 180:
        r_interna_real = 3.2  # Dobra a resistência interna real devido à degradação

    v_oc = seebeck_alpha * delta_t
    i_amp = v_oc / (r_interna_real + r_carga)
    v_medida = i_amp * r_carga

    # Adiciona ruído de medição aos sensores (ADC / MAX6675 / INA219)
    v_medida_ruido = max(0.0, v_medida + random.uniform(-0.02, 0.02))
    i_mA_ruido = max(0.0, (i_amp * 1000.0) + random.uniform(-2.0, 2.0))
    p_mW = v_medida_ruido * i_mA_ruido

    # 2. Processamento EKF em tempo real
    r_interna_estimada = ekf_filter.update(t_quente, t_frio, i_mA_ruido, v_medida_ruido)

    # 3. Payload contendo os dados físicos + estado estimado pelo EKF
    return {
        "device_id": "ESP32_BANCADA",
        "t_quente": round(t_quente, 2),
        "t_frio": round(t_frio, 2),
        "tensao_V": round(v_medida_ruido, 3),
        "corrente_mA": round(i_mA_ruido, 2),
        "potencia_mW": round(p_mW, 2),
        "r_interna_ekf": round(r_interna_estimada, 3)  # <-- Estado Oculto Estimado pelo EKF!
    }


def main():
    # Instancia o filtro EKF fora do loop para manter o estado persistente
    ekf = TEGExtendedKalmanFilter(seebeck_alpha=0.085, r_init=1.6)

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
            telemetry = generate_physical_data(step, ekf)
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