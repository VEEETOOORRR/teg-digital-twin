import os
import time
import logging
import torch
import torch.nn as nn
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Variáveis de Ambiente
INFLUX_URL = os.getenv("INFLUX_URL", "http://influxdb:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "my-super-secret-admin-token")
INFLUX_ORG = os.getenv("INFLUX_ORG", "teg_org")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "teg_telemetry")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/teg_model.pt")


class TEGDigitalTwinNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=2):
        super(TEGDigitalTwinNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)


def create_default_model_if_missing(model_path, model_class):
    """Gera um modelo PyTorch inicial caso o arquivo de pesos ainda não exista."""
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        logging.warning(f"⚠️ Modelo '{model_path}' não encontrado. Gerando modelo inicial com pesos aleatórios...")
        dummy_model = model_class()
        torch.save(dummy_model.state_dict(), model_path)
        logging.info(f"✅ Modelo inicial salvo em '{model_path}'")


def get_latest_measurement_record(query_api, measurement):
    """Realiza uma única consulta otimizada trazendo todos os campos da measurement."""
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30s)
      |> filter(fn: (r) => r["_measurement"] == "{measurement}")
      |> last()
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    try:
        tables = query_api.query(query=query)
        for table in tables:
            for record in table.records:
                return record.values
    except Exception as e:
        logging.warning(f"Aguardando dados da measurement '{measurement}': {e}")
    return {}


def main():
    # Inicialização da Conexão com InfluxDB
    client = None
    while client is None:
        try:
            logging.info(f"Conectando ao InfluxDB em {INFLUX_URL}...")
            client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
            query_api = client.query_api()
            write_api = client.write_api(write_options=SYNCHRONOUS)
            logging.info("Conectado ao InfluxDB com sucesso!")
        except Exception as e:
            logging.warning(f"Aguardando InfluxDB iniciar... Erro: {e}")
            client = None
            time.sleep(3)

    # Garante a existência do arquivo de modelo
    create_default_model_if_missing(MODEL_PATH, TEGDigitalTwinNet)

    # Carrega a rede neural PyTorch
    model = TEGDigitalTwinNet()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    logging.info("🧠 Modelo PyTorch carregado e pronto para inferências!")

    while True:
        try:
            # 1. Consulta otimizada dos dados brutos do ESP32/Simulador
            esp32_data = get_latest_measurement_record(query_api, "esp32_telemetry")
            
            # 2. Consulta otimizada dos dados processados pelo ANSYS (teg_analytics)
            ansys_data = get_latest_measurement_record(query_api, "teg_analytics")

            # Extração dos valores necessários
            t_q = esp32_data.get("t_quente")
            t_f = esp32_data.get("t_frio")
            v_real = esp32_data.get("tensao_V")
            p_real = esp32_data.get("potencia_mW")

            v_ideal = ansys_data.get("v_ideal")
            p_ideal = ansys_data.get("p_ideal_mW")

            # Executa a inferência apenas quando todos os dados estiverem disponíveis
            if None not in [t_q, t_f, v_real, p_real, v_ideal, p_ideal]:
                features = [float(t_q), float(t_f), float(v_real), float(v_ideal), float(p_real), float(p_ideal)]
                input_tensor = torch.tensor([features], dtype=torch.float32)

                with torch.no_grad():
                    output = model(input_tensor).squeeze().numpy()

                score_degradacao = float(output[0])
                eficiencia_relativa = float(output[1] * 100.0)

                # Gravando resultado da IA no InfluxDB
                point = Point("teg_analytics") \
                    .tag("engine", "pytorch") \
                    .field("score_degradacao_ml", score_degradacao) \
                    .field("eficiencia_relativa_ml", eficiencia_relativa)

                write_api.write(bucket=INFLUX_BUCKET, record=point)
                logging.info(
                    f"⚡ [ML Predict] Degradação IA: {score_degradacao:.3f} | "
                    f"Eficiência IA: {eficiencia_relativa:.1f}% | Entrada: ΔT={(t_q-t_f):.1f}°C"
                )
            else:
                logging.info("Aguardando sincronização completa de dados (ESP32 + ANSYS Twin)...")

        except Exception as e:
            logging.error(f"Erro na execução da inferência: {e}")

        time.sleep(1)


if __name__ == "__main__":
    main()