import os
import json
import time
import logging
import numpy as np
from scipy.interpolate import interp1d
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Variáveis de Ambiente do InfluxDB
INFLUX_URL = os.getenv("INFLUX_URL", "http://influxdb:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "my-super-secret-admin-token")
INFLUX_ORG = os.getenv("INFLUX_ORG", "teg_org")
INFLUX_BUCKET_RAW = os.getenv("INFLUX_BUCKET_RAW", "teg_telemetry")
INFLUX_BUCKET_ANALYTICS = os.getenv("INFLUX_BUCKET_ANALYTICS", "teg_telemetry")
INFLUX_MEASUREMENT = os.getenv("INFLUX_MEASUREMENT", "esp32_telemetry")
LUT_FILEPATH = os.getenv("LUT_FILEPATH", "/app/models/ansys_lut.json")


def generate_default_lut(filepath):
    """Gera uma tabela Look-Up (LUT) teórica do ANSYS caso o arquivo não exista."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    delta_ts = np.linspace(5.0, 120.0, 24)
    data = []
    
    for dt in delta_ts:
        v_oc = 0.085 * dt
        r_int = 1.2 + 0.002 * dt
        p_max = ((v_oc ** 2) / (4.0 * r_int)) * 1000.0  # mW
        data.append({
            "delta_t": float(round(dt, 2)),
            "v_oc_ideal": float(round(v_oc, 4)),
            "r_int_nominal": float(round(r_int, 4)),
            "p_max_ideal": float(round(p_max, 2))
        })
    
    with open(filepath, 'w') as f:
        json.dump({"lut_data": data}, f, indent=2)
    logging.info(f"📁 Tabela LUT do ANSYS autogerada em: '{filepath}'")


class TEGDigitalTwinEngine:
    def __init__(self, lut_filepath):
        if not os.path.exists(lut_filepath):
            generate_default_lut(lut_filepath)

        logging.info(f"Carregando matriz de dados do ANSYS de '{lut_filepath}'...")
        with open(lut_filepath, 'r') as f:
            data = json.load(f)["lut_data"]
        
        # Extrair vetores para interpolação contínua (1D Spline)
        delta_ts = [d["delta_t"] for d in data]
        v_ocs = [d["v_oc_ideal"] for d in data]
        p_maxs = [d["p_max_ideal"] for d in data]
        r_ints = [d["r_int_nominal"] for d in data]

        # Funções de Interpolação para o Modelo Virtual
        self.interp_v_oc = interp1d(delta_ts, v_ocs, fill_value="extrapolate")
        self.interp_p_max = interp1d(delta_ts, p_maxs, fill_value="extrapolate")
        self.interp_r_int = interp1d(delta_ts, r_ints, fill_value="extrapolate")

    def inferir(self, t_quente, t_frio, v_real, i_real_mA, p_real_mW, r_ekf=None):
        delta_t = max(0.1, t_quente - t_frio)
        i_real_A = i_real_mA / 1000.0

        # 1. Simulação ANSYS em Tempo Real (Estimativa Ideal do Modelo)
        v_oc_ideal = float(self.interp_v_oc(delta_t))
        p_max_ideal_mW = float(self.interp_p_max(delta_t))
        r_int_nominal = float(self.interp_r_int(delta_t))

        # Tensão teórica sob a mesma corrente de carga
        v_ideal = max(0.0, v_oc_ideal - (i_real_A * r_int_nominal))
        p_ideal_mW = max(0.0, v_ideal * i_real_mA)

        # 2. Métricas de Diagnóstico do Gêmeo Digital
        residuo_tensao = abs(v_real - v_ideal)
        
        # Eficiência relativa em relação ao modelo ideal do ANSYS (%)
        eficiencia_relativa = (p_real_mW / p_ideal_mW * 100.0) if p_ideal_mW > 0.05 else 100.0
        eficiencia_relativa = min(120.0, max(0.0, eficiencia_relativa))

        # 3. Estimativa da Resistência Interna Real
        # Prioriza a leitura tratada pelo Filtro de Kalman (EKF) do ESP32 se disponível
        if r_ekf is not None and r_ekf > 0.1:
            r_int_medida = r_ekf
        elif i_real_A > 0.01:
            r_int_medida = (v_oc_ideal - v_real) / i_real_A
        else:
            r_int_medida = r_int_nominal

        # Índice de Anomalia / Degradação (0 = Perfeito, 1 = Falha Crítica)
        desvio_r_int = abs(r_int_medida - r_int_nominal) / r_int_nominal
        score_degradacao = min(1.0, max(0.0, desvio_r_int))

        return {
            "delta_t": delta_t,
            "v_ideal": v_ideal,
            "p_ideal_mW": p_ideal_mW,
            "r_int_nominal": r_int_nominal,
            "r_int_medida": r_int_medida,
            "residuo_tensao": residuo_tensao,
            "eficiencia_relativa": eficiencia_relativa,
            "score_degradacao": score_degradacao,
            "status_saude": "CRÍTICO" if score_degradacao > 0.35 else ("ATENÇÃO" if score_degradacao > 0.15 else "NORMAL")
        }


def main():
    # Inicialização do Cliente InfluxDB com tentativa de reconexão
    client = None
    while client is None:
        try:
            logging.info(f"Conectando ao InfluxDB em {INFLUX_URL}...")
            client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
            query_api = client.query_api()
            write_api = client.write_api(write_options=SYNCHRONOUS)
            logging.info("Conectado ao InfluxDB com sucesso!")
        except Exception as e:
            logging.warning(f"Aguardando InfluxDB inicializar... Erro: {e}")
            client = None
            time.sleep(3)

    engine = TEGDigitalTwinEngine(LUT_FILEPATH)
    logging.info("Engine de Inferência Preditiva (ANSYS Twin) iniciado com sucesso!")

    while True:
        try:
            # Flux Query otimizada com Pivot para trazer todos os campos em uma única estrutura
            query = f'''
            from(bucket: "{INFLUX_BUCKET_RAW}")
              |> range(start: -15s)
              |> filter(fn: (r) => r["_measurement"] == "{INFLUX_MEASUREMENT}")
              |> last()
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            tables = query_api.query(query=query)

            for table in tables:
                for record in table.records:
                    rec = record.values
                    
                    t_q = float(rec.get("t_quente", 0.0))
                    t_f = float(rec.get("t_frio", 0.0))
                    v_r = float(rec.get("tensao_V", 0.0))
                    i_r = float(rec.get("corrente_mA", 0.0))
                    p_r = float(rec.get("potencia_mW", 0.0))
                    r_ekf = float(rec.get("r_interna_ekf", 0.0)) if "r_interna_ekf" in rec else None

                    if t_q > 0.0:
                        # Executar Inferência do Gêmeo Digital
                        metrics = engine.inferir(t_q, t_f, v_r, i_r, p_r, r_ekf)

                        # Gravar Resultados de Analytics no InfluxDB
                        p = Point("teg_analytics") \
                            .tag("device_id", "Bancada_TEG_01") \
                            .tag("status_saude", metrics["status_saude"]) \
                            .field("v_ideal", metrics["v_ideal"]) \
                            .field("p_ideal_mW", metrics["p_ideal_mW"]) \
                            .field("r_int_nominal", metrics["r_int_nominal"]) \
                            .field("r_int_medida", metrics["r_int_medida"]) \
                            .field("residuo_tensao", metrics["residuo_tensao"]) \
                            .field("eficiencia_relativa", metrics["eficiencia_relativa"]) \
                            .field("score_degradacao", metrics["score_degradacao"])

                        write_api.write(bucket=INFLUX_BUCKET_ANALYTICS, record=p)
                        logging.info(
                            f"Twin Processado | ΔT: {metrics['delta_t']:.1f}°C | "
                            f"Eficiência: {metrics['eficiencia_relativa']:.1f}% | "
                            f"R_int: {metrics['r_int_medida']:.3f}Ω | Saúde: {metrics['status_saude']}"
                        )

        except Exception as e:
            logging.error(f"Erro no ciclo de inferência do ANSYS Twin: {e}")

        time.sleep(1)


if __name__ == "__main__":
    main()