import numpy as np
import pandas as pd
import os

def generate_teg_dataset(samples=10000, output_file="teg_dataset.csv"):
    np.random.seed(42)
    
    # 1. Variáveis de entrada físicas (Temperaturas)
    t_quente = np.random.uniform(40.0, 200.0, samples) # °C
    t_frio = np.random.uniform(15.0, 35.0, samples)    # °C
    delta_t = np.maximum(5.0, t_quente - t_frio)
    
    # 2. Comportamento ANSYS (Modelo Ideal)
    v_oc_ideal = 0.05 * delta_t                         # Tensão de circuito aberto ideal (V)
    r_int_nominal = 2.0 + 0.004 * delta_t               # Resistência interna nominal (Ohms)
    i_mA = np.random.uniform(10.0, 600.0, samples)      # Corrente em mA
    i_A = i_mA / 1000.0
    
    v_ideal = np.maximum(0.0, v_oc_ideal - (i_A * r_int_nominal))
    p_ideal_mW = v_ideal * i_mA
    
    # 3. Degradação do TEG (Alvo / Target do Modelo)
    # Score 0.0 = Perfeito Estado | Score 1.0 = Falha Grave (Trinca Interna)
    score_degradacao = np.random.uniform(0.0, 1.0, samples)
    
    # Degradação aumenta a resistência interna real em até 250%
    r_int_real = r_int_nominal * (1.0 + 2.5 * score_degradacao)
    
    # Adiciona ruído de medição dos sensores
    ruido_v = np.random.normal(0, 0.015, samples)
    v_real = np.maximum(0.0, v_oc_ideal - (i_A * r_int_real) + ruido_v)
    p_real_mW = v_real * i_mA
    
    # Eficiência Relativa (%)
    eficiencia_relativa = np.clip((p_real_mW / (p_ideal_mW + 1e-6)) * 100.0, 0.0, 100.0)
    
    df = pd.DataFrame({
        "t_quente": t_quente,
        "t_frio": t_frio,
        "v_real": v_real,
        "v_ideal": v_ideal,
        "p_real_mW": p_real_mW,
        "p_ideal_mW": p_ideal_mW,
        "score_degradacao": score_degradacao,
        "eficiencia_relativa": eficiencia_relativa / 100.0 # Normalizado [0, 1]
    })
    
    df.to_csv(output_file, index=False)
    print(f"📊 Dataset sintético com {samples} amostras gerado em '{output_file}'.")

if __name__ == "__main__":
    generate_teg_dataset()