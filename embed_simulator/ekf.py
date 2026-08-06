# ==============================================================================
# 🧠 IMPLEMENTAÇÃO DO EKF (EXTENDED KALMAN FILTER)
# ==============================================================================
class TEGExtendedKalmanFilter:
    """
    Filtro de Kalman Estendido para estimar a resistência interna (R_in) do TEG
    a partir das medições ruidosas de T_quente, T_frio, Corrente e Tensão.
    """
    def __init__(self, seebeck_alpha=0.085, r_init=1.6, q_process=1e-5, r_sensor=0.001):
        self.alpha = seebeck_alpha  # Coeficiente Seebeck do módulo (V/K)
        self.x_rin = r_init         # Estado estimado inicial: R_in (Ohms)
        self.P = 0.1                # Covariância inicial do erro de estimativa
        self.Q = q_process          # Ruído do processo (velocidade da degradação)
        self.R = r_sensor           # Ruído de medição do sensor de tensão

    def update(self, t_quente, t_frio, corrente_mA, v_medido):
        delta_t = max(0.1, t_quente - t_frio)
        i_amp = corrente_mA / 1000.0
        v_oc = self.alpha * delta_t

        # --- 1. PREDIÇÃO ---
        # O estado R_in é assumido constante no curto prazo (R_in[k] = R_in[k-1])
        self.P = self.P + self.Q

        # --- 2. ATUALIZAÇÃO (CORREÇÃO DE KALMAN) ---
        # Tensão prevista com base na física e no estado atual de R_in
        v_predito = v_oc - (i_amp * self.x_rin)

        # Jacobiana H = d(v_predito) / d(R_in)
        H = -i_amp

        # Resíduo da medição (inovação)
        y = v_medido - v_predito

        # Covariância do resíduo (S) e Ganho de Kalman (K)
        S = (H * self.P * H) + self.R
        
        # Proteção contra divisão por zero se corrente for nula
        if S != 0:
            K = (self.P * H) / S
        else:
            K = 0.0

        # Atualização do estado R_in e da covariância P
        self.x_rin = self.x_rin + (K * y)
        self.x_rin = max(0.5, self.x_rin)  # Limite físico para evitar valores irrealistas
        self.P = (1.0 - (K * H)) * self.P

        return self.x_rin