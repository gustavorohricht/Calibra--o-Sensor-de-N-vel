# %%
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
file_path = "C:/Users/gusta/Downloads/WT_32C_40_73HL_26_0407.xlsx"
start_idx = 217
end_idx = 851
level_col = 'Nível da Água [mm]'
time_col = 'Tempo Corrido [s]'
subset = [level_col]
moving_avg_window = 40
flow_window = 20 
Q_val = 0.001  # Ruído de processo para 2ª ordem (menor = mais estável)

# Ajuste esta equação conforme sua calibração: V = a * nível + b
volume_eq = lambda mm: 78.4206 * mm + (-142.6356)

# ==========================================
# 2. CARGA E FILTRAGEM (KALMAN 2ª ORDEM ADAPTATIVO)
# ==========================================
df_full = pd.read_excel(file_path, sheet_name='Dados')
df_data = df_full.iloc[start_idx:end_idx].copy().reset_index(drop=True)

# Limpeza de NaNs e remoção de valores negativos do nível
df_data = df_data.dropna(subset=subset + [time_col])
df_data = df_data[df_data[level_col] >= 0].copy().reset_index(drop=True)

if df_data.empty:
    raise ValueError('Nenhum dado válido permaneceu após remover valores negativos e NaNs.')

# Suavização do nível antes de converter para volume
df_data['Nível_MA_40'] = (
    df_data[level_col]
    .rolling(window=moving_avg_window, min_periods=1)
    .mean()
)

df_data['Volume'] = df_data['Nível_MA_40'].apply(volume_eq)

v_raw = df_data['Volume'].to_numpy(dtype=float)
tempo = df_data[time_col].to_numpy(dtype=float)

def apply_adaptive_kalman_2d(time, volume, Q_param):
    n = len(volume)
    dt = np.mean(np.diff(time))
    
    # Estados: [Volume, Taxa (ml/s)]
    x = np.array([[volume[0]], [0.0]])
    P = np.eye(2) * 10
    F = np.array([[1, -dt], [0, 1]]) # Modelo físico: V = V0 - Taxa*dt
    H = np.array([[1, 0]])          # Medimos apenas o Volume
    Q = np.array([[Q_param, 0], [0, Q_param]])
    
    # R dinâmico (Variância Móvel) - Correção para Pandas 2.0
    R_dynamic = pd.Series(volume).rolling(window=20, center=True).var().bfill().ffill().values
    
    estimates = []
    for k in range(n):
        # 1. Predição
        x = F @ x
        P = F @ P @ F.T + Q
        
        # 2. Atualização Adaptativa
        curr_R = R_dynamic[k] if not np.isnan(R_dynamic[k]) else 1.0
        curr_R = max(curr_R, 0.1)
        
        S = H @ P @ H.T + curr_R
        K = P @ H.T / S
        
        y = volume[k] - (H @ x) # Inovação
        x = x + K * y
        P = (np.eye(2) - K @ H) @ P
        
        estimates.append(x[0,0])
    return np.array(estimates)

# Aplicação do Filtro
df_data['Volume_KF'] = apply_adaptive_kalman_2d(tempo, v_raw, Q_val)

# ==========================================
# 3. REGRESSÕES LINEARES E Taxa
# ==========================================
slope_raw, int_raw, r_raw, p_raw, std_err_raw = stats.linregress(tempo, v_raw)
slope_kf, int_kf, r_kf, p_kf, std_err_kf = stats.linregress(tempo, df_data['Volume_KF'])

vazao_bruta_slope = abs(slope_raw) * 3600
vazao_kalman_slope = abs(slope_kf) * 3600

# Taxa Instantânea (Janelada)
flow_rates = []
for i in range(len(df_data) - flow_window):
    v_s, v_e = df_data['Volume_KF'].iloc[i], df_data['Volume_KF'].iloc[i+flow_window]
    t_s, t_e = tempo[i], tempo[i+flow_window]
    flow = ((v_s - v_e) / (t_e - t_s)) * 3600
    flow_rates.append(flow)

# ==========================================
# 4. CÁLCULOS TÉRMICOS CONSOLIDADOS
# ==========================================
def get_basic_stats(col_name):
    if col_name in df_data.columns:
        return df_data[col_name].mean(), df_data[col_name].std()
    return np.nan, np.nan

# Entrada Consolidada (Média Global 1 & 2)
avg_in1, _ = get_basic_stats('Ar_UC_In_1 [°C]')
avg_in2, _ = get_basic_stats('Ar_UC_In_2 [°C]')
avg_Ar_In_Global = (avg_in1 + avg_in2) / 2
std_Ar_In_Global = np.sqrt((df_data['Ar_UC_In_1 [°C]'].var() + df_data['Ar_UC_In_2 [°C]'].var()) / 2)

# Coleta dos demais dados
params = {
    "Temp RH Entrada": get_basic_stats('Temperatura RH In [°C]'),
    "Umidade Rel. Entrada": get_basic_stats('Umidade Relativa In [%]'),
    "Ar UC Saída 1": get_basic_stats('Ar_UC_Out_1 [°C]'),
    "Ar UC Saída 2": get_basic_stats('Ar_UC_Out_2 [°C]'),
    "Temp RH Saída": get_basic_stats('Temperatura RH Out [°C]'),
    "Umidade Rel. Saída": get_basic_stats('Umidade Relativa Out [%]'),
    "Vazão de Ar (m3/h)": get_basic_stats('Vazão [m3/h]'),
    "UC Entrada": get_basic_stats('UC_In [°C]'),
    "UC HL Saída (UC_HL_Out)": get_basic_stats('UC_HL_Out [°C]'),
    "UC Saída": get_basic_stats('UC_Out [°C]'),
    "Água T1": get_basic_stats('H20_1 [°C]'),
    "Água T2": get_basic_stats('H20_2 [°C]'),
    "Água T3": get_basic_stats('H20_3 [°C]'),
    "Água T4": get_basic_stats('H20_4 [°C]')
}

# ==========================================
# 5. IMPRESSÃO DA TABELA E RESUMO
# ==========================================
print("\n" + "="*75)
print(f"{'PARÂMETRO':<45} | {'MÉDIA':<12} | {'DESVIO PADRÃO':<12}")
print("-" * 75)
print(f"{'Média Ar UC Entrada (Global 1 & 2)':<45} | {avg_Ar_In_Global:<12.2f} | {std_Ar_In_Global:<12.2f}")
for label, (m, s) in params.items():
    print(f"{label:<45} | {m:<12.2f} | {s:<12.2f}")
print("="*75)

print(f"Taxa BRUTA: {vazao_bruta_slope:.2f} ml/h | Desvio Padrão: {std_err_raw*3600:.4f} ml/h")
print(f"Taxa KALMAN: {vazao_kalman_slope:.2f} ml/h | Desvio Padrão: {std_err_kf*3600:.4f} ml/h")
print(f"R² Kalman: {r_kf**2:.6f}")
print("="*75)

# ==========================================
# 6. GRÁFICOS SEPARADOS
# ==========================================

# Gráfico 1: Regressão Bruta
plt.figure(figsize=(10, 5))
plt.scatter(tempo, v_raw, color='gray', alpha=0.3, s=10, label='Volume Calculado')
plt.plot(tempo, slope_raw * tempo + int_raw, color='red', 
         label=f'Regressão Bruta: y={slope_raw:.4f}x + {int_raw:.2f}\n$R^2={r_raw**2:.4f}$')
plt.title('Análise: Volume Calculado a partir do Nível')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# Gráfico 2: Regressão Kalman 2ª Ordem
plt.figure(figsize=(10, 5))
plt.scatter(tempo, v_raw, color='gray', alpha=0.15, s=10, label='Volume Calculado')
plt.plot(tempo, df_data['Volume_KF'], color='green', label='Kalman 2ª Ordem (Filtrado)')
plt.plot(tempo, slope_kf * tempo + int_kf, color='blue', linestyle='--',
          label=f'Regressão Kalman: y={slope_kf:.4f}x + {int_kf:.2f}\n$R^2={r_kf**2:.4f}$')
plt.title('Análise: Filtro de Kalman Adaptativo no Volume')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# Gráfico 3: Estabilidade da Taxa
plt.figure(figsize=(10, 5))
plt.plot(tempo[:-flow_window], flow_rates, color='purple', alpha=0.7, label='Taxa Instantânea')
plt.axhline(y=vazao_kalman_slope, color='blue', linestyle='--', label=f'Média: {vazao_kalman_slope:.2f} ml/h')
plt.title('Taxa Instantânea vs Média (ml/h)')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

plt.figure(figsize=(10, 5))
#plt.plot(tempo, df_data['Vazão [m3/h]'], alpha=0.7,label='Vazão [m3/h]')
#plt.plot(tempo, df_data['UC_In [°C]'], alpha=0.7,label='UC_In')
plt.plot(tempo, df_data['H20_1 [°C]'], alpha=0.7,label='H20_1')
plt.plot(tempo, df_data['H20_2 [°C]'], alpha=0.7,label='H20_2')
plt.plot(tempo, df_data['H20_3 [°C]'], alpha=0.7,label='H20_3')
#plt.plot(tempo, df_data['H20_4 [°C]'], alpha=0.7,label='H20_4')
plt.title('H20')
plt.xlabel('Tempo [s]')
plt.ylabel('Vazão [°C]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%