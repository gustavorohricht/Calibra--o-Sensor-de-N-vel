# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
caminho_arquivo = 'C:/Users/gusta/Downloads/WT_32C_40_75NOHL_26_0313.xlsx'
nome_aba = 'Nível'
tolerancia = 0.7   # mm para cima ou para baixo da sua estimativa 'on'
min_pontos = 10     # Mínimo de pontos para considerar um patamar

# ==========================================
# 2. CARREGAR E TRATAR REFERÊNCIAS
# ==========================================
df = pd.read_excel(caminho_arquivo, sheet_name=nome_aba)

def corrigir_valor(val):
    if pd.isna(val) or str(val) == '-1': return np.nan
    try: return float(val)
    except: return np.nan # Caso seja data, você pode reativar a lógica anterior aqui

# Pega a tabela de ml/on/off (independente de onde esteja no tempo)
ref_data = df[['ml', 'on']].dropna(subset=['ml']).copy()
ref_data['on_ref'] = ref_data['on'].apply(corrigir_valor)
ref_data = ref_data[ref_data['ml'] > 0] # Ignora 0 ml

# ==========================================
# 3. BUSCA DO MAIOR PATAMAR ESTÁVEL (LÓGICA NOVA)
# ==========================================
niveis_sensor = df['Nível da Água [mm]'].values
resultados = []

for _, row in ref_data.iterrows():
    vol = row['ml']
    target = row['on_ref']
    
    # Encontrar o maior bloco contínuo dentro da tolerância
    best_start, best_end = 0, 0
    max_len = 0
    
    atual_start = 0
    em_bloco = False
    
    for i in range(len(niveis_sensor)):
        if abs(niveis_sensor[i] - target) <= tolerancia:
            if not em_bloco:
                atual_start = i
                em_bloco = True
        else:
            if em_bloco:
                comprimento = i - atual_start
                if comprimento > max_len:
                    max_len = comprimento
                    best_start, best_end = atual_start, i
                em_bloco = False
    
    # Verifica último bloco se o arquivo acabar em estabilidade
    if em_bloco:
        if (len(niveis_sensor) - atual_start) > max_len:
            max_len = len(niveis_sensor) - atual_start
            best_start, best_end = atual_start, len(niveis_sensor)

    if max_len >= min_pontos:
        media_real = np.mean(niveis_sensor[best_start:best_end])
        resultados.append({'ml': vol, 'mm_real': media_real, 'pontos': max_len})
        print(f"Volume {vol}ml: Encontrado patamar de {max_len} pontos. Média: {media_real:.3f}mm")
    else:
        print(f"Volume {vol}ml: Nenhum patamar estável de pelo menos {min_pts} pontos encontrado!")

df_final = pd.DataFrame(resultados)

# ==========================================
# 4. REGRESSÃO E GRÁFICO
# ==========================================
X = df_final['mm_real'].values.reshape(-1, 1)
y = df_final['ml'].values
modelo = LinearRegression().fit(X, y)

plt.figure(figsize=(10, 6))
plt.scatter(df_final['mm_real'], df_final['ml'], color='blue', label='Médias Estáveis (ON)')
x_plot = np.linspace(df_final['mm_real'].min(), df_final['mm_real'].max(), 100).reshape(-1, 1)
plt.plot(x_plot, modelo.predict(x_plot), color='red', 
         label=f'Regressão: ml = {modelo.coef_[0]:.4f}*mm + ({modelo.intercept_:.4f})\nR²: {modelo.score(X, y):.4f}')

plt.title('Regressão Linear: Volume vs Nível (Busca de Patamar Estável)')
plt.xlabel('Nível Médio Real [mm]')
plt.ylabel('Volume [ml]')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

print(f"\nEquação: ml = {modelo.coef_[0]:.4f} * mm + ({modelo.intercept_:.4f})")