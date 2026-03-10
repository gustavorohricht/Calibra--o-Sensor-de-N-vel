# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
# Altere para o caminho do seu arquivo
caminho_arquivo = 'C:/Users/gusta/Downloads/WT_24C_50__75NOHL_26_0306.xlsx'
nome_aba = 'Nível'
tolerancia = 0.5 # Agora a tolerância é de 0.5 mm (fixa)

# ==========================================
# 2. CARREGAR E TRATAR DADOS
# ==========================================
df = pd.read_excel(caminho_arquivo, sheet_name=nome_aba)

# Função para corrigir valores que o Excel converteu para Data
def corrigir_valor_on(val):
    if pd.isna(val) or val == -1 or str(val) == '-1':
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    try:
        if hasattr(val, 'day') and hasattr(val, 'month'):
            # Lógica: dia 5 e mês 5 -> 5.5; dia 9 e mês 1 -> 9.0
            decimal = val.month / 10.0 if val.month > 1 else 0.0
            return val.day + decimal
    except:
        pass
    return np.nan

# Preparar dados de referência (colunas ml e on)
ref_data = df[['ml', 'on']].dropna().copy()
ref_data['on_ajustado'] = ref_data['on'].apply(corrigir_valor_on)
ref_data = ref_data[ref_data['ml'] > 0] # Ignora o ponto de 0 ml

# ==========================================
# 3. CÁLCULO DAS MÉDIAS REAIS (FAIXA FIXA EM MM)
# ==========================================
niveis_brutos = df['Nível da Água [mm]'].values
resultados = []

for _, row in ref_data.iterrows():
    val_ref = row['on_ajustado']
    
    # MUDANÇA: Tolerância fixa em mm
    limite_inf = val_ref - tolerancia
    limite_sup = val_ref + tolerancia
    
    # Seleciona pontos na faixa absoluta (ex: ref +/- 0.5mm)
    faixa_pontos = niveis_brutos[(niveis_brutos >= limite_inf) & (niveis_brutos <= limite_sup)]
    
    if len(faixa_pontos) > 0:
        media_real = np.mean(faixa_pontos)
        resultados.append({'ml': row['ml'], 'mm_real': media_real})

df_final = pd.DataFrame(resultados)

# ==========================================
# 4. REGRESSÃO LINEAR (ml = f(mm))
# ==========================================
X = df_final['mm_real'].values.reshape(-1, 1)
y = df_final['ml'].values
modelo = LinearRegression().fit(X, y)

# ==========================================
# 5. VISUALIZAÇÃO E RESULTADOS
# ==========================================
plt.figure(figsize=(10, 6))

# Plotar os pontos médios calculados
plt.scatter(df_final['mm_real'], df_final['ml'], color='blue', label='Médias Reais (±0.5mm)')

# Gerar linha de tendência
x_tendencia = np.array([df_final['mm_real'].min(), df_final['mm_real'].max()]).reshape(-1, 1)
y_tendencia = modelo.predict(x_tendencia)
plt.plot(x_tendencia, y_tendencia, color='red', 
         label=f'Regressão: ml = {modelo.coef_[0]:.4f}*mm + ({modelo.intercept_:.4f})\nR²: {modelo.score(X, y):.4f}')

plt.xlabel('Nível da Água [mm]')
plt.ylabel('Volume [ml]')
plt.title('Regressão Linear: Volume em função do Nível (Tolerância Fixa 0.5mm)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

print(f"Equação da Regressão: ml = {modelo.coef_[0]:.4f} * mm + ({modelo.intercept_:.4f})")
print(f"R² (Precisão): {modelo.score(X, y):.4f}")
print("\nTabela de Médias Calculadas:")
print(df_final)