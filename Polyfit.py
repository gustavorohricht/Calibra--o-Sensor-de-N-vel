# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

file_path = "C:/Users/gusta/Downloads/WT_24C_60_65NOHL_26_0401_300ml (1).xlsx"
start_idx = 250
end_idx = 2115
deg = 1  

df_full = pd.read_excel(file_path, sheet_name='Dados')
df_data = df_full.iloc[start_idx:end_idx].copy().reset_index(drop=True)


w_level_raw = df_data['Volume'].values
time = df_data["Tempo Corrido [s]"].values

polyfit_coeffs = np.polyfit(time, w_level_raw, deg=deg)
w_level_polyfit = np.polyval(polyfit_coeffs, time)
derivative_coeffs = np.polyder(polyfit_coeffs)
volume_rate = np.polyval(derivative_coeffs, time) * 3600
volume_rate_module = np.abs(volume_rate)

ss_res = np.sum((w_level_raw - w_level_polyfit) ** 2)
ss_tot = np.sum((w_level_raw - np.mean(w_level_raw)) ** 2)
r2 = 1 - (ss_res / ss_tot) if not np.isclose(ss_tot, 0) else 0.0

terms = []
poly_degree = len(polyfit_coeffs) - 1
for i, coef in enumerate(polyfit_coeffs):
	power = poly_degree - i
	if np.isclose(coef, 0):
		continue

	coef_abs = abs(coef)
	if power == 0:
		term = f"{coef_abs:.4g}"
	elif power == 1:
		term = f"{coef_abs:.4g}t"
	else:
		term = f"{coef_abs:.4g}t^{power}"

	sign = "+" if coef >= 0 else "-"
	terms.append((sign, term))

if terms:
	first_sign, first_term = terms[0]
	equation = ("- " if first_sign == "-" else "") + first_term
	for sign, term in terms[1:]:
		equation += f" {sign} {term}"
	poly_eq_label = f"y = {equation}"
else:
	poly_eq_label = "y = 0"

derivative_terms = []
derivative_degree = len(derivative_coeffs) - 1
for i, coef in enumerate(derivative_coeffs):
	power = derivative_degree - i
	if np.isclose(coef, 0):
		continue

	coef_abs = abs(coef)
	if power == 0:
		term = f"{coef_abs:.4g}"
	elif power == 1:
		term = f"{coef_abs:.4g}t"
	else:
		term = f"{coef_abs:.4g}t^{power}"

	sign = "+" if coef >= 0 else "-"
	derivative_terms.append((sign, term))

if derivative_terms:
	first_sign, first_term = derivative_terms[0]
	derivative_equation = ("- " if first_sign == "-" else "") + first_term
	for sign, term in derivative_terms[1:]:
		derivative_equation += f" {sign} {term}"
	derivative_eq_label = f"dy/dt = {derivative_equation}"
else:
	derivative_eq_label = "dy/dt = 0"

plt.plot(time, w_level_raw, label='Volume (Raw)', color='blue', alpha=0.6)
plt.plot(time, w_level_polyfit, label=f'Volume (Polyfit grau {deg}) | {poly_eq_label} | R² = {r2:.4f}', color='red', linestyle='--')
plt.xlabel('Tempo Corrido [s]')
plt.ylabel('Volume [mL]')
plt.title('Volume ao Longo do Tempo com Polyfit')
plt.legend()
plt.grid()
plt.show()

plt.plot(time, volume_rate_module, label=f'{derivative_eq_label} × 3600|', color='green')
plt.xlabel('Tempo Corrido [s]')
plt.ylabel('Módulo da taxa de variação [mL/h]')
plt.title('Módulo da Taxa de Variação do Volume (Derivada do Polyfit)')
#plt.legend()
plt.grid()
plt.show()

# %%