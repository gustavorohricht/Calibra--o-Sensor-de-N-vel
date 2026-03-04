# -*- coding: utf-8 -*-
"""
Created on Fri May 16 12:25:39 2025

@author: gusta
"""

import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import textwrap
from scipy.interpolate import UnivariateSpline



def _detect_transient_start(t, y_smoothed, initial_baseline_points=10, detection_sensitivity_ratio=0.02):
    """
    Helper function to automatically detect the start time of a transient.
    Operates on the smoothed response data.
    """
    if len(y_smoothed) < initial_baseline_points:
        return t.iloc[0]

    initial_baseline_value = y_smoothed.head(initial_baseline_points).mean()
    signal_range = y_smoothed.max() - y_smoothed.min()

    if abs(signal_range) < np.finfo(float).eps:
        return t.iloc[0]

    deviation_threshold = detection_sensitivity_ratio * signal_range

    for i in range(initial_baseline_points, len(y_smoothed)):
        current_y = y_smoothed.iloc[i]
        current_t = t.iloc[i]
        
        if abs(current_y - initial_baseline_value) > deviation_threshold:
            return current_t

    return t.iloc[0]


def CalculateSettlingTime(t, y, tolerance=0.02, absolute_tolerance=0.02, transient_start_time=None,
                          auto_detect_transient_start=True, smoothing_window_size=10,
                          initial_baseline_points=10, detection_sensitivity_ratio=0.02):
    """
    Calculate the settling time from a time series response, with optional smoothing.
    
    Returns:
        tuple: (settling_time_relative, final_value, upper_bound, lower_bound, actual_transient_start_time, y_smoothed)
               Returns np.nan for numerical values and None for y_smoothed if an error occurs.
    """
    if not isinstance(t, pd.Series):
        t = pd.Series(t)
        y = pd.Series(y)

    if len(t) != len(y):
        print("ERROR: Time and response arrays must have the same length.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, None

    if len(t) == 0:
        print("ERROR: Input arrays are empty.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, None
    
    if smoothing_window_size < 1:
        print("ERROR: smoothing_window_size must be at least 1.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, None
    if smoothing_window_size > len(y):
        smoothing_window_size = len(y)
    
    # --- Apply Smoothing ---
    if smoothing_window_size > 1:
        y_smoothed = y.rolling(window=smoothing_window_size, center=True, min_periods=1).mean()
    else:
        y_smoothed = y.copy()

    # Determine the actual transient start time based on parameters
    actual_transient_start_time = t.iloc[0]

    if auto_detect_transient_start:
        actual_transient_start_time = _detect_transient_start(
            t, y_smoothed, initial_baseline_points, detection_sensitivity_ratio
        )
    elif transient_start_time is not None:
        actual_transient_start_time = transient_start_time


    # 1. Determine Final Value (Steady-State Value)
    num_points_for_final = max(10, int(len(y_smoothed) * 0.1))
    if len(y_smoothed) < num_points_for_final:
        num_points_for_final = len(y_smoothed)
    final_value = y_smoothed.tail(num_points_for_final).mean()

    # 2. Define Effective Absolute Tolerance for the Settling Band
    percent_based_abs_tolerance = abs(final_value * tolerance)
    effective_absolute_tolerance = percent_based_abs_tolerance

    if absolute_tolerance is not None and absolute_tolerance >= 0:
        effective_absolute_tolerance = max(percent_based_abs_tolerance, absolute_tolerance)
    elif absolute_tolerance is not None and absolute_tolerance < 0:
        print(f"WARNING: Provided absolute_tolerance ({absolute_tolerance:.4f}) is negative. Ignoring it.")


    if abs(final_value) < np.finfo(float).eps and effective_absolute_tolerance < np.finfo(float).eps:
        default_min_abs_tolerance_for_zero_final = 0.01
        effective_absolute_tolerance = default_min_abs_tolerance_for_zero_final


    upper_bound = final_value + effective_absolute_tolerance
    lower_bound = final_value - effective_absolute_tolerance
    
    # 3. Find Settling Point (Absolute Time)
    settling_time_absolute = t.iloc[0]

    for i in range(len(y_smoothed) - 1, -1, -1):
        current_y_smoothed = y_smoothed.iloc[i]
        current_t = t.iloc[i]

        if not (lower_bound <= current_y_smoothed <= upper_bound):
            if i < len(y_smoothed) - 1:
                settling_time_absolute = t.iloc[i + 1]
            else:
                settling_time_absolute = t.iloc[-1]
            break
    else:
        settling_time_absolute = t.iloc[0]

    # 4. Calculate Settling Time Relative to `actual_transient_start_time`
    settling_time_relative = settling_time_absolute - actual_transient_start_time

    if settling_time_relative < 0:
        settling_time_relative = 0

    return settling_time_relative



def find_steady_state_and_tau(time, response, tolerance=0.5):
    
    time = time - time.iloc[0]
    final_value = np.mean(response[-10:])  # estimate y_infinity
    upper_bound = final_value * (1 + tolerance)
    lower_bound = final_value * (1 - tolerance)

    # Step 1: Find time when system enters and stays in the band
    steady_index = None
    for i in range(len(response)):
        if all(lower_bound <= y <= upper_bound for y in response[i:]):
            steady_index = i
            break
    
    if steady_index is None:
        raise ValueError("System does not reach steady-state within tolerance.")

    steady_time = time[steady_index]

    # Step 2: Estimate tau using a point around 63.2% of final value (1 - 1/e)
    target_value = final_value * (1 - 1/np.e)
    closest_index = np.argmin(np.abs(response - target_value))
    t_tau = time[closest_index]

    # Alternative tau estimation formula
    y_t = response[closest_index]
    tau = -t_tau / np.log(1 - y_t / final_value)

    return steady_time, tau


def SamplingRate(SampleRates):
    
    IndvSample = {}
    forLen = SampleRates.shape
    
    for i in range(forLen[0]-1):
        IndvSample[i] = SampleRates[i+1] - SampleRates[i]
    
    return IndvSample
        
        
def HystesisGraphPlot(x_lineforward,x_linecomingback,forwardWay,comingBackWay,xlab,ylab,legLoc = "best"):

        x_forw = x_lineforward - x_lineforward.iloc[0]
        x_back = x_linecomingback - x_linecomingback.iloc[0]
        
        comingBack = comingBackWay[::-1]
        
      #  dpi = 100
      #  size_in_inches = 324 / dpi
        
       # plt.figure(figsize=(1.5*size_in_inches, size_in_inches))
        plt.figure(figsize=(10, 6))
        plt.scatter(x_forw, forwardWay, label='890-1074 RPM')
        plt.scatter(x_back, comingBack, label='1074-890 RPM')      
        plt.tick_params(axis='both', labelsize=12)  # Aumenta a fonte dos ticks dos eixos X e Y
        plt.legend(loc=legLoc)
       # plt.title("Temperatura de Descarga")
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        #plt.grid(True)
        plt.tight_layout()
    
        return plt.show()
    
def CalculateHysteris(x_going,y_going,x_back,y_back):
    
  intgoing = sp.integrate.simpson(y_going,x_going)
  intback= sp.integrate.simpson(y_back,x_back)

  TotalHyst =  intback - intgoing 
  
  return TotalHyst


def CalculateNormalizedHysteresis(x_going, y_going, x_back, y_back):
    """
    Calculates the normalized hysteresis area using your Simpson-based approach.
    Normalization is done by dividing the raw area by the x-span and y-span.
    """

    TotalHyst = CalculateHysteris(x_going,y_going,x_back,y_back)

    # Calculate x and y span for normalization
    x_all = np.concatenate([x_going, x_back])
    y_all = np.concatenate([y_going, y_back])

    x_range = np.max(x_all) - np.min(x_all)
    y_range = np.max(y_all) - np.min(y_all)

    if x_range == 0 or y_range == 0:
        raise ValueError("x or y range is zero; cannot normalize.")

    normalized_hyst = TotalHyst / (x_range * y_range)

    return normalized_hyst


def plot_normalizedHystVariables(normalized_hysts, variables):
    plt.figure(figsize=(max(8, len(variables) * 0.8), 6))  # Dynamically scale width

    for i, (var, value) in enumerate(zip(variables, normalized_hysts)):
        plt.plot([i, i], [0, value], linewidth=2)               # Uses default color cycle
        plt.scatter(i, value, s=100, edgecolor='black')         # Default marker color

    # Optional: wrap long labels
    wrapped_labels = ['\n'.join(textwrap.wrap(str(v), 15)) for v in variables]

    plt.xticks(range(len(variables)), wrapped_labels, rotation=45, ha='right',fontsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.ylabel(r'$\Delta\tau$ [s]',fontsize=12,labelpad=20)
  #  plt.ylabel(r'Histerese Normalizada [-]',fontsize=12,labelpad=20)
 #   plt.title('Módulo das Histereses Normalizadas')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    

def plot_settlingTimes(settling_times, variables,rotationLabel):
    plt.figure(figsize=(max(8, len(variables) * 0.8), 6))  # Dynamically scale width

    for i, (var, value) in enumerate(zip(variables, settling_times)):
        plt.plot([i, i], [0, value], linewidth=2)               # Uses default color cycle
        plt.scatter(i, value, s=100, edgecolor='black')         # Default marker color

    # Optional: wrap long labels
    wrapped_labels = ['\n'.join(textwrap.wrap(str(v), 15)) for v in variables]

    plt.xticks(range(len(variables)), wrapped_labels, rotation=45, ha='right',fontsize=12)
    plt.tick_params(axis='both', labelsize=12)  
    plt.ylabel('Tempo de Acomodação [s]',fontsize=12,labelpad=20)
    plt.title('Tempos de Acomodação de Diferentes Variáveis - ' + rotationLabel)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()   

def plot_graphs(data_series, legloc="best",xlabel = 'Tempo [s]',ylabel="Temperatura (°C)"):
    """
    Plots multiple time vs. value series, automatically zeroing time axes.
    
    Parameters:
        data_series: list of tuples (time_data, value_data, label)
        legloc: legend location string (e.g. "upper left")
        ylabel: y-axis label
        title: plot title
    """
   # plt.figure(figsize=(10, 6))

    for time_data, value_data, label in data_series:
        if(xlabel == 'Tempo [s]'):
            time_zeroed = time_data - time_data.iloc[0]  # Automatic time alignment
        else:
            time_zeroed = time_data       
        plt.plot(time_zeroed, value_data, label=label)

    plt.legend(loc=legloc, fontsize=12)
    plt.tick_params(axis='both', labelsize=12)  # Aumenta a fonte dos ticks dos eixos X e Y
    plt.xlabel(xlabel,fontsize=12,labelpad=20)
    plt.ylabel(ylabel,fontsize=12,labelpad=20)
    plt.tight_layout()
    plt.show()
    


def plot_scatter_graphs(data_series, legloc="best", xlabel='Tempo [s]', ylabel="Temperatura [°C]", window_size=30, series_with_max_line=None):
    """
    Plota várias séries de tempo vs. valor com uma curva de média móvel e, opcionalmente,
    uma reta vertical no valor máximo de uma série específica.

    Parâmetros:
        data_series: lista de tuplas (time_data, value_data, label)
        legloc: string da localização da legenda (ex: "upper left")
        xlabel: rótulo do eixo x
        ylabel: rótulo do eixo y
        window_size: tamanho da janela para o cálculo da média móvel.
        series_with_max_line: O rótulo (string) da série de dados para a qual
                              a linha vertical no valor máximo deve ser plotada.
                              Se None, nenhuma linha será plotada.
    """

 #   plt.figure(figsize=(10, 6))

    for time_data, value_data, label in data_series:
        if xlabel == 'Tempo [s]':
            time_zeroed = time_data - time_data.iloc[0]  # Alinhamento automático
        else:
            time_zeroed = time_data

        # Plota os pontos
        plt.scatter(time_zeroed, value_data, label=label)

        # SELECIONA a série para a qual a linha vertical deve ser plotada
        if series_with_max_line and label == series_with_max_line:
            # Encontra o tempo do valor máximo nos dados brutos
            max_time = time_zeroed.iloc[value_data.idxmax()]
            
            # Adiciona uma linha vertical no tempo do valor máximo
            plt.axvline(x=max_time, color='red', linestyle='--', linewidth=1.5)#, label=f"Máximo {label}")

        # Média Móvel
        df = pd.DataFrame({'time': time_zeroed, 'value': value_data})
        smoothed_data = df['value'].rolling(window=window_size, center=True).mean().dropna()
        smoothed_time = df['time'].iloc[smoothed_data.index]
        
        # Plota a curva da média móvel
        #plt.plot(smoothed_time, smoothed_data, linewidth=3, linestyle='-', label=f"Média Móvel {label}")
        
    plt.legend(loc=legloc, fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlabel(xlabel, fontsize=12, labelpad=20)
    plt.ylabel(ylabel, fontsize=12, labelpad=20)
    plt.tight_layout()
    plt.show()


#def plot_scatter_graphs(data_series, legloc="best", xlabel='Tempo [s]', ylabel="Temperatura [°C]", window_size=30):
    """
    Plota várias séries de tempo vs. valor com uma curva de média móvel.

    Parâmetros:
        data_series: lista de tuplas (time_data, value_data, label)
        legloc: string da localização da legenda (ex: "upper left")
        xlabel: rótulo do eixo x
        ylabel: rótulo do eixo y
        window_size: tamanho da janela para o cálculo da média móvel.
    """
"""
    plt.figure(figsize=(10, 6))

    for time_data, value_data, label in data_series:
        if xlabel == 'Tempo [s]':
            time_zeroed = time_data - time_data.iloc[0]  # Alinhamento automático
        else:
            time_zeroed = time_data

        # Plota os pontos
        plt.scatter(time_zeroed, value_data, label=label)

        # Média Móvel
        # Combina tempo e valor em um DataFrame para usar a função rolling()
        df = pd.DataFrame({'time': time_zeroed, 'value': value_data})
        
        # Calcula a média móvel
        # O .dropna() remove os valores 'NaN' que surgem no início do cálculo da média
        smoothed_data = df['value'].rolling(window=window_size, center=True).mean().dropna()
        smoothed_time = df['time'].iloc[smoothed_data.index]
        
        # Plota a curva da média móvel
 #       plt.plot(smoothed_time, smoothed_data, linewidth=3, linestyle='-')#, label=f"Média Móvel {label}")
        
    plt.legend(loc=legloc, fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlabel(xlabel, fontsize=12, labelpad=20)
    plt.ylabel(ylabel, fontsize=12, labelpad=20)
    plt.tight_layout()
    plt.show()
"""

def plot_dual_axis_scatter_graphs(
    data_series1, 
    data_series2, 
    legloc="best",
    xlabel='Tempo [s]',
    ylabel1="Temperatura [°C]",
    ylabel2="Pressão [kPa]"
):
    """
    Plots multiple time vs. value series on two separate Y-axes.
    Uses specific hexadecimal colors for plotting.
    """
    # 1. Cria a figura e o primeiro eixo (eixo principal)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 2. Plota os dados da primeira série em AZUL (#1f77b4) no eixo principal (ax1)
    for time_data, value_data, label in data_series1:
        if xlabel == 'Tempo [s]':
            time_zeroed = time_data - time_data.iloc[0]  # Alinhamento automático
        else:
            time_zeroed = time_data
        ax1.scatter(time_zeroed, value_data, label=label)
     #   ax1.plot(time_zeroed, value_data, linewidth=0.8, color='#1f77b4')

    # 3. Configura o eixo principal (ax1)
    ax1.set_xlabel(xlabel, fontsize=12, labelpad=20)
    ax1.set_ylabel(ylabel1, fontsize=12, labelpad=20)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.axis(ymin=0,ymax=3.5)

    # 4. Cria o segundo eixo que compartilha o mesmo eixo X
    ax2 = ax1.twinx()

    # 5. Plota os dados da segunda série em LARANJA (#ff7f0e) no eixo secundário (ax2)
    for time_data, value_data, label in data_series2:
        if xlabel == 'Tempo [s]':
            time_zeroed = time_data - time_data.iloc[0] # Alinhamento automático
        else:
            time_zeroed = time_data
        ax2.scatter(time_zeroed, value_data, label=label)
        ax2.axis(ymin=0,ymax=0.5)
       # ax2.plot(time_zeroed, value_data, linewidth=0.8, color='#ff7f0e')

    # 6. Configura o segundo eixo (ax2)
    ax2.set_ylabel(ylabel2, fontsize=12, labelpad=20)
    ax2.tick_params(axis='y', labelsize=12)

    # 7. Combina as legendas de ambos os eixos
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=legloc, fontsize=12)

    # 8. Ajustes finais e visualização
    plt.tight_layout()
    plt.show()

#def plot_scatter_graphs(data_series, legloc="best",xlabel = 'Tempo [s]' ,ylabel="Temperatura [°C]"):
"""
    Plots multiple time vs. value series, automatically zeroing time axes.
    
    Parameters:
        data_series: list of tuples (time_data, value_data, label)
        legloc: legend location string (e.g. "upper left")
        ylabel: y-axis label
        title: plot title
    """
"""
    plt.figure(figsize=(10, 6))

    for time_data, value_data, label in data_series:
        if(xlabel == 'Tempo [s]'):
            time_zeroed = time_data - time_data.iloc[0]  # Automatic time alignment
        else:
            time_zeroed = time_data
        plt.scatter(time_zeroed, value_data, label=label)
        plt.plot(time_zeroed, value_data, linewidth=0.8)
    plt.legend(loc=legloc, fontsize=12)
    plt.tick_params(axis='both', labelsize=12)  
    plt.xlabel(xlabel,fontsize=12,labelpad=20)
    plt.ylabel(ylabel,fontsize=12,labelpad=20)
    plt.tight_layout()
    plt.show()
"""





# --- New plotting function for detailed settling time visualization ---
"""
def plot_scatter_graphs(data_series, legloc="best", xlabel='Tempo [s]', ylabel="Temperatura [°C]", smoothing_factor=3.0):

    plt.figure(figsize=(10, 6)) # Create a new figure for each plot
   
    
    for time_data, value_data, label in data_series:
        if(xlabel == 'Tempo [s]'):
            time_zeroed = time_data - time_data.iloc[0]  # Automatic time alignment
        else:
            time_zeroed = time_data       
        plt.plot(time_zeroed, value_data, label=label)
        
    plt.plot(time_zeroed, raw_response, label='Raw Data', alpha=0.6, color='skyblue', linewidth=1)
    
    if smoothing_window_size > 1:
        plt.plot(time_zeroed, y_smoothed, label=f'Smoothed Data (Window={smoothing_window_size})', color='orange', linewidth=2)
    

   # plt.title(f'Análise de Tempo de Assentamento para {variable_name}\nSobressinal: {percent_overshoot:.2f}%', fontsize=16)
    plt.xlabel('Tempo [s]', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
 #   plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust plot to prevent labels overlapping
    plt.show() # Display the plot
"""
"""    
def plot_settling_time_details(time_data, raw_response, y_smoothed, final_value,
                           upper_bound, lower_bound, actual_transient_start_time, initial_value, # Added initial_value
                           settling_time_relative, variable_name, ylabel,
                           smoothing_window_size, absolute_tolerance,
                           peak_value, peak_time, percent_overshoot): # Added peak info

    plt.figure(figsize=(12, 7)) # Create a new figure for each plot
    
    plt.plot(time_data, raw_response, label='Raw Data', alpha=0.6, color='skyblue', linewidth=1)
    
    if smoothing_window_size > 1:
        plt.plot(time_data, y_smoothed, label=f'Smoothed Data (Window={smoothing_window_size})', color='orange', linewidth=2)
    
    # Plot horizontal lines for initial, final value and tolerance band
    plt.axhline(y=initial_value, color='darkgray', linestyle=':', label=f'Initial Value ({initial_value:.2f})') # New: Initial Value
    plt.axhline(y=final_value, color='red', linestyle='--', label=f'Final Value ({final_value:.2f})')
    plt.axhline(y=upper_bound, color='green', linestyle=':', label=f'Upper Bound ({upper_bound:.2f})')
    plt.axhline(y=lower_bound, color='green', linestyle=':', label=f'Lower Bound ({lower_bound:.2f})')
    
    # Plot vertical line for transient start time
    plt.axvline(x=actual_transient_start_time, color='purple', linestyle='-.', label=f'Transient Start ({actual_transient_start_time:.2f})')
    
    # Calculate absolute settling time marker for plotting
    settling_time_absolute_marker = actual_transient_start_time + settling_time_relative
    # Plot vertical line for settling time
    plt.axvline(x=settling_time_absolute_marker, color='blue', linestyle='-', label=f'Settling Time ({settling_time_relative:.2f}s relative)')
    
    # --- Plot Peak Value ---
    if not np.isnan(peak_value):
        # Determine if it was an overshoot or undershoot for label clarity
        if final_value is not None and initial_value is not None:
            if final_value > initial_value: # Upward step
                if peak_value > final_value: # True overshoot
                    label_text = f'Max Peak ({peak_value:.2f})'
                    color = 'red'
                else: # No overshoot, or undershoot on upward step
                    label_text = f'Peak ({peak_value:.2f})'
                    color = 'gray'
            else: # Downward step
                if peak_value < final_value: # True undershoot
                    label_text = f'Min Peak ({peak_value:.2f})'
                    color = 'red'
                else: # No undershoot, or overshoot on downward step
                    label_text = f'Peak ({peak_value:.2f})'
                    color = 'gray'
        else: # Final or initial value unknown or problematic, just plot the peak
            label_text = f'Peak Value ({peak_value:.2f})'
            color = 'red'
    
        plt.scatter([peak_time], [peak_value], color=color, marker='o', s=100, zorder=5, label=label_text)
        
        # Add overshoot percentage to plot title or as annotation
        if not np.isnan(percent_overshoot) and percent_overshoot != 0.0:
            if percent_overshoot != np.inf:
                # Adjust text position based on overshoot direction
                va = 'bottom' if final_value > initial_value else 'top'
                plt.text(peak_time, peak_value, f'{percent_overshoot:.2f}%',
                         fontsize=10, ha='center', va=va,
                         color='red')
    
    
    plt.title(f'Análise de Tempo de Assentamento para {variable_name}\nSobressinal: {percent_overshoot:.2f}%', fontsize=16)
    plt.xlabel('Tempo [s]', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust plot to preve
   """