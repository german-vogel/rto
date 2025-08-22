# batch_analyzer.py

import os
import io
import pandas as pd
import numpy as np
import requests
from scipy.signal import savgol_filter, find_peaks
import datetime
import ai_analyzer 
import matplotlib.pyplot as plt

import spectrometry_analyzer as spec

# --- CONFIGURACIÓN ---
SHOTS_TO_ANALYZE = [46644, 46645] 
FLATTOP_THRESHOLD = 0.90 
SPEC_REF_IDX = 4
SPEC_PEAK_HEIGHT = 50

# --- FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS ---

def load_data_from_url(url, column_names, **kwargs):
    """Carga datos desde una URL del servidor GOLEM."""
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text), header=None, names=column_names, **kwargs)
    except requests.exceptions.RequestException as e:
        print(f"Advertencia: No se pudo cargar desde {url}. Error: {e}")
        return pd.DataFrame(columns=column_names)

def get_dominant_ions(shot_number: int, nist_df: pd.DataFrame) -> list:
    """
    Realiza el análisis de espectrometría para un disparo y devuelve los iones dominantes.
    """
    print(f"Iniciando análisis de espectrometría para el disparo #{shot_number}...")
    
    h5_path = None
    try:
        h5_path = spec.download_h5(shot_number)
        if h5_path is None:
            return ["Análisis de espectro fallido (descarga)"]

        with spec.h5py.File(h5_path, 'r') as f:
            all_wl = f['Wavelengths'][:]
            all_spectra = f['Spectra'][:]
            # <-- CAMBIO: Usar la variable de este archivo, no del módulo spec.
            if SPEC_REF_IDX >= len(all_spectra):
                return ["Análisis de espectro fallido (índice fuera de rango)"]
            ref_spectrum_raw = all_spectra[SPEC_REF_IDX]

        bg = savgol_filter(ref_spectrum_raw, window_length=spec.BASELINE_WIN, polyorder=spec.BASELINE_POLY)
        residual = np.maximum(ref_spectrum_raw - bg, 0)
        smooth = np.maximum(savgol_filter(residual, window_length=spec.SMOOTH_WIN, polyorder=spec.SMOOTH_POLY), 0)
        
        mask = (all_wl >= spec.WL_MIN) & (all_wl <= spec.WL_MAX)
        # <-- CAMBIO: Usar la variable de este archivo, no del módulo spec.
        ions, _, intensities = spec._map_peaks(all_wl[mask], smooth[mask], nist_df, SPEC_PEAK_HEIGHT, peak_distance=5)

        if not ions:
            return ["No se detectaron iones significativos"]

        sorted_ions = sorted(zip(ions, intensities), key=lambda x: x[1], reverse=True)
        dominant_ions = [ion for ion, _ in sorted_ions if ion != "Unknown"][:spec.MAX_IONS_TO_PLOT]
        
        return dominant_ions if dominant_ions else ["No se identificaron iones conocidos"]

    except Exception as e:
        print(f"Error durante el análisis de espectrometría para el disparo #{shot_number}: {e}")
        return ["Error en análisis de espectro"]
    finally:
        if h5_path and os.path.exists(h5_path):
            os.remove(h5_path)


def process_shot(shot_number: int, nist_df: pd.DataFrame) -> dict:
    """
    Carga todos los datos de un disparo y extrae un conjunto completo de parámetros físicos.
    """
    print(f"\n--- Procesando Disparo #{shot_number} ---")
    base_url = f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics"
    
    ip_data = load_data_from_url(f"{base_url}/BasicDiagnostics/Results/Ip.csv", ['time_ms', 'Ip'])
    u_loop_data = load_data_from_url(f"{base_url}/BasicDiagnostics/Results/U_loop.csv", ['time_ms', 'U_loop'])
    ne_data = load_data_from_url(f"{base_url}/Interferometry/ne_lav.csv", ['time_ms', 'ne'])
    pos_r_data = load_data_from_url(f"{base_url}/FastCameras/Camera_Radial/CameraRadialPosition", ['time_ms', 'pos_r'], sep=',')
    pos_v_data = load_data_from_url(f"{base_url}/FastCameras/Camera_Vertical/CameraVerticalPosition", ['time_ms', 'pos_v'], sep=',')

    if ip_data.empty:
        return {"shot_number": shot_number, "error": "Datos de corriente (Ip) no disponibles."}
    
    df = pd.merge(ip_data, u_loop_data, on='time_ms', how='outer')
    df = pd.merge(df, ne_data, on='time_ms', how='outer')
    df = pd.merge(df, pos_r_data, on='time_ms', how='outer')
    df = pd.merge(df, pos_v_data, on='time_ms', how='outer')
    df = df.interpolate(method='linear').fillna(0)

    df['P_ohm_kW'] = df['U_loop'] * df['Ip']

    max_ip = df['Ip'].max()
    
    flattop_df = df[df['Ip'] >= max_ip * FLATTOP_THRESHOLD]
    flattop_duration = 0
    if not flattop_df.empty:
        flattop_duration = flattop_df['time_ms'].iloc[-1] - flattop_df['time_ms'].iloc[0]

    features = {
        "shot_number": shot_number,
        "max_ip_kA": max_ip,
        "pulse_duration_ms": df['time_ms'].iloc[-1] if not df.empty else 0,
        "flattop_duration_ms": flattop_duration,
        "max_ne_m3": df['ne'].max(),
        "avg_ne_flattop_m3": flattop_df['ne'].mean() if not flattop_df.empty else 0,
        "avg_U_loop_V": flattop_df['U_loop'].mean() if not flattop_df.empty else 0,
        "peak_P_ohm_kW": df['P_ohm_kW'].max(),
        "avg_P_ohm_flattop_kW": flattop_df['P_ohm_kW'].mean() if not flattop_df.empty else 0,
        "avg_pos_r_mm": flattop_df['pos_r'].mean() if not flattop_df.empty else 0,
        "std_pos_r_mm": flattop_df['pos_r'].std() if not flattop_df.empty else 0,
        "dominant_ions": get_dominant_ions(shot_number, nist_df)
    }
    
    print(f"Características físicas extraídas para #{shot_number}.")
    return features

def format_features_for_prompt(features_list: list) -> str:
    """Convierte la lista de características en una tabla de texto para el prompt."""
    text_summary = ""
    for features in features_list:
        if "error" in features:
            text_summary += f"\n### Disparo #{features['shot_number']}\n- Error: {features['error']}\n"
        else:
            text_summary += (
                f"\n### Disparo #{features['shot_number']}\n"
                f"- **Rendimiento:** Ip_max={features['max_ip_kA']:.1f} kA, Duración_pulso={features['pulse_duration_ms']:.1f} ms, Duración_flattop={features['flattop_duration_ms']:.1f} ms.\n"
                f"- **Parámetros de Plasma (en flattop):** ne_avg={features['avg_ne_flattop_m3']:.2e} m⁻³, U_loop_avg={features['avg_U_loop_V']:.2f} V.\n"
                f"- **Potencia y Estabilidad (en flattop):** P_ohm_avg={features['avg_P_ohm_flattop_kW']:.1f} kW, Posición_radial_avg={features['avg_pos_r_mm']:.2f} mm (std={features['std_pos_r_mm']:.2f}).\n"
                f"- **Composición/Pureza:** Iones dominantes detectados: {', '.join(features['dominant_ions'])}.\n"
            )
    return text_summary

def save_analysis_to_markdown(analysis_text: str, shots: list):
    """Guarda el texto del análisis en un archivo Markdown."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    shot_str = "-".join(map(str, shots))
    filename = f"analisis_comparativo_shots_{shot_str}_{timestamp}.md"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Análisis Comparativo de Descargas del Tokamak GOLEM\n")
            f.write(f"**Descargas Analizadas:** {', '.join(map(str, shots))}\n")
            f.write(f"**Fecha de Análisis:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(analysis_text)
        
        print(f"\nInforme guardado exitosamente como: '{filename}'")
    except Exception as e:
        print(f"\nError al guardar el informe: {e}")

def create_comparison_infographic(features_list: list):
    """Crea y guarda un gráfico comparativo de los parámetros clave."""
    
    valid_features = [f for f in features_list if "error" not in f]
    if len(valid_features) < 1:
        print("No hay datos válidos para generar la infografía.")
        return

    shot_labels = [f"#{f['shot_number']}" for f in valid_features]
    
    params_to_plot = {
        'max_ip_kA': 'Corriente Máxima (Ip) [kA]',
        'flattop_duration_ms': 'Duración Flattop [ms]',
        'avg_ne_flattop_m3': 'Densidad Media (ne) [m⁻³]',
        'std_pos_r_mm': 'Inestabilidad Radial (std) [mm]'
    }
    
    num_params = len(params_to_plot)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    axes = axes.flatten() 
    
    fig.suptitle('Infografía Comparativa de Descargas en Tokamak GOLEM', fontsize=16, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_features)))

    for i, (param_key, title) in enumerate(params_to_plot.items()):
        ax = axes[i]
        values = [f[param_key] for f in valid_features]
        
        bars = ax.bar(shot_labels, values, color=colors)
        ax.set_title(title, fontsize=12, fontweight='medium')
        ax.set_ylabel('Valor', fontsize=10)
        ax.tick_params(axis='x', rotation=0, labelsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for bar in bars:
            yval = bar.get_height()
            if yval > 1000:
                label = f'{yval:.2e}'
            else:
                label = f'{yval:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, label, va='bottom', ha='center', fontsize=9)

    for i in range(num_params, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    shot_str = "-".join(map(str, [f['shot_number'] for f in valid_features]))
    filename = f"infografia_comparativa_shots_{shot_str}_{timestamp}.png"
    
    try:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Infografía guardada exitosamente como: '{filename}'")
    except Exception as e:
        print(f"\nError al guardar la infografía: {e}")
    plt.close(fig)

# --- EJECUCIÓN PRINCIPAL ---

if __name__ == "__main__":
    
    nist_data = spec.load_nist()
    if nist_data is None:
        print("CRÍTICO: No se pudo cargar 'nist_spectral_lines.csv'. El análisis de iones no funcionará.")
        exit()

    all_shots_features = [process_shot(shot, nist_data) for shot in SHOTS_TO_ANALYZE]

    data_summary_text = format_features_for_prompt(all_shots_features)
    
    prompt = f"""
    Eres un físico investigador senior especializado en física de plasmas y fusión por confinamiento magnético, con experiencia en la operación y diagnóstico de tokamaks. 
    Tu tarea es redactar un análisis comparativo de varias descargas experimentales realizadas en el tokamak GOLEM.
    Tu análisis debe ser riguroso, cuantitativo y seguir la estructura de un artículo científico.

    A continuación, se presenta un resumen de los parámetros clave extraídos de cada descarga:
    {data_summary_text}

    Por favor, redacta tu análisis siguiendo estrictamente la siguiente estructura:

    **1. Resumen (Abstract):**
    Presenta en un párrafo conciso el objetivo del análisis, los disparos comparados y el hallazgo principal (ej. "el disparo X exhibió un rendimiento superior...").

    **2. Metodología de Análisis:**
    Describe brevemente cómo se evaluó el rendimiento. Menciona los parámetros clave utilizados para la comparación (ej. "El rendimiento se evaluó en base a la corriente de plasma máxima alcanzada, la duración del pulso, la potencia óhmica de entrada y la estabilidad posicional...").

    **3. Resultados y Discusión Comparativa:**
    Esta es la sección principal. Realiza un análisis comparativo detallado:
    - Compara el rendimiento general de los disparos. ¿Cuál fue el más exitoso y por qué? Utiliza los datos para justificar tu conclusión.
    - Discute las posibles causas físicas de las diferencias. Por ejemplo, ¿cómo pudo la diferencia en la densidad (ne) o en las impurezas (iones) afectar la potencia necesaria (P_ohm) o la estabilidad?
    - Analiza la estabilidad del plasma. ¿La desviación estándar de la posición radial (std_pos_r_mm) se correlaciona con otros parámetros de rendimiento?
    - Formula una hipótesis sobre las condiciones operativas que pudieron ser diferentes entre los disparos (ej. "El mayor voltaje de bucle y la presencia de impurezas metálicas en el disparo Z sugieren una posible interacción plasma-pared más intensa...").

    **4. Conclusión:**
    Resume los hallazgos clave del análisis comparativo y, si es posible, sugiere una dirección para futuros experimentos para validar tus hipótesis (ej. "Se concluye que la optimización del control de densidad y la minimización de impurezas son cruciales... Se recomienda realizar un barrido de gas para...").

    Mantén un tono formal, objetivo y científico en todo el documento.
    """

    final_analysis = ai_analyzer.get_ai_analysis(prompt)

    print("\n\n" + "="*50)
    print("      ANÁLISIS COMPARATIVO DE NIVEL ACADÉMICO")
    print("="*50)
    print(final_analysis)
