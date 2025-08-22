import os
import tempfile
import requests
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from scipy.signal import savgol_filter, find_peaks
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from scipy.integrate import trapz

# ——————————————————————————————————————————————
# Parámetros iniciales
# ——————————————————————————————————————————————
SHOT_NUMBER             = 46644
SPECTROMETER_IDENTIFIER = "IRVISUV_0.h5"
SPECTROMETER_URL_FMT    = (
    "http://golem.fjfi.cvut.cz/shots/{shot_no}/"
    "Devices/Radiation/MiniSpectrometer/{identifier}"
)
WL_MIN, WL_MAX          = 400, 900
PEAK_HEIGHT             = 200
PEAK_DISTANCE           = 1
NIST_CSV                = "nist_spectral_lines.csv"
TOLERANCE               = 0.7
PRIORITY                = ['AAA','AA','A','B+','B','C+','C','D+','D','E']
BASELINE_WIN            = 101
BASELINE_POLY           = 3
INITIAL_IDX             = 4
N_BASELINE_FRAMES       = 3
SAVGOL_WINDOW           = 3
SAVGOL_POLY             = 2

# ——————————————————————————————————————————————
# Funciones de utilería
# ——————————————————————————————————————————————
def Gas_identifier(shot_no):
    url = f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Operation/Discharge/X_working_gas_discharge_request'
    try:
        resp = requests.get(url, timeout=10)
        return BeautifulSoup(resp.text, 'html.parser').get_text(strip=True)[:1]
    except requests.RequestException:
        return "?"

def download_h5(shot, identifier, fmt):
    url = fmt.format(shot_no=shot, identifier=identifier)
    r = requests.get(url, timeout=30); r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=identifier)
    tmp.write(r.content); tmp.close()
    return tmp.name

def load_nist(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
    return df.dropna(subset=['Wavelength']).reset_index(drop=True)

def map_peaks(wl_arr, signal, nist_df, tol, peak_height_val):
    idxs, props = find_peaks(signal, height=peak_height_val, distance=PEAK_DISTANCE)
    wls, heights = wl_arr[idxs], props['peak_heights']
    labels = []
    for x in wls:
        sel = nist_df[(nist_df['Wavelength']>=x-tol)&(nist_df['Wavelength']<=x+tol)].copy()
        if sel.empty:
            labels.append("Unknown")
        else:
            sel['rank']  = sel['Acc.'].apply(lambda a: PRIORITY.index(a) if a in PRIORITY else len(PRIORITY))
            sel['delta'] = np.abs(sel['Wavelength']-x)
            best = sel.sort_values(['rank','delta']).iloc[0]
            labels.append(f"{best['Ion']} {best['Wavelength']:.1f} nm")
    return wls, heights, labels

# <-- CAMBIO: Función de integración mejorada
def integrate_peak_local_baseline(spectrum, wavelengths, center_wl, integration_width=5.0):
    """
    Integra un pico sobre una línea de base local.
    
    Args:
        spectrum (np.array): El espectro de intensidades.
        wavelengths (np.array): El array de longitudes de onda.
        center_wl (float): La longitud de onda central del pico a integrar.
        integration_width (float): El ancho total en nm para la integración.

    Returns:
        float: El área integrada neta del pico.
    """
    # 1. Definir la región de interés (ROI) alrededor del pico
    roi_mask = (wavelengths >= center_wl - integration_width / 2) & \
               (wavelengths <= center_wl + integration_width / 2)
    
    roi_wl = wavelengths[roi_mask]
    roi_spec = spectrum[roi_mask]

    if len(roi_wl) < 3:
        return 0.0

    # 2. Definir los "hombros" (endpoints de la ROI)
    start_point = (roi_wl[0], roi_spec[0])
    end_point = (roi_wl[-1], roi_spec[-1])

    # 3. Crear la línea de base local (una línea recta entre los hombros)
    baseline_coeffs = np.polyfit([start_point[0], end_point[0]], [start_point[1], end_point[1]], 1)
    local_baseline = np.polyval(baseline_coeffs, roi_wl)

    # 4. Restar la línea de base local para obtener el área neta
    net_spectrum = np.maximum(roi_spec - local_baseline, 0)
    
    # 5. Integrar el área neta
    return trapz(net_spectrum, x=roi_wl)

def plot_ion_evolution(ion_list, wl_list, shot_number, h5_path, baseline_frames):
    """
    Calcula, suaviza y grafica la evolución temporal de la intensidad de los iones,
    integrando la señal SUAVIZADA (la curva verde del gráfico principal).
    """
    with h5py.File(h5_path, 'r') as f:
        all_wl = f['Wavelengths'][:]
        spectra = f['Spectra'][:]
        time_points = spectra.shape[0]
        
    time_axis_ms = np.arange(time_points) * 2
    
    fig_evo = plt.figure(figsize=(10, 6))
    ax_evo = fig_evo.add_subplot(111)
    has_valid_ions = False

    all_ions_intensities = []

    # Bucle principal para cada línea espectral a seguir
    for ion, center_wl in zip(ion_list, wl_list):
        if ion == "Unknown": continue
        
        raw_integrated_intensities = []
        
        # Bucle para cada instante de tiempo del disparo
        for i in range(time_points):
            spectrum_raw_at_t = spectra[i] # El espectro completo en el tiempo i
            
            # --- Paso 1: Replicar el procesamiento del gráfico principal ---
            # Se calcula el 'residual' (datos crudos - fondo ancho)
            bg = savgol_filter(spectrum_raw_at_t, BASELINE_WIN, BASELINE_POLY)
            residual = np.maximum(spectrum_raw_at_t - bg, 0)
            
            # Se calcula el 'suavizado' (la curva verde)
            smooth_signal = savgol_filter(residual, SAVGOL_WINDOW, SAVGOL_POLY)
            
            # --- Paso 2: Integrar el área bajo la curva 'suavizado' (la verde) ---
            # Se usa la función de línea de base local sobre la señal ya suavizada
            integral = integrate_peak_local_baseline(smooth_signal, all_wl, center_wl)
            raw_integrated_intensities.append(integral)
        
        # Solo procesar si la línea tiene una señal significativa
        if np.max(raw_integrated_intensities) > 0:
            has_valid_ions = True
            
            # --- Paso 3: Suavizar la EVOLUCIÓN TEMPORAL de las integrales ---
            # Esto elimina el ruido entre los puntos de tiempo
            if len(raw_integrated_intensities) > 5:
                 smoothed_evolution = savgol_filter(raw_integrated_intensities, window_length=5, polyorder=2)
            else:
                 smoothed_evolution = raw_integrated_intensities
            
            all_ions_intensities.append({
                "label": f"{ion} (λ≈{center_wl:.1f}nm)",
                "intensities": smoothed_evolution
            })

    # --- Paso 4: Graficar los resultados finales (sin normalizar) ---
    if has_valid_ions:
        for data in all_ions_intensities:
            ax_evo.plot(time_axis_ms, data["intensities"], label=data["label"], linewidth=1.5)
            
        ax_evo.set_xlabel("Time (ms)")
        ax_evo.set_ylabel("Integrated Intensity (A.U.)")
        ax_evo.set_title(f"Smoothed Evolution of Smoothed Spectra - Shot #{shot_number}")
        ax_evo.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_evo.grid(True)
        ax_evo.set_ylim(bottom=0)
    else:
        ax_evo.text(0.5, 0.5, "No valid ions found", ha='center', va='center', transform=ax_evo.transAxes)
        
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

def find_global_max_intensity(h5_path):
    max_intensity = 0
    with h5py.File(h5_path, 'r') as f:
        for spec_raw in f['Spectra']:
            bg = savgol_filter(spec_raw, BASELINE_WIN, BASELINE_POLY)
            residual = np.maximum(spec_raw - bg, 0)
            if residual.max() > max_intensity:
                max_intensity = residual.max()
    return max_intensity * 1.1

# ——————————————————————————————————————————————
# Estado global y recarga
# ——————————————————————————————————————————————
nist = load_nist(NIST_CSV)
h5_path = None
GAS = ""
current_idx = INITIAL_IDX
total_indices = 0
global_ymax = 1000

def reload_data(shot_num):
    global h5_path, GAS, total_indices, current_idx, global_ymax
    if h5_path and os.path.exists(h5_path): os.remove(h5_path)
    h5_path = download_h5(shot_num, SPECTROMETER_IDENTIFIER, SPECTROMETER_URL_FMT)
    GAS = Gas_identifier(shot_num)
    with h5py.File(h5_path, 'r') as f:
        total_indices = f['Spectra'].shape[0]
    current_idx = min(INITIAL_IDX, total_indices - 1)
    global_ymax = find_global_max_intensity(h5_path)

# ——————————————————————————————————————————————
# Construcción de la Interfaz y Callbacks
# (El resto del código no necesita cambios)
# ——————————————————————————————————————————————
plt.rcParams.update({'font.size':9})
fig = plt.figure(figsize=(13,7))
outer = GridSpec(1, 3, width_ratios=[1,4,1], left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.3)

ctrl = outer[0,0].subgridspec(7,1, hspace=1.0) 
ax_shot  = fig.add_subplot(ctrl[0])
ax_gas   = fig.add_subplot(ctrl[1])
ax_xmin  = fig.add_subplot(ctrl[3]) 
ax_xmax  = fig.add_subplot(ctrl[4])
ax_ymax  = fig.add_subplot(ctrl[5])
ax_btn   = fig.add_subplot(ctrl[6])

tb = TextBox(ax_shot, "Shot #", initial=str(SHOT_NUMBER))
ax_gas.axis('off'); gas_text = ax_gas.text(0.5,0.5,"", fontsize=12, weight='bold', ha='center')

nav_grid = ctrl[2].subgridspec(1, 3, wspace=0.1, width_ratios=[1, 2, 1])
ax_prev = fig.add_subplot(nav_grid[0, 0]); prev_btn = Button(ax_prev, '<')
ax_idx_text = fig.add_subplot(nav_grid[0, 1]); ax_idx_text.axis('off'); idx_text = ax_idx_text.text(0.5, 0.5, "", ha='center', va='center')
ax_next = fig.add_subplot(nav_grid[0, 2]); next_btn = Button(ax_next, '>')

slider_xmin = Slider(ax_xmin, "Xmin", WL_MIN, WL_MAX-10, valinit=WL_MIN, valstep=1)
slider_xmax = Slider(ax_xmax, "Xmax", WL_MIN+10, WL_MAX, valinit=WL_MAX, valstep=1)
slider_ymax = Slider(ax_ymax, "Ymax", 10, 1000, valinit=1000, valstep=10)
btn_reset = Button(ax_btn,  "Reset")

ax_main = fig.add_subplot(outer[0,1]); ax_list = fig.add_subplot(outer[0,2]); ax_list.axis('off')
res_line, = ax_main.plot([], [], label='Residual ≥0', lw=1.5, color='C1')
smo_line, = ax_main.plot([], [], '--', label='Suavizado', lw=1.5, color='C2')
res_scatter = ax_main.scatter([], [], marker='x', c='C1', s=30)
smo_scatter = ax_main.scatter([], [], marker='o', s=50, facecolors='none', edgecolors='C2')
text_res = ax_list.text(0,0.95,"", transform=ax_list.transAxes, va='top', color='C1')
text_smo = ax_list.text(0,0.45,"", transform=ax_list.transAxes, va='top', color='C2')
ax_main.set_xlabel("Longitud de onda λ (nm)"); ax_main.set_ylabel("Intensidad (A.U.)")
ax_main.legend(loc='upper right'); ax_main.grid(True)

def update_plot_content(_=None):
    if not h5_path: return
    
    xmin, xmax, ymax = slider_xmin.val, slider_xmax.val, slider_ymax.val
    if xmin >= xmax: return

    with h5py.File(h5_path, 'r') as f:
        wl_full = f['Wavelengths'][:]
        mask = (wl_full >= xmin) & (wl_full <= xmax)
        wl, raw_data = wl_full[mask], f['Spectra'][current_idx][mask]
    
    bg = savgol_filter(raw_data, BASELINE_WIN, BASELINE_POLY)
    residual = np.maximum(raw_data - bg, 0)
    smooth = np.maximum(savgol_filter(residual, SAVGOL_WINDOW, SAVGOL_POLY), 0)

    rwl, rht, rlb = map_peaks(wl, residual, nist, TOLERANCE, PEAK_HEIGHT)
    swl, sht, slb = map_peaks(wl, smooth, nist, TOLERANCE, PEAK_HEIGHT)

    res_line.set_data(wl, residual); smo_line.set_data(wl, smooth)
    res_scatter.set_offsets(np.c_[rwl, rht]); smo_scatter.set_offsets(np.c_[swl, sht])
    text_res.set_text("Picos (Residual):\n" + "\n".join(rlb))
    text_smo.set_text("Picos (Suavizado):\n" + "\n".join(slb))
    
    ax_main.set_xlim(xmin, xmax)
    ax_main.set_ylim(0, ymax)
    
    global current_ions, current_wls
    current_ions = [l.split()[0] for l in slb if l != "Unknown"]
    current_wls = swl[[i for i, l in enumerate(slb) if l != "Unknown"]]
    
    fig.canvas.draw_idle()

def change_index(new_idx):
    global current_idx
    if 0 <= new_idx < total_indices:
        current_idx = new_idx
        idx_text.set_text(f"{current_idx+1}/{total_indices}")
        ax_main.set_title(f"Espectro - Shot #{SHOT_NUMBER} - Índice: {current_idx+1}", pad=10)
        update_plot_content()

def on_shot_submit(text):
    global SHOT_NUMBER, slider_ymax
    try: SHOT_NUMBER = int(text)
    except ValueError: 
        tb.set_val(str(SHOT_NUMBER))
        return

    reload_data(SHOT_NUMBER)
    gas_text.set_text(f"Gas: {GAS}")
    
    ax_ymax.clear()
    slider_ymax = Slider(ax_ymax, "Ymax", 10, global_ymax, valinit=global_ymax, valstep=10)
    slider_ymax.on_changed(update_plot_content)
    
    change_index(INITIAL_IDX)

def reset_view(event):
    slider_xmin.reset()
    slider_xmax.reset()
    slider_ymax.set_val(global_ymax)
    update_plot_content()

prev_btn.on_clicked(lambda e: change_index(current_idx - 1))
next_btn.on_clicked(lambda e: change_index(current_idx + 1))
evo_btn_ax = fig.add_axes([0.9, 0.15, 0.08, 0.05]); evo_btn = Button(evo_btn_ax, 'Evolución')
evo_btn.on_clicked(lambda e: plot_ion_evolution(current_ions, current_wls, SHOT_NUMBER, h5_path, N_BASELINE_FRAMES))

tb.on_submit(on_shot_submit)
slider_xmin.on_changed(update_plot_content)
slider_xmax.on_changed(update_plot_content)
btn_reset.on_clicked(reset_view)

on_shot_submit(str(SHOT_NUMBER))
plt.show()

if h5_path and os.path.exists(h5_path): os.remove(h5_path)