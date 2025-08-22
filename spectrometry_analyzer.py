
import os
import tempfile
import requests
import h5py
import pandas as pd
import numpy as np
import itertools
from scipy.signal import savgol_filter, find_peaks
from numpy import trapz

SPECTROMETER_IDENTIFIER = "IRVISUV_0.h5"
SPECTROMETER_URL_FMT    = "http://golem.fjfi.cvut.cz/shots/{shot_no}/Devices/Radiation/MiniSpectrometer/{identifier}"
WL_MIN, WL_MAX          = 400, 900
TOLERANCE               = 0.7
N_BASELINE_FRAMES       = 3
MAX_IONS_TO_PLOT        = 7
BASELINE_WIN            = 101
BASELINE_POLY           = 3
SMOOTH_WIN              = 5
SMOOTH_POLY             = 2
PRIORITY = ['AAA','AA','A','B+','B','C+','C','D+','D','E']

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    rgb_color = tuple(max(0, min(255, int(c))) for c in rgb_color)
    return '#{:02x}{:02x}{:02x}'.format(rgb_color[0], rgb_color[1], rgb_color[2])

def lighten_color(hex_color, amount=0.3):
    try:
        r, g, b = hex_to_rgb(hex_color)
        r = min(255, int(r * (1 + amount)))
        g = min(255, int(g * (1 + amount)))
        b = min(255, int(b * (1 + amount)))
        return rgb_to_hex((r, g, b))
    except Exception as e:
        return hex_color

def download_h5(shot_no):
    url = SPECTROMETER_URL_FMT.format(shot_no=shot_no, identifier=SPECTROMETER_IDENTIFIER)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{shot_no}_{SPECTROMETER_IDENTIFIER}")
        tmp.write(r.content)
        tmp.close()
        return tmp.name
    except requests.exceptions.RequestException as e:
        print(f"Error descargando datos de espectrómetro para el disparo {shot_no}: {e}")
        return None

def load_nist(csv_path=None):
    if csv_path is None:
        base = os.path.dirname(__file__)
        csv_path = os.path.join(base, "nist_spectral_lines.csv")
    else:
        if not os.path.isabs(csv_path):
            base = os.path.dirname(__file__)
            csv_path = os.path.join(base, csv_path)
    try:
        df = pd.read_csv(csv_path, sep=';')
        df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
        return df.dropna(subset=['Wavelength']).reset_index(drop=True)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{csv_path}'.")
        return None
    except Exception as e:
        print(f"Error inesperado al cargar el archivo NIST: {e}")
        return None

def _map_peaks(wl_arr, signal, nist_df, peak_height, peak_distance):
    idxs, _ = find_peaks(signal, height=peak_height, distance=peak_distance)
    if not idxs.any(): return [], [], []
    wls, intensities = wl_arr[idxs], signal[idxs]
    ions, mapped_wls = [], []
    for wl_peak in wls:
        sel = nist_df[(nist_df['Wavelength'] >= wl_peak - TOLERANCE) & (nist_df['Wavelength'] <= wl_peak + TOLERANCE)].copy()
        if not sel.empty:
            sel['rank'] = sel['Acc.'].apply(lambda a: PRIORITY.index(a) if a in PRIORITY else len(PRIORITY))
            sel['delta'] = np.abs(sel['Wavelength'] - wl_peak)
            best = sel.sort_values(['rank', 'delta']).iloc[0]
            ions.append(f"{best['Ion']} ({best['Wavelength']:.1f} Å)")
            mapped_wls.append(best['Wavelength'])
        else:
            ions.append("Unknown")
            mapped_wls.append(wl_peak)
    return ions, mapped_wls, intensities

def _integrate_peak_local_baseline(spectrum, wavelengths, center_wl, integration_width=5.0):
    roi_mask = (wavelengths >= center_wl - integration_width / 2) & (wavelengths <= center_wl + integration_width / 2)
    roi_wl, roi_spec = wavelengths[roi_mask], spectrum[roi_mask]
    if len(roi_wl) < 2: return 0.0
    start_point, end_point = (roi_wl[0], roi_spec[0]), (roi_wl[-1], roi_spec[-1])
    try:
        baseline_coeffs = np.polyfit([start_point[0], end_point[0]], [start_point[1], end_point[1]], 1)
    except np.linalg.LinAlgError:
        baseline_coeffs = (0, start_point[1])
    local_baseline = np.polyval(baseline_coeffs, roi_wl)
    net_spectrum = np.maximum(roi_spec - local_baseline, 0)
    return trapz(net_spectrum, x=roi_wl)

def plot_ion_evolution_on_ax(ax, shot_number, shot_color, h5_path, nist_df, peak_height, 
                           ions_to_plot=None, scaling_dict=None, formation_time=0.0):
    ax.set_xlabel("Tiempo [ms]")
    ax.set_ylabel("Intensidad (A.U.)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if h5_path is None or nist_df is None:
        return
    try:
        import h5py
        from scipy.signal import savgol_filter
        all_wl, all_spectra = None, None
        with h5py.File(h5_path, 'r') as f:
            all_wl, all_spectra = f['Wavelengths'][:], f['Spectra'][:]
        time_points = all_spectra.shape[0]
        total_intensities_per_frame = np.sum(all_spectra, axis=1)
        auto_ref_idx = np.argmax(total_intensities_per_frame) if len(total_intensities_per_frame) > 0 else 0
        ref_spectrum_raw = all_spectra[auto_ref_idx]
        bg_ref = savgol_filter(ref_spectrum_raw, window_length=BASELINE_WIN, polyorder=BASELINE_POLY)
        residual_ref = np.maximum(ref_spectrum_raw - bg_ref, 0)
        smooth_ref = savgol_filter(residual_ref, window_length=SMOOTH_WIN, polyorder=SMOOTH_POLY)
        mask_ref = (all_wl >= WL_MIN) & (all_wl <= WL_MAX)
        ions, wls, intensities_ref = _map_peaks(all_wl[mask_ref], smooth_ref[mask_ref], nist_df, peak_height, peak_distance=5)
        sorted_ions_data = sorted(zip(ions, wls, intensities_ref), key=lambda x: x[2], reverse=True)
        ions_to_plot_data = [item for item in sorted_ions_data if item[0] != "Unknown"][:MAX_IONS_TO_PLOT]
        # NUEVO: Si se entregó una lista personalizada (desde la UI)
        if ions_to_plot is not None and scaling_dict is not None:
            ions_to_plot_data = [item for item in ions_to_plot_data if item[0] in ions_to_plot]
        if not ions_to_plot_data:
            return
        time_axis_ms = np.arange(time_points) * 2.0 + formation_time
        color_shades = [lighten_color(shot_color, amount=i * 0.2) for i in range(len(ions_to_plot_data))]
        for i, (ion_label, center_wl, _) in enumerate(ions_to_plot_data):
            # Si está desactivado, saltar
            if ions_to_plot is not None and scaling_dict is not None:
                if ion_label not in ions_to_plot: continue
                scale_factor = scaling_dict.get(ion_label, 1.0)
            else:
                scale_factor = 1.0
            raw_integrated_intensities = []
            for frame_idx in range(time_points):
                spectrum_at_t = all_spectra[frame_idx]
                bg_t = savgol_filter(spectrum_at_t, BASELINE_WIN, BASELINE_POLY)
                residual_t = np.maximum(spectrum_at_t - bg_t, 0)
                smooth_t = savgol_filter(residual_t, SMOOTH_WIN, SMOOTH_POLY)
                integral = _integrate_peak_local_baseline(smooth_t, all_wl, center_wl)
                raw_integrated_intensities.append(integral * scale_factor)
            smoothed_evolution = savgol_filter(raw_integrated_intensities, window_length=5, polyorder=2) if len(raw_integrated_intensities) > 5 else raw_integrated_intensities
            if np.max(smoothed_evolution) > 0:
                ion_color_shade = color_shades[i % len(color_shades)]
                label_text = f"{ion_label.split(' ')[0]} ({shot_number})"
                ax.plot(time_axis_ms, smoothed_evolution, color=ion_color_shade, linestyle='-', label=label_text, linewidth=1.5)
        ax.legend(fontsize='x-small', ncol=2)
        ax.set_ylim(bottom=0)
    except Exception as e:
        print(f"Error procesando el archivo H5 {h5_path} para shot {shot_number}: {e}")
        import traceback
        traceback.print_exc()

def _detect_main_ions_for_panel(h5_path, nist_df, peak_height=50):
    # Devuelve (ion_labels, wls, intensidades)
    import h5py
    from scipy.signal import savgol_filter
    ions, wls, intens = [], [], []
    with h5py.File(h5_path, 'r') as f:
        all_wl = f['Wavelengths'][:]
        all_spectra = f['Spectra'][:]
        ref_idx = np.argmax(np.sum(all_spectra, axis=1))
        spectrum = all_spectra[ref_idx]
        bg = savgol_filter(spectrum, BASELINE_WIN, BASELINE_POLY)
        residual = np.maximum(spectrum - bg, 0)
        smooth = savgol_filter(residual, SMOOTH_WIN, SMOOTH_POLY)
        mask = (all_wl >= WL_MIN) & (all_wl <= WL_MAX)
        ions, wls, intens = _map_peaks(all_wl[mask], smooth[mask], nist_df, peak_height, peak_distance=5)
        # Solo los conocidos
        ions, wls, intens = zip(*[(i, w, h) for i, w, h in zip(ions, wls, intens) if i != "Unknown"]) if ions else ([],[],[])
    return list(ions), list(wls), list(intens)

