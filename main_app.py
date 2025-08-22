import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import constants, signal
import numpy as np
import requests
from PIL import Image, ImageTk
import io
import os
import itertools
from scipy import interpolate
import platform
import subprocess
import pyperclip
import spectrometry_analyzer

# ------------------ PANEL DE IONES (SIDEBAR) -------------------
class IonSidebarPanel(tk.Toplevel):
    def __init__(self, parent, shot_ions_dict, on_update, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title("Control de Iones por Disparo")
        self.transient(parent)
        self.resizable(True, True)
        self.shot_ions_dict = shot_ions_dict
        self.on_update = on_update
        self.ion_vars = {}
        self.scale_vars = {}
        self.cursor_lines = []
        self.last_cursor_x = None

        # ---------- Canvas + Scrollbar -----------
        canvas = tk.Canvas(self, borderwidth=0)
        vscrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscrollbar.set)
        vscrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # --------- El contenido real va aquí ---------
        frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor="nw")

        # --- Manejo de resizing para scroll
        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        frame.bind("<Configure>", _on_frame_configure)

        # --- Permitir scroll con rueda de mouse
        def _on_mouse_wheel(event):
            if event.delta:
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            elif event.num == 5:  # Linux
                canvas.yview_scroll(1, "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
        canvas.bind_all("<MouseWheel>", _on_mouse_wheel)
        canvas.bind_all("<Button-4>", _on_mouse_wheel)
        canvas.bind_all("<Button-5>", _on_mouse_wheel)

        # ---- Construcción del panel igual que antes ----
        title = tk.Label(frame, text="Panel de Iones por Disparo", font=("Arial", 12, "bold"))
        title.grid(row=0, column=0, columnspan=len(shot_ions_dict), sticky="w", pady=3)

        for col, shot in enumerate(shot_ions_dict.keys()):
            lbl = tk.Label(frame, text=f"Shot {shot}", font=("Arial", 10, "bold"))
            lbl.grid(row=1, column=col, sticky="n")

        all_ions = set()
        for ions in shot_ions_dict.values():
            all_ions.update(ion for ion,_,_ in ions)
        all_ions = sorted(list(all_ions))

        for row, ion in enumerate(all_ions, start=2):
            for col, shot in enumerate(shot_ions_dict.keys()):
                f = tk.Frame(frame)
                f.grid(row=row, column=col, sticky="w")
                shot_ions = [ion_tuple for ion_tuple in shot_ions_dict[shot] if ion_tuple[0] == ion]
                if shot_ions:
                    var = tk.BooleanVar(value=True)
                    scale_var = tk.DoubleVar(value=1.0)
                    key = (shot, ion)
                    self.ion_vars[key] = var
                    self.scale_vars[key] = scale_var
                    cb = tk.Checkbutton(f, text=ion, variable=var, command=self.on_update)
                    cb.pack(side=tk.LEFT)
                    tk.Label(f, text=" x ").pack(side=tk.LEFT)
                    entry = tk.Entry(f, width=5, textvariable=scale_var)
                    entry.pack(side=tk.LEFT)
                    entry.bind("<Return>", lambda e: self.on_update())
                    entry.bind("<FocusOut>", lambda e: self.on_update())
                else:
                    tk.Label(f, text="—").pack(side=tk.LEFT)

        tk.Button(frame, text="Actualizar gráfica", command=self.on_update)\
            .grid(row=len(all_ions)+2, column=0, columnspan=len(shot_ions_dict), pady=6)

    def get_active_ions_and_scales(self):
        shots = set(shot for shot, _ in self.ion_vars.keys())
        res = {shot: [] for shot in shots}
        for (shot, ion), var in self.ion_vars.items():
            if var.get():
                scale = self.scale_vars[(shot, ion)].get()
                try:
                    scale = float(scale)
                except Exception:
                    scale = 1.0
                res[shot].append((ion, scale))
        return res


# ------------------ DIALOGO FILTRO S-G -------------------
class FilterConfigDialog(simpledialog.Dialog):
    def __init__(self, parent, viewer):
        self.viewer = viewer
        super().__init__(parent, title="Configure Savitzky-Golay Filter")

    def body(self, master):
        tk.Label(master, text="Window Length (odd int):").grid(row=0, column=0, sticky="e")
        tk.Label(master, text="Polynomial Order (< window):").grid(row=1, column=0, sticky="e")
        self.window_entry = tk.Entry(master)
        self.polyorder_entry = tk.Entry(master)
        self.window_entry.insert(0, str(self.viewer.savgol_window))
        self.polyorder_entry.insert(0, str(self.viewer.savgol_polyorder))
        self.window_entry.grid(row=0, column=1)
        self.polyorder_entry.grid(row=1, column=1)
        return self.window_entry

    def apply(self):
        try:
            window = int(self.window_entry.get())
            polyorder = int(self.polyorder_entry.get())
            if window % 2 == 0 or polyorder >= window:
                raise ValueError
            self.result = (window, polyorder)
        except ValueError:
            messagebox.showerror("Invalid Input", "Ensure window is odd and polyorder < window.")
            self.result = None

# ------------------ MAIN APP VIEWER -------------------
class TokamakDataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("GOLEM Tokamak Data Viewer")
        self.shots = {}
        self.current_shot = None
        self.shot_ions_for_panel = {}
        self.ion_sidebar_panel = None
        self.color_palette = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']

        self.spec_peak_height = 50 
        self.filter_enabled = False
        self.cursor_dynamics_enabled = False
        self.savgol_window = 9
        self.savgol_polyorder = 3
        self.image_refs = []

        plt.rcParams.update({
            'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
            'ytick.labelsize': 8, 'legend.fontsize': 'small',
            'lines.linewidth': 1.2, 'axes.titlesize': 10
        })

        try:
            self.nist_df = spectrometry_analyzer.load_nist("nist_spectral_lines.csv")
            if self.nist_df is None:
                messagebox.showwarning("Advertencia", "No se pudo cargar 'nist_spectral_lines.csv'.\nEl análisis de espectrometría no funcionará.")
        except Exception as e:
            self.nist_df = None
            messagebox.showerror("Error", f"Error al cargar el archivo de NIST: {e}")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.top_button_frame = tk.Frame(self.main_frame)
        self.top_button_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(self.top_button_frame, text="Load Shot", command=self.load_shot).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.top_button_frame, text="Clear Shots", command=self.clear_shots).pack(side=tk.LEFT, padx=5, pady=5)
        self.filter_button = tk.Button(self.top_button_frame, text="Enable Savitzky-Golay", command=self.toggle_filter)
        self.filter_button.pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.top_button_frame, text="Configure Filter", command=self.configure_filter).pack(side=tk.LEFT, padx=5, pady=5)
        self.cursor_toggle_button = tk.Button(self.top_button_frame, text="Enable Cursor Dynamics", command=self.toggle_cursor_dynamics)
        self.cursor_toggle_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.sidebar_button = tk.Button(self.top_button_frame, text="Mostrar Panel de Iones", command=self.show_ion_sidebar)
        self.sidebar_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.plot_frame = tk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # --- Canvas + Toolbar in a dedicated container (Option 2)
        # Create a dedicated container inside plot_frame
        self.canvas_container = tk.Frame(self.plot_frame)
        self.canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create the figure and the embedded canvas
        self.fig, self.axs = plt.subplots(4, 2, figsize=(14, 12))

        # Connect callback to rescale Y automatically when X-limits change (zoom/pan)
        for ax_row in self.axs:
            for ax in ax_row:
                ax.callbacks.connect('xlim_changed', self.on_xlim_changed)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_container)
        canvas_widget = self.canvas.get_tk_widget()

        # 1) Toolbar at the bottom of the container
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_container)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # 2) Canvas above, expandable
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # (Optional) extra label inside the toolbar
        self.data_box_label = tk.Label(self.toolbar, text="", anchor="w", justify="left", font=("Courier New", 8))
        self.data_box_label.pack(side=tk.LEFT, padx=10)

        # (Optional) draw once
        self.canvas.draw()


        self.right_panel = tk.Frame(self.main_frame, bg='white')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.png_frame = tk.Frame(self.right_panel, bg='white')
        self.png_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.png_label = tk.Label(self.png_frame, bg='white')
        self.png_label.pack(expand=True)

        self.cursor_lines = []

    # --------- FUNCIONES PRINCIPALES ---------
    def show_ion_sidebar(self):
        if not self.shot_ions_for_panel:
            messagebox.showwarning("Sin datos", "Carga primero algún disparo para acceder al panel.")
            return
        if self.ion_sidebar_panel and tk.Toplevel.winfo_exists(self.ion_sidebar_panel):
            self.ion_sidebar_panel.lift()
            return
        self.ion_sidebar_panel = IonSidebarPanel(self.root, self.shot_ions_for_panel, self.on_ion_panel_update)

    def configure_filter(self):
        dialog = FilterConfigDialog(self.root, self)
        if dialog.result:
            self.savgol_window, self.savgol_polyorder = dialog.result
            if self.shots: self.plot_data()
    
    def toggle_filter(self):
        self.filter_enabled = not self.filter_enabled
        self.filter_button.config(text="Disable Savitzky-Golay" if self.filter_enabled else "Enable Savitzky-Golay")

        xlims = []
        for ax_row in self.axs:
            row_limits = []
            for ax in ax_row:
                row_limits.append(ax.get_xlim())
            xlims.append(row_limits)

        if self.shots:
           self.plot_data()

        for i, ax_row in enumerate(self.axs):
            for j, ax in enumerate(ax_row):
                x_min, x_max = xlims[i][j]
                ax.set_xlim(x_min, x_max)

                y_vals = []
                for line in ax.get_lines():
                    xdata = np.asarray(line.get_xdata())
                    ydata = np.asarray(line.get_ydata())
                    if len(xdata) != len(ydata) or len(xdata) == 0:
                        continue
                    mask = (xdata >= x_min) & (xdata <= x_max)
                    if np.any(mask):
                        y_vals.extend(ydata[mask])
                if y_vals:
                    y_min = float(np.nanmin(y_vals))
                    y_max = float(np.nanmax(y_vals))
                    if np.isfinite(y_min) and np.isfinite(y_max) and y_max != y_min:
                        ax.set_ylim(y_min * 0.9, y_max * 1.1)

        if getattr(self, "cursor_dynamics_enabled", False) and not getattr(self, "cursor_lines", []):
            fake_event = type('Event', (object,), {
                'inaxes': self.axs[0][0],
                'xdata': self.axs[0][0].get_xlim()[0]
            })()
            self.on_mouse_move(fake_event)

        self.draw_cursor_at(getattr(self, 'last_cursor_x', None))

        self.canvas.draw()


    def apply_filter(self, y):
        if self.filter_enabled and len(y) > self.savgol_window:
            return signal.savgol_filter(y, window_length=self.savgol_window, polyorder=self.savgol_polyorder)
        return y

    def load_shot(self):
        shot_number = simpledialog.askinteger("Input", "Enter the shot number:", parent=self.root)
        if not shot_number: return
        try:
            local_folder = f"shot_{shot_number}"
            os.makedirs(local_folder, exist_ok=True)
            bt_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/Bt.csv", f"{local_folder}/Bt.csv", ['time_ms', 'Bt'])
            ip_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/Ip.csv", f"{local_folder}/Ip.csv", ['time_ms', 'Ip'])
            u_loop_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/U_loop.csv", f"{local_folder}/U_loop.csv", ['time_ms', 'U_loop'])
            ne_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/Interferometry/ne_lav.csv", f"{local_folder}/ne.csv", ['time_ms', 'ne'])
            fast_camera_vertical_data = self.load_fast_camera_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/FastCameras/Camera_Vertical/CameraVerticalPosition", 'vertical_displacement')
            fast_camera_radial_data = self.load_fast_camera_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/FastCameras/Camera_Radial/CameraRadialPosition", 'radial_displacement')

            combined_data = pd.merge(ip_data, u_loop_data, on='time_ms', how='outer')
            combined_data = pd.merge(combined_data, bt_data, on='time_ms', how='outer').interpolate().fillna(0)
            R0, a0, nu = 0.4, 0.085, 2
            combined_data['R'], combined_data['a'] = R0, a0
            combined_data['j_avg_a'] = combined_data['Ip'] * 1e3 / (np.pi * combined_data['a']**2)
            combined_data['j_0'] = combined_data['j_avg_a'] * (nu + 1)
            l_i = np.log(1.65 + 0.89 * nu)
            combined_data['L_p'] = constants.mu_0 * combined_data['R'] * (np.log(8 * combined_data['R'] / combined_data['a']) - 7/4 + l_i / 2)
            dt = np.diff(combined_data['time_ms'].values[:2]).item() if len(combined_data['time_ms'].values) > 1 else 1.0
            n_win = max(5, int(0.5 / dt) + (1 - int(0.5 / dt) % 2)) if dt > 0 else 5
            combined_data['dIp_dt'] = signal.savgol_filter(combined_data['Ip'] * 1e3, n_win, 3, 1, delta=dt * 1e-3)
            combined_data['E_phi'] = (combined_data['U_loop'] - combined_data['L_p'] * combined_data['dIp_dt']) / (2 * np.pi * combined_data['R'])
            combined_data['eta_0'] = combined_data['E_phi'] / combined_data['j_0'].replace(0, np.nan)
            combined_data['eta_avg_a'] = combined_data['E_phi'] / combined_data['j_avg_a'].replace(0, np.nan)
            combined_data['Te_0'] = self.electron_temperature_Spitzer_eV(combined_data['eta_0'], eps=combined_data['a']/combined_data['R'])
            combined_data['Te_avg_a'] = self.electron_temperature_Spitzer_eV(combined_data['eta_avg_a'], eps=combined_data['a']/combined_data['R'])
            te_data = combined_data[['time_ms', 'Te_0', 'Te_avg_a']]
            
            Ip_interp = interpolate.interp1d(ip_data['time_ms'], ip_data['Ip'], bounds_error=False, fill_value=np.nan)(ne_data['time_ms'])
            U_l_interp = interpolate.interp1d(u_loop_data['time_ms'], u_loop_data['U_loop'], bounds_error=False, fill_value=np.nan)(ne_data['time_ms'])
            valid_idx = (ne_data['ne'] > 0) & (Ip_interp > 0) & (U_l_interp > 0)
            tau = (1.0345 * ne_data['ne'][valid_idx]) / (16e19 * Ip_interp[valid_idx]**(1/3) * U_l_interp[valid_idx]**(5/3))
            confinement_time_data = pd.DataFrame({'time_ms': ne_data['time_ms'][valid_idx], 'tau': tau.values})
            h5_file_path = spectrometry_analyzer.download_h5(shot_number)
            ion_labels, wls, intens = [], [], []
            t_spec_0 = self.find_plasma_formation_time(ip_data, threshold=0.01)
            print(f"Ip rises at t={t_spec_0:.2f} ms")
            if h5_file_path and self.nist_df is not None:
                ions, wls, intens = spectrometry_analyzer._detect_main_ions_for_panel(
                    h5_file_path, self.nist_df, peak_height=self.spec_peak_height)
                ion_labels = ions
            self.shot_ions_for_panel[shot_number] = list(zip(ion_labels, wls, intens))
            self.shots[shot_number] = {
                'Bt': bt_data, 'Ip': ip_data, 'U_loop': u_loop_data, 'ne': ne_data,
                'fast_camera_vertical': fast_camera_vertical_data, 'fast_camera_radial': fast_camera_radial_data,
                'Te': te_data, 'confinement_time': confinement_time_data, 'h5_path': h5_file_path,
                'formation_time': t_spec_0
            }
            self.current_shot = shot_number
            self.plot_data()
            self.root.after(100, lambda: self.load_png_image(shot_number, local_folder))
        except Exception as e:
            messagebox.showerror("Error", f"Fallo al cargar el disparo {shot_number}: {e}")

    def on_ion_panel_update(self):
        self.plot_data()

    def on_xlim_changed(self, event):
        """
        When the user zooms/pans (changes X limits), recompute Y limits from
        the currently visible data only. Keep X as-is (scalex=False).
        """
        if getattr(self, "ignore_xlim_callback", False):
            return
        for ax_row in self.axs:
            for ax in ax_row:
                try:
                    ax.relim()
                    ax.autoscale_view(scalex=False, scaley=True)
                except Exception:
                    pass
        self.canvas.draw_idle()

    def plot_data(self):
        for ax_row in self.axs:
            for ax in ax_row:
                ax.clear()
        color_cycle = itertools.cycle(self.color_palette)
        for shot, data in self.shots.items():
            color = next(color_cycle)
            lighter_color = self.lighter_color(color, 1.5)
            self.axs[0, 0].plot(data['Bt']['time_ms'], self.apply_filter(data['Bt']['Bt']), label=f'Bt ({shot})', color=color)
            self.axs[0, 1].plot(data['Ip']['time_ms'], self.apply_filter(data['Ip']['Ip']), label=f'Ip ({shot})', color=color)
            self.axs[1, 0].plot(data['U_loop']['time_ms'], self.apply_filter(data['U_loop']['U_loop']), label=f'U_loop ({shot})', color=color)
            self.axs[1, 1].plot(data['ne']['time_ms'], self.apply_filter(data['ne']['ne']), label=f'ne ({shot})', color=color)
            self.axs[2, 0].plot(data['fast_camera_radial']['time_ms'], self.apply_filter(data['fast_camera_radial']['radial_displacement']), label=f'Δr ({shot})', color=color)
            self.axs[2, 0].plot(data['fast_camera_vertical']['time_ms'], self.apply_filter(data['fast_camera_vertical']['vertical_displacement']), label=f'Δv ({shot})', color=lighter_color)
            self.axs[2, 1].plot(data['Te']['time_ms'], self.apply_filter(data['Te']['Te_0']), label=f'Te_0 ({shot})', color=color)
            self.axs[2, 1].plot(data['Te']['time_ms'], self.apply_filter(data['Te']['Te_avg_a']), label=f'Te_avg_a ({shot})', color=lighter_color, linestyle='--')
            if not data['confinement_time'].empty:
                self.axs[3, 0].plot(data['confinement_time']['time_ms'], self.apply_filter(data['confinement_time']['tau'] * 1e6), label=f'τ_e ({shot})', color=color)
            ax_spec = self.axs[3, 1]
            ions_scales_dict = {}
            if self.ion_sidebar_panel and tk.Toplevel.winfo_exists(self.ion_sidebar_panel):
                user_config = self.ion_sidebar_panel.get_active_ions_and_scales()
                ions_scales_dict = user_config.get(shot, {})
            if ions_scales_dict:
                spectrometry_analyzer.plot_ion_evolution_on_ax(
                    ax=ax_spec,
                    shot_number=shot,
                    shot_color=color,
                    h5_path=data.get('h5_path'),
                    nist_df=self.nist_df,
                    peak_height=self.spec_peak_height,
                    ions_to_plot=[ion for ion, _ in ions_scales_dict],
                    scaling_dict={ion: scale for ion, scale in ions_scales_dict},
                    formation_time=data.get('formation_time', 0.0)
                )
            else:
                ions_this_shot = [ion for ion,_,_ in self.shot_ions_for_panel.get(shot,[])]
                spectrometry_analyzer.plot_ion_evolution_on_ax(
                    ax=ax_spec,
                    shot_number=shot,
                    shot_color=color,
                    h5_path=data.get('h5_path'),
                    nist_df=self.nist_df,
                    peak_height=self.spec_peak_height,
                    ions_to_plot=ions_this_shot,
                    scaling_dict={ion:1.0 for ion in ions_this_shot},
                    formation_time=data.get('formation_time', 0.0)
                )
            ax_spec.relim()
            ax_spec.autoscale(axis="y")
            _, top = ax_spec.get_ylim()
            ax_spec.set_ylim(0, top)
        labels = [['Bt [T]', 'Ip [kA]'], ['U_loop [V]', 'ne [m^-3]'], ['Desplazamiento [mm]', 'Te [eV]'], ['τ_e [μs]', '']]
        for i in range(4):
            for j in range(2):
                ax = self.axs[i, j]
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                if i < 3:
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.set_xlabel('Tiempo [ms]')
                if ax.has_data():
                    if (i, j) == (3, 1):
                        ax.legend(fontsize='x-small', ncol=2)
                    else:
                        ax.legend(loc='best')
                ax.set_ylabel(labels[i][j], labelpad=2)
                if not (i == 0 and j == 0):
                    ax.sharex(self.axs[0, 0])
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()

    # --------- FUNCIONES AUXILIARES DE DATOS Y VISUAL ---------
    def clear_shots(self):
        h5_paths_to_remove = {data['h5_path'] for data in self.shots.values() if data.get('h5_path') and os.path.exists(data['h5_path'])}
        for path in h5_paths_to_remove:
            try: os.remove(path)
            except OSError as e: print(f"Error al eliminar archivo {path}: {e}")
        self.shots = {}
        self.current_shot = None
        self.shot_ions_for_panel = {}
        if self.ion_sidebar_panel and tk.Toplevel.winfo_exists(self.ion_sidebar_panel):
            self.ion_sidebar_panel.destroy()
            self.ion_sidebar_panel = None
        for widget in self.png_frame.winfo_children(): widget.destroy()
        self.image_refs.clear()
        self.plot_data()

    def load_data(self, url, local_path, column_names, sep=','):
        if os.path.exists(local_path):
            return pd.read_csv(local_path, sep=sep)
        response = requests.get(url)
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.text), header=None, names=column_names, sep=sep)
        data.to_csv(local_path, index=False)
        return data

    def load_fast_camera_data(self, url, column_name):
        response = requests.get(url)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        time_ms, values = [], []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                try:
                    time_val, disp_val = float(parts[0]), float(parts[1])
                    time_ms.append(time_val)
                    values.append(disp_val)
                except (ValueError, IndexError):
                    continue
        return pd.DataFrame({'time_ms': time_ms, column_name: values})

    def load_png_image(self, shot_number, local_folder):
        local_full_path = f"{local_folder}/ScreenShotAll_full.png"
        if not os.path.exists(local_full_path):
            png_url = f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/FastCameras/ScreenShotAll.png"
            try:
                response = requests.get(png_url)
                response.raise_for_status()
                with open(local_full_path, 'wb') as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                print(f"No se pudo cargar la imagen para el disparo {shot_number}: {e}")
                return

        for widget in self.png_frame.winfo_children():
            if hasattr(widget, "shot_number") and widget.shot_number == shot_number:
                return

        wrapper_frame = tk.Frame(self.png_frame, bg='white', padx=10)
        wrapper_frame.pack(side=tk.TOP, pady=5)
        wrapper_frame.shot_number = shot_number

        tk.Label(wrapper_frame, text=f"Shot #{shot_number}", bg='white', fg='black', font=("Arial", 10, "bold")).pack(side=tk.TOP)

        image = Image.open(local_full_path)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        img_label = tk.Label(wrapper_frame, image=photo, bg='white')
        img_label.image = photo
        img_label.pack(side=tk.TOP, pady=5)
        self.image_refs.append(photo)
        img_label.bind("<Button-1>", lambda event, path=local_full_path: self.open_in_system_viewer(path))

    def open_in_system_viewer(self, image_path):
        try:
            if not os.path.exists(image_path): raise FileNotFoundError(f"File not found: {image_path}")
            if platform.system() == "Windows":
                os.startfile(image_path)
            elif platform.system() == "Darwin":
                subprocess.run(["open", image_path], check=True)
            else:
                subprocess.run(["xdg-open", image_path], check=True)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image: {e}")

    def electron_temperature_Spitzer_eV(self, eta_measured, Z_eff=3, eps=0, coulomb_logarithm=14):
        if not isinstance(eta_measured, pd.Series) or eta_measured.empty:
            return pd.Series(dtype=float)
        eta_s = eta_measured / Z_eff * (1 - np.sqrt(eps))**2
        term = 1.96 * constants.epsilon_0**2 / (np.sqrt(constants.m_e) * constants.elementary_charge**2 * coulomb_logarithm)
        Te_eV = (term * eta_s)**(-2 / 3) / (constants.elementary_charge * 2 * np.pi)
        return Te_eV.replace([np.inf, -np.inf], np.nan)

    def connect_cursor_events(self):
        self.motion_cid = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.right_click_cid = self.canvas.mpl_connect('button_press_event', self.on_right_click)

    def disconnect_cursor_events(self):
        if hasattr(self, 'motion_cid'):
            self.canvas.mpl_disconnect(self.motion_cid)
            del self.motion_cid
        if hasattr(self, 'right_click_cid'):
            self.canvas.mpl_disconnect(self.right_click_cid)
            del self.right_click_cid

    def toggle_cursor_dynamics(self):
        self.cursor_dynamics_enabled = not self.cursor_dynamics_enabled
        self.cursor_toggle_button.config(
            text="Disable Cursor Dynamics" if self.cursor_dynamics_enabled else "Enable Cursor Dynamics"
        )
        if self.cursor_dynamics_enabled:
            self.connect_cursor_events()
        else:
            for line in getattr(self, 'cursor_lines', []):
                try:
                    line.remove()
                except Exception:
                    pass
            self.cursor_lines.clear()
            if hasattr(self, "data_box_label"):
                self.data_box_label.config(text="")
            self.disconnect_cursor_events()
        self.canvas.draw()

    
    def on_mouse_move(self, event):
        if not event.inaxes or not self.cursor_dynamics_enabled:
            return

        x = event.xdata
        self.last_cursor_x = x 

        for line in list(getattr(self, 'cursor_lines', [])):
            try:
                line.remove()
            except Exception:
                pass
        self.cursor_lines.clear()

        for ax_row in self.axs:
            for ax in ax_row:
                self.cursor_lines.append(ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.8))

        header = "Shot\tTime(ms)\tBt(T)\tIp(kA)\tne(m-3)\tTe_0(eV)\ttau_e(us)"
        data_table = [header]
        for shot, data in self.shots.items():
            vals = {}
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and 'time_ms' in df.columns and not df.empty:
                    idx = (df['time_ms'] - x).abs().idxmin()
                    for col in df.columns:
                        if col != 'time_ms':
                            vals[col] = df.loc[idx, col]
            row = (f"{shot}\t{x:.2f}\t"
                   f"{vals.get('Bt', np.nan):.2f}\t{vals.get('Ip', np.nan):.2f}\t"
                   f"{vals.get('ne', np.nan):.2e}\t{vals.get('Te_0', np.nan):.1f}\t"
                   f"{vals.get('tau', np.nan) * 1e6:.1f}")
            data_table.append(row.replace("nan", "---"))

        if hasattr(self, "data_box_label") and self.data_box_label:
            self.data_box_label.config(text="\n".join(data_table))

        self.canvas.draw_idle()

    def draw_cursor_at(self, x):
        "Redraw the dynamic cursor and data box at a given x (no event needed)."
    
        if x is None or not getattr(self, 'cursor_dynamics_enabled', False):
            return

        for line in list(getattr(self, 'cursor_lines', [])):
            try:
                line.remove()
            except Exception:
                pass
        self.cursor_lines.clear()

        for ax_row in self.axs:
            for ax in ax_row:
                self.cursor_lines.append(ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.8))

        header = "Shot\tTime(ms)\tBt(T)\tIp(kA)\tne(m-3)\tTe_0(eV)\ttau_e(us)"
        data_table = [header]
        for shot, data in self.shots.items():
            vals = {}
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and 'time_ms' in df.columns and not df.empty:
                    idx = (df['time_ms'] - x).abs().idxmin()
                    for col in df.columns:
                        if col != 'time_ms':
                            vals[col] = df.loc[idx, col]
            row = (f"{shot}\t{x:.2f}\t"
                   f"{vals.get('Bt', np.nan):.2f}\t{vals.get('Ip', np.nan):.2f}\t"
                   f"{vals.get('ne', np.nan):.2e}\t{vals.get('Te_0', np.nan):.1f}\t"
                   f"{vals.get('tau', np.nan) * 1e6:.1f}")
            data_table.append(row.replace("nan", "---"))

        if hasattr(self, "data_box_label") and self.data_box_label:
            self.data_box_label.config(text="\n".join(data_table))

        self.canvas.draw_idle()

    def on_right_click(self, event):
        if not event.inaxes or not self.cursor_dynamics_enabled:
           return
        if event.button != 3:
           return

        x = event.xdata
        clipboard_text = "Shot\tTime(ms)\tBt(T)\tIp(kA)\tne(m-3)\tTe_0(eV)\ttau_e(us)\n"
        for shot, data in self.shots.items():
            vals = {}
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and 'time_ms' in df.columns and not df.empty:
                    idx = (df['time_ms'] - x).abs().idxmin()
                    for col in df.columns:
                        if col != 'time_ms':
                            vals[col] = df.loc[idx, col]
            row = (f"{shot}\t{x:.2f}\t"
                   f"{vals.get('Bt', np.nan):.2f}\t{vals.get('Ip', np.nan):.2f}\t"
                   f"{vals.get('ne', np.nan):.2e}\t{vals.get('Te_0', np.nan):.1f}\t"
                   f"{vals.get('tau', np.nan) * 1e6:.1f}\n")
            clipboard_text += row.replace("nan", "")
        pyperclip.copy(clipboard_text)
        messagebox.showinfo("Copiado", "Datos del cursor copiados al portapapeles.")

    @staticmethod
    def lighter_color(color, factor=1.5):
        r, g, b = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    
    def find_plasma_formation_time(self, ip_data, threshold=0.01):
        """
        Find the time when Ip starts to rise significantly.
        threshold: fraction of max Ip to consider as start of plasma formation (5%)
        """
        if ip_data.empty or 'Ip' not in ip_data.columns:
            return 0.0
    
        ip_values = ip_data['Ip'].values
        time_values = ip_data['time_ms'].values
    
        # Find maximum Ip and threshold value (5% of max)
        max_ip = np.max(ip_values)
        threshold_value = max_ip * threshold
    
        # Find the first time Ip exceeds the threshold
        above_threshold = np.where(ip_values > threshold_value)[0]
        if len(above_threshold) > 0:
            t_spec_0 = time_values[above_threshold[0]]
            return t_spec_0
    
        return 0.0  # Fallback if not found

# --------- LAUNCHER ---------
if __name__ == "__main__":
    root = tk.Tk()
    app = TokamakDataViewer(root)
    root.mainloop()
