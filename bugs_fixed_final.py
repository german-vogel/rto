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
import pyperclip
import os
import itertools
from scipy import interpolate # Explicitly added for confinement time calculation

class TokamakDataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("GOLEM Tokamak Data Viewer")
        self.shots = {}
        self.current_shot = None
        self.base_dir = os.path.dirname(os.path.abspath(__file__))


        # Define filter flag
        self.filter_enabled = False  # Initially no filter
        self.ignore_xlim_callback = False  # Flag to suppress autoscale override
        
        # Define the custom color palette
        self.color_palette = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']
        self.color_cycle = itertools.cycle(self.color_palette)  # Cycle through colors
    
        # Create a main frame to hold everything
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for top-level buttons (Load Shot, Clear Shots, Cursor Dynamics)
        self.top_button_frame = tk.Frame(self.main_frame)
        self.top_button_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Add buttons to the top button frame
        self.load_button = tk.Button(self.top_button_frame, text="Load Shot", command=self.load_shot)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.clear_button = tk.Button(self.top_button_frame, text="Clear Shots", command=self.clear_shots)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.cursor_toggle_button = tk.Button(self.top_button_frame, text="Enable Cursor Dynamics", command=self.toggle_cursor_dynamics)
        self.cursor_toggle_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.cursor_dynamics_enabled = False

        # Add Savitzky-Golay filter toggle button
        self.filter_button = tk.Button(self.top_button_frame, text="Enable Savitzky-Golay Filter", command=self.toggle_filter)
        self.filter_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.filter_config_button = tk.Button(self.top_button_frame, text="Configure Filter", command=self.configure_filter)
        self.filter_config_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Create a frame for the plot on the left
        self.plot_frame = tk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a single container for both canvas and toolbar
        self.canvas_container = tk.Frame(self.plot_frame)
        self.canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create the matplotlib figure and subplots
        self.fig, self.axs = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
        self.fig.subplots_adjust(left=0.07, right=0.96, top=0.95, bottom=0.09, hspace=0.25, wspace=0.15)

        # Embed the canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_container)
        self.canvas_widget = self.canvas.get_tk_widget()

        # Create and pack toolbar below canvas
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_container)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Now pack the canvas above the toolbar
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.root.update_idletasks()

        # Add a custom label for the data box with increased padding
        self.data_box_label = tk.Label(self.toolbar, text="", anchor="w", justify="left", font=("Courier New", 8))
        self.spacer_label = tk.Label(self.toolbar, text="   ", anchor="w")  # Spacer for padding
        self.spacer_label.pack(side=tk.LEFT, padx=10)  # Double the current padding
        self.data_box_label.pack(side=tk.LEFT, padx=5)
        
        # Create a frame for the PNG images on the right
        self.png_frame = tk.Frame(self.main_frame, bg='white')  # White background
        self.png_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Center the PNG image inside the png_frame
        self.png_label = tk.Label(self.png_frame, bg='white')
        self.png_label.pack(expand=True)
        self.png_label.bind("<Button-1>", self.open_in_system_viewer)  # Bind left-click to enlarge image

        # Add a cursor to show vertical time bar
        self.cursor_lines = []
        self.cursor_text = None

        self.savgol_window = 9  # Default window length
        self.savgol_polyorder = 3  # Default polynomial order

    def toggle_filter(self):
        """
        Toggle the Savitzky-Golay filter on/off, preserve current X zoom,
        and reapply Y rescaling based on visible data.
        """
        self.filter_enabled = not self.filter_enabled
        if self.filter_enabled:
            self.filter_button.config(text="Disable Savitzky-Golay Filter")
        else:
            self.filter_button.config(text="Enable Savitzky-Golay Filter")

        # Save current X-limits of all axes
        xlims = [[ax.get_xlim() for ax in ax_row] for ax_row in self.axs]

        # Re-plot data (this resets both X and Y limits)
        self.plot_data()

        # Restore X-limits and recompute Y-limits based on visible data
        for i, ax_row in enumerate(self.axs):
            for j, ax in enumerate(ax_row):
                x_min, x_max = xlims[i][j]
                ax.set_xlim(x_min, x_max)

                # Recalculate Y-limits based on data visible within current X range
                y_all = []
                for line in ax.get_lines():
                    xdata = np.asarray(line.get_xdata())
                    ydata = np.asarray(line.get_ydata())
                    if len(xdata) != len(ydata):
                        continue
                    mask = (xdata >= x_min) & (xdata <= x_max)
                    if np.any(mask):
                        y_visible = ydata[mask]
                        y_all.extend(y_visible)

                if y_all:
                    y_min = np.nanmin(y_all)
                    y_max = np.nanmax(y_all)
                    if not np.isnan(y_min) and not np.isnan(y_max) and y_max != y_min:
                        ax.set_ylim(y_min * 0.9, y_max * 1.1)

        if self.cursor_dynamics_enabled and self.cursor_lines == []:
            fake_event = type('Event', (object,), {
                'inaxes': self.axs[0][0],
                'xdata': self.axs[0][0].get_xlim()[0]
            })()
            self.on_mouse_move(fake_event)

        self.canvas.draw()

    def connect_cursor_events(self):
        """Connect the dynamic cursor and click events if the cursor is enabled"""
        self.motion_cid = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.right_click_cid = self.canvas.mpl_connect('button_press_event', self.on_right_click)

    def disconnect_cursor_events(self):
        """Disconnect the events of the cursor if is active"""
        if hasattr(self, 'motion_cid'):
            self.canvas.mpl_disconnect(self.motion_cid)
        if hasattr(self, 'right_click_cid'):
            self.canvas.mpl_disconnect(self.right_click_cid)


    def apply_filter(self, y):
        if self.filter_enabled and len(y) > self.savgol_window:
            return signal.savgol_filter(y, window_length=self.savgol_window, polyorder=self.savgol_polyorder)
        return y

    def save_confinement_time_to_csv(self, shot_number, confinement_time_data):
        """
        Save confinement time data to a CSV file.
        """
        local_folder = os.path.join(self.base_dir, f"shot_{shot_number}")
        os.makedirs(local_folder, exist_ok=True)
        output_file = os.path.join(local_folder, "tau_e.csv")
        confinement_time_data.to_csv(output_file, index=False)
        print(f"CSV file '{output_file}' has been generated successfully.")

    def plot_data(self):
        for ax_row in self.axs:
            for ax in ax_row:
                self.cursor_lines.clear()
                self.cursor_text = None
                ax.clear()

        color_cycle = itertools.cycle(self.color_palette)
        for shot, data in self.shots.items():
            color = next(color_cycle)
            lighter_color = self.lighter_color(color, 1.5)

            bt_data = data['Bt']
            ip_data = data['Ip']
            u_loop_data = data['U_loop']
            ne_data = data['ne']
            fast_camera_vertical_data = data['fast_camera_vertical']
            fast_camera_radial_data = data['fast_camera_radial']

            self.axs[0, 0].plot(bt_data['time_ms'], self.apply_filter(bt_data['Bt']), label=f'Bt ({shot})', color=color)
            self.axs[0, 1].plot(ip_data['time_ms'], self.apply_filter(ip_data['Ip']), label=f'Ip ({shot})', color=color)
            self.axs[1, 0].plot(u_loop_data['time_ms'], self.apply_filter(u_loop_data['U_loop']), label=f'U_loop ({shot})', color=color)
            self.axs[1, 1].plot(ne_data['time_ms'], self.apply_filter(ne_data['ne']), label=f'ne ({shot})', color=color)
            self.axs[2, 0].plot(fast_camera_radial_data['time_ms'], self.apply_filter(fast_camera_radial_data['radial_displacement']), label=f'\u0394r ({shot})', color=color)
            self.axs[2, 0].plot(fast_camera_vertical_data['time_ms'], self.apply_filter(fast_camera_vertical_data['vertical_displacement']), label=f'\u0394v ({shot})', color=lighter_color)

            if 'Te' in data:
                te_data = data['Te']
                self.axs[2, 1].plot(te_data['time_ms'], self.apply_filter(te_data['Te_0']), label=f'Te_0 ({shot})', color=color)
                self.axs[2, 1].plot(te_data['time_ms'], self.apply_filter(te_data['Te_avg_a']), label=f'Te_avg_a ({shot})', color=lighter_color, linestyle='--')

            if 'confinement_time' in data and not data['confinement_time'].empty:
                tau_data = data['confinement_time']
                self.axs[3, 0].plot(tau_data['time_ms'], self.apply_filter(tau_data['tau'] * 1e6), label=f'\u03c4_e ({shot})', color=color)    


    def electron_temperature_Spitzer_eV(self, eta_measured, Z_eff=3, eps=0, coulomb_logarithm=14):
        eta_s = eta_measured / Z_eff * (1 - np.sqrt(eps))**2
        term = 1.96 * constants.epsilon_0**2 / (np.sqrt(constants.m_e) * constants.elementary_charge**2 * coulomb_logarithm)
        return (term * eta_s)**(-2 / 3) / (constants.elementary_charge * 2 * np.pi)


    def load_shot(self):
        shot_number = simpledialog.askinteger("Input", "Enter the shot number:", parent=self.root)
        if shot_number:
            try:
                # Create a local folder for the shot if it doesn't exist
                local_folder = os.path.join(self.base_dir, f"shot_{shot_number}")
                os.makedirs(local_folder, exist_ok=True)
                
                # Load Bt, Ip, U_loop, ne, and fast camera data
                bt_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/Bt.csv",
                                         os.path.join(local_folder, "Bt.csv"), ['time_ms', 'Bt'])
                ip_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/Ip.csv",
                                        os.path.join(local_folder, "Ip.csv"), ['time_ms', 'Ip'])
                u_loop_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/U_loop.csv",
                                             os.path.join(local_folder, "U_loop.csv"), ['time_ms', 'U_loop'])
                ne_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/Interferometry/ne_lav.csv",
                                        os.path.join(local_folder, "ne.csv"), ['time_ms', 'ne'])

                # Load fast camera data from HTML pages
                fast_camera_vertical_data = self.load_fast_camera_data(
                    f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/FastCameras/Camera_Vertical/CameraVerticalPosition",
                    'vertical_displacement')
                fast_camera_radial_data = self.load_fast_camera_data(
                    f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/FastCameras/Camera_Radial/CameraRadialPosition",
                    'radial_displacement')
                
                # Truncate fast camera data to where Ip > 0.2
                plasma_start_time = ip_data[ip_data['Ip'] > 0.2]['time_ms'].min()
                fast_camera_vertical_data = fast_camera_vertical_data[fast_camera_vertical_data['time_ms'] >= plasma_start_time]
                fast_camera_radial_data = fast_camera_radial_data[fast_camera_radial_data['time_ms'] >= plasma_start_time]
                
                # Combine data into a single DataFrame for Te calculations
                combined_data = pd.DataFrame({
                    'time_ms': ip_data['time_ms'],
                    'Bt': bt_data['Bt'],
                    'Ip': ip_data['Ip'],
                    'U_loop': u_loop_data['U_loop']
                    })
                
                # Plasma geometry parameters
                R0 = 0.4  # Major plasma radius [m]
                a0 = 0.085  # Minor plasma radius [m]
                combined_data['R'] = R0
                combined_data['a'] = a0
                
                # Calculate safety factor and current densities
                nu = 2  # Peaking factor (more realistic than parabolic)
                combined_data['j_avg_a'] = combined_data['Ip'] * 1e3 / (np.pi * combined_data['a']**2)  # Average current density
                combined_data['j_0'] = combined_data['j_avg_a'] * (nu + 1)  # Maximum current density
                
                # Calculate plasma inductance
                l_i = np.log(1.65 + 0.89 * nu)  # Normalized internal inductance
                combined_data['L_p'] = constants.mu_0 * combined_data['R'] * (np.log(8 * combined_data['R'] / combined_data['a']) - 7 / 4 + l_i / 2)
                
                # Calculate toroidal electric field
                dt = np.diff(combined_data['time_ms'].values[:2]).item()  # Time step
                n_win = int(0.5 / dt)  # Smoothing window size
                if n_win % 2 == 0:
                    n_win += 1  # Ensure window size is odd
                    if n_win <= 3:  # Ensure n_win > polyorder
                        n_win = 5  # Set a minimum window size of 5
                    
                combined_data['dIp_dt'] = signal.savgol_filter(combined_data['Ip'] * 1e3, n_win, 3, 1, delta=dt * 1e-3)  # Derivative of Ip
                combined_data['E_phi_naive'] = combined_data['U_loop'] / (2 * np.pi * combined_data['R'])  # Naive estimate of E_phi
                combined_data['E_phi'] = (combined_data['U_loop'] - combined_data['L_p'] * combined_data['dIp_dt']) / (2 * np.pi * combined_data['R'])  # Corrected E_phi

                # Calculate resistivity and electron temperatures
                for s in ('0', 'avg_a'):
                    for k in ('', '_naive'):
                        combined_data[f'eta_{s}{k}'] = combined_data[f'E_phi{k}'] / combined_data[f'j_{s}']

                combined_data['eps'] = combined_data['a'] / combined_data['R']  # Inverse aspect ratio

                for s in ('0', 'avg_a'):
                    for k in ('', '_naive'):
                        combined_data[f'Te_{s}{k}'] = self.electron_temperature_Spitzer_eV(combined_data[f'eta_{s}{k}'], eps=combined_data['eps'])

                # Save Te_0 and Te_avg_a to a CSV file
                te_data = combined_data[['time_ms', 'Te_0', 'Te_avg_a']]
                te_csv_path = os.path.join(local_folder, "Te.csv")
                te_data.to_csv(te_csv_path, index=False)
                print(f"CSV file '{te_csv_path}' has been generated successfully.")

                # --- Confinement Time Calculation ---
                # Align time series of Ip and U_loop with ne
                common_time = ne_data['time_ms']

                # Interpolate Ip and U_loop to the common time base
                Ip_interp_func = interpolate.interp1d(ip_data['time_ms'], ip_data['Ip'], bounds_error=False, fill_value=np.nan)
                U_l_interp_func = interpolate.interp1d(u_loop_data['time_ms'], u_loop_data['U_loop'], bounds_error=False, fill_value=np.nan)

                Ip_interp = Ip_interp_func(common_time)
                U_l_interp = U_l_interp_func(common_time)

                # Create series for interpolation results
                Ip_interp_series = pd.Series(Ip_interp, index=common_time, name='Ip_interp')
                U_l_interp_series = pd.Series(U_l_interp, index=common_time, name='U_l_interp')
                ne_series = ne_data.set_index('time_ms')['ne']

                # Filter for positive values of ne, Ip, and U_l
                valid_idx = (ne_series > 0) & (Ip_interp_series > 0) & (U_l_interp_series > 0)
                ne_valid = ne_series[valid_idx]
                Ip_valid = Ip_interp_series[valid_idx]
                U_l_valid = U_l_interp_series[valid_idx]

                # Calculate confinement time
                # Convert Ip from kA to A for the formula: (1.0345 * ne_valid) / (16*10**19 * Ip_valid ** (1/3) * U_l_valid ** (5/3))
                # If ne is in m^-3, Ip is in A, U_l in V. Ip_valid is in kA, so convert to A.
                if not ne_valid.empty and not Ip_valid.empty and not U_l_valid.empty:
                    tau = (1.0345 * ne_valid) / (16 * 10**19 * Ip_valid **(1/3) * U_l_valid**(5/3))
                else:
                    tau = pd.Series([], dtype='float64') # Empty series if no valid data

                confinement_time_data = pd.DataFrame({'time_ms': tau.index, 'tau': tau.values})

                # Save confinement time to CSV
                self.save_confinement_time_to_csv(shot_number, confinement_time_data)
                
                # Store the data
                self.shots[shot_number] = {
                    'Bt': bt_data,
                    'Ip': ip_data,
                    'U_loop': u_loop_data,
                    'ne': ne_data,
                    'fast_camera_vertical': fast_camera_vertical_data,
                    'fast_camera_radial': fast_camera_radial_data,
                    'Te': te_data,
                    'confinement_time': confinement_time_data # Add confinement time data
                    }

                self.current_shot = shot_number
                self.plot_data()
                
                # Delay loading the PNG image to ensure the root window is ready
                self.root.after(100, lambda: self.load_png_image(shot_number, local_folder))
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load shot {shot_number}: {e}")


    def save_Te_to_csv(self, shot_number, Te_data):
        """
        Save Te_0 and Te_avg_a data to a CSV file.
        """
        local_folder = os.path.join(self.base_dir, f"shot_{shot_number}")
        os.makedirs(local_folder, exist_ok=True)  # Ensure the folder exists
        output_file = os.path.join(local_folder, "Te.csv")
        Te_data.to_csv(output_file, index=False)
        print(f"CSV file '{output_file}' has been generated successfully.")

    def load_data(self, url, local_path, column_names):
        """Load data from URL or local file if cached."""
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        else:
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch data from {url}")
            data = pd.read_csv(io.StringIO(response.text), header=None, names=column_names)
            data.to_csv(local_path, index=False)
            return data

    def load_fast_camera_data(self, url, column_name):
        """Load fast camera data from HTML page."""
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data from {url}")
        
        lines = response.text.strip().split('\n')
        time_ms = []
        values = []
        current_time = None
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                if parts[0]:  # Time value
                    try:
                        current_time = float(parts[0])
                    except ValueError:
                        current_time = None
                if parts[1]:  # Displacement value
                    try:
                        value = float(parts[1])
                        if current_time is not None:
                            time_ms.append(current_time)
                            values.append(value)
                            current_time = None
                    except ValueError:
                        pass
        return pd.DataFrame({'time_ms': time_ms, column_name: values})

    import os
    import subprocess  # Allows opening images with the system viewer

    def load_png_image(self, shot_number, local_folder):
        """Load and display the PNG image for multiple shots stacked vertically with labels and padding."""
        local_thumbnail_path = os.path.join(local_folder, "ScreenShotAll_thumbnail.png")
        local_full_path = os.path.join(local_folder, "ScreenShotAll_full.png")
    
        # Ensure Tkinter root exists before creating images
        if not hasattr(self, "root") or not self.root.winfo_exists():
            return  # Prevents premature image creation
    
        # Download the image if not already cached
        if not os.path.exists(local_full_path):
            png_url = f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/FastCameras/ScreenShotAll.png"
            response = requests.get(png_url)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch PNG image for shot {shot_number}")
        
            # Save the full-sized image locally
            with open(local_full_path, 'wb') as f:
                f.write(response.content)
        
            # Create and save the thumbnail version
            image = Image.open(io.BytesIO(response.content))
            image.thumbnail((300, 300))  # Resize for thumbnail
            image.save(local_thumbnail_path)
    
        # Ensure image reference list exists
        if not hasattr(self, "image_refs"):
            self.image_refs = []
    
        # Check if a thumbnail for this shot already exists
        for widget in self.png_frame.winfo_children():
            if hasattr(widget, "shot_number") and widget.shot_number == shot_number:
                print(f"Thumbnail for shot {shot_number} already exists. Skipping...")
                return  # Skip creating a duplicate thumbnail
    
        # Create a frame inside png_frame to center images vertically
        wrapper_frame = tk.Frame(self.png_frame, bg='white', padx=20)  # Adds right padding
        wrapper_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)
        
        # Add a text label for the shot number
        label = tk.Label(wrapper_frame, text=f"Shot #{shot_number}", bg='white', fg='black', font=("Arial", 12, "bold"))
        label.pack(side=tk.TOP, pady=5)
    
        # Load the thumbnail image
        thumbnail_image = Image.open(local_thumbnail_path)
        photo = ImageTk.PhotoImage(thumbnail_image)
    
        # Create a new label for the image
        img_label = tk.Label(wrapper_frame, image=photo, bg='white')
        img_label.image = photo  # Prevent garbage collection
        img_label.pack(side=tk.TOP, pady=5)
    
        # Store image reference to prevent garbage collection
        self.image_refs.append(photo)
        
        # Bind click event for opening image in the default system viewer
        img_label.bind("<Button-1>", lambda event, path=local_full_path: self.open_in_system_viewer(path))
    
        # Assign a unique identifier to the wrapper_frame
        wrapper_frame.shot_number = shot_number
    
    def open_in_system_viewer(self, image_path):
        """Open the image in the default system viewer."""
        try:
            # Normalize the file path for compatibility across operating systems
            image_path = os.path.normpath(image_path)
        
            # Check if the file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"File not found: {image_path}")
        
            # Open the image using the appropriate method for the OS
            if os.name == "nt":  # Windows
                os.startfile(image_path)
            elif os.name == "posix":  # Linux & macOS
                subprocess.run(["xdg-open", image_path], check=True)  # Linux
            else:
                subprocess.run(["open", image_path], check=True)  # macOS
        
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Could not open image: {e}")
        except PermissionError as e:
            messagebox.showerror("Error", f"Permission denied: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def clear_shots(self):
        self.shots = {}
        self.current_shot = None

        # Clear all plots
        for ax_row in self.axs:
            for ax in ax_row:
                ax.clear()

        # Clear the data box label
        if hasattr(self, "data_box_label"):
            self.data_box_label.config(text="")

        # Clear the PNG thumbnails by destroying all child widgets in png_frame
        for widget in self.png_frame.winfo_children():
            widget.destroy()

        # Reset the image references to prevent memory leaks
        if hasattr(self, "image_refs_dict"):
            self.image_refs_dict.clear()

        # Redraw the canvas
        self.canvas.draw()

    @staticmethod
    def lighter_color(color, factor=1.5):
        """Lighten a color by increasing its RGB values."""
        r, g, b = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"
    

    def plot_data(self):
        # Clear all subplots
        for ax_row in self.axs:
            for ax in ax_row:
                ax.clear()

        # Reset color cycle
        color_cycle = itertools.cycle(self.color_palette)

        for shot, data in self.shots.items():
            color = next(color_cycle)
            lighter_color = self.lighter_color(color, 1.5)

            bt_data = data['Bt']
            ip_data = data['Ip']
            u_loop_data = data['U_loop']
            ne_data = data['ne']
            fast_camera_vertical_data = data['fast_camera_vertical']
            fast_camera_radial_data = data['fast_camera_radial']

            def f(y): return self.apply_filter(y)

            self.axs[0, 0].plot(bt_data['time_ms'], f(bt_data['Bt']), label=f'Bt ({shot})', color=color)
            self.axs[0, 0].set_ylabel('Bt [T]')
            self.axs[0, 0].legend()
            self.axs[0, 0].grid(True, color='lightgray', alpha=0.4)

            self.axs[0, 1].plot(ip_data['time_ms'], f(ip_data['Ip']), label=f'Ip ({shot})', color=color)
            self.axs[0, 1].set_ylabel('Ip [kA]')
            self.axs[0, 1].legend()
            self.axs[0, 1].grid(True, color='lightgray', alpha=0.4)

            self.axs[1, 0].plot(u_loop_data['time_ms'], f(u_loop_data['U_loop']), label=f'U_loop ({shot})', color=color)
            self.axs[1, 0].set_ylabel('U_loop [V]')
            self.axs[1, 0].legend()
            self.axs[1, 0].grid(True, color='lightgray', alpha=0.4)

            self.axs[1, 1].plot(ne_data['time_ms'], f(ne_data['ne']), label=f'ne ({shot})', color=color)
            self.axs[1, 1].set_ylabel('ne [m^-3]')
            self.axs[1, 1].legend()
            self.axs[1, 1].grid(True, color='lightgray', alpha=0.4)

            self.axs[2, 0].plot(fast_camera_radial_data['time_ms'], f(fast_camera_radial_data['radial_displacement']), label=f'Δr ({shot})', color=color)
            self.axs[2, 0].plot(fast_camera_vertical_data['time_ms'], f(fast_camera_vertical_data['vertical_displacement']), label=f'Δv ({shot})', color=lighter_color)
            self.axs[2, 0].axhline(0, color='black', linestyle='--')
            self.axs[2, 0].set_ylabel('Displacement [mm]')
            self.axs[2, 0].legend()
            self.axs[2, 0].grid(True, color='lightgray', alpha=0.4)

            if 'Te' in data:
                te_data = data['Te']
                self.axs[2, 1].plot(te_data['time_ms'], f(te_data['Te_0']), label=f'Te_0 ({shot})', color=color)
                self.axs[2, 1].plot(te_data['time_ms'], f(te_data['Te_avg_a']), label=f'Te_avg_a ({shot})', color=lighter_color, linestyle='--')
                self.axs[2, 1].set_ylabel('Te [eV]')
                self.axs[2, 1].legend()
                self.axs[2, 1].grid(True, color='lightgray', alpha=0.4)

                visible_te_data = te_data[(te_data['time_ms'] >= 0) & (te_data['time_ms'] <= 35)]
                y_min = np.nanmin([visible_te_data['Te_0'].min(), visible_te_data['Te_avg_a'].min()])
                y_max = np.nanmax([visible_te_data['Te_0'].max(), visible_te_data['Te_avg_a'].max()])
                if not np.isnan(y_min) and not np.isnan(y_max):
                    self.axs[2, 1].set_ylim(y_min * 0.9, y_max * 1.1)

            if 'confinement_time' in data and not data['confinement_time'].empty:
                tau_data = data['confinement_time']
                self.axs[3, 0].plot(tau_data['time_ms'], f(tau_data['tau'] * 1e6), label=f'τ_e ({shot})', color=color)
                self.axs[3, 0].set_ylabel('τ_e [μs]')
                self.axs[3, 0].legend()
                self.axs[3, 0].grid(True, color='lightgray', alpha=0.4)

        for ax_row in self.axs:
            for ax in ax_row:
                ax.set_xlim(0, 35)
                if ax != self.axs[2, 1]:
                    ax.relim()
                    ax.autoscale_view()

        self.axs[3, 0].set_xlabel('Time [ms]')
        self.axs[3, 1].set_xlabel('Time [ms]')
        self.canvas.draw()
            
    def toggle_cursor_dynamics(self):
        self.cursor_dynamics_enabled = not self.cursor_dynamics_enabled

        if self.cursor_dynamics_enabled:
            self.cursor_toggle_button.config(text="Disable Cursor Dynamics")
            self.connect_cursor_events()
        else:
            self.cursor_toggle_button.config(text="Enable Cursor Dynamics")
            for line in self.cursor_lines:
                try:
                    if line.figure:
                        line.remove()
                except Exception:
                    pass
            self.cursor_lines.clear()
            if self.cursor_text:
                self.cursor_text.set_visible(False)
                self.cursor_text = None
            self.disconnect_cursor_events()

        self.canvas.draw()



    def on_mouse_move(self, event):
        if event.inaxes and self.cursor_dynamics_enabled:
            x = event.xdata  # Current x-coordinate (time)

            # Hide existing cursor lines and text
            for line in self.cursor_lines:
                try:
                    if line.figure:
                        line.remove()
                except Exception:
                    pass
            self.cursor_lines.clear()
            
            # Reset self.cursor_text before creating new text
            if self.cursor_text:
                self.cursor_text.set_visible(False)  # Hide the old cursor text
                self.cursor_text = None  # Clear the reference to the old text


            # Draw new cursor lines on all subplots
            for ax_row in self.axs:
                for ax in ax_row:
                    line = ax.axvline(x=x, color='lightgray', linestyle='--', linewidth=1.5)
                    self.cursor_lines.append(line)

            # Prepare the data table for all shots
            header = "Shot     Time(ms)  Bt(T)   Ip(kA)   ne(m^-3)   Te_0(eV)   τ_e(μs)" # Added τ_e
            data_table = [header]  # Header row
            for shot, data in self.shots.items():
                bt_data = data['Bt']
                ip_data = data['Ip']
                ne_data = data['ne']
                te_data = data.get('Te', None)
                confinement_time_data = data.get('confinement_time', None) # Get confinement time data

                # Find the nearest data point for each parameter
                bt_idx = np.abs(bt_data['time_ms'] - x).idxmin()
                ip_idx = np.abs(ip_data['time_ms'] - x).idxmin()
                ne_idx = np.abs(ne_data['time_ms'] - x).idxmin()

                # Get values at the nearest data point
                time_value = bt_data.loc[bt_idx, 'time_ms']
                bt_value = bt_data.loc[bt_idx, 'Bt']
                ip_value = ip_data.loc[ip_idx, 'Ip']
                ne_value = ne_data.loc[ne_idx, 'ne']

                # Get Te_0 value if available
                te_0_value = np.nan
                if te_data is not None and not te_data.empty: # Check if te_data is not None and not empty
                    te_idx = np.abs(te_data['time_ms'] - x).idxmin()
                    te_0_value = te_data.loc[te_idx, 'Te_0']

                # Get confinement time value if available
                tau_value = np.nan
                if confinement_time_data is not None and not confinement_time_data.empty: # Check if confinement_time_data is not None and not empty
                    tau_idx = np.abs(confinement_time_data['time_ms'] - x).idxmin()
                    tau_value = confinement_time_data.loc[tau_idx, 'tau'] * 1e6 # Convert to microseconds

                # Format the row for this shot with fixed-width columns
                row = (
                    f"{shot:<7}  {time_value:<8.2f}  {bt_value:<6.2f}  {ip_value:<7.2f}  "
                    f"{ne_value:<9.2e}  {te_0_value:<8.1f}  {tau_value:<8.1f}" # Added tau_value
                )
                data_table.append(row)
                
            # Update the data box at the bottom
            self.data_box_label.config(text="\n".join(data_table))

            # Redraw the canvas to reflect changes
            self.canvas.draw()


    def update_data_box(self, text):
        """
        Updates the data box at the bottom of the GUI.
        """
        # Create or update the data box label
        if not hasattr(self, "data_box_label"):
            self.data_box_label = tk.Label(self.main_frame, text=text, anchor="w", justify="left", font=("Arial", 8))
            self.data_box_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        else:
            self.data_box_label.config(text=text)

    def on_xlim_changed(self, event):
        """
        Triggered when x-limits change (e.g. from manual zoom).
        Prevents re-autoscaling Y if zoom was user-triggered manually.
        """
        if getattr(self, "ignore_xlim_callback", False):
            return  # Skip autoscaling during controlled zoom
        for ax_row in self.axs:
            for ax in ax_row:
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
        self.canvas.draw()


    def on_right_click(self, event):
        if event.button == 3 and self.cursor_dynamics_enabled:  # Right-click
            if event.inaxes:
                x = event.xdata  # Current x-coordinate (time)

                # Prepare clipboard content for all shots
                clipboard_text = "Shot\tTime(ms)\tBt(T)\tIp(kA)\tne(m^-3)\tTe_0(eV)\tτ_e(μs)\n" # Added τ_e
                for shot, data in self.shots.items():
                    bt_data = data['Bt']
                    ip_data = data['Ip']
                    ne_data = data['ne']
                    te_data = data.get('Te', None)
                    confinement_time_data = data.get('confinement_time', None) # Get confinement time data

                    # Find the nearest data point for each parameter
                    bt_idx = np.abs(bt_data['time_ms'] - x).idxmin()
                    ip_idx = np.abs(ip_data['time_ms'] - x).idxmin()
                    ne_idx = np.abs(ne_data['time_ms'] - x).idxmin()

                    # Get values at the nearest data point
                    time_value = bt_data.loc[bt_idx, 'time_ms']
                    bt_value = bt_data.loc[bt_idx, 'Bt']
                    ip_value = ip_data.loc[ip_idx, 'Ip']
                    ne_value = ne_data.loc[ne_idx, 'ne']

                    # Get Te_0 value if available
                    te_0_value = np.nan
                    if te_data is not None and not te_data.empty: # Check if te_data is not None and not empty
                        te_idx = np.abs(te_data['time_ms'] - x).idxmin()
                        te_0_value = te_data.loc[te_idx, 'Te_0']

                    # Get confinement time value if available
                    tau_value = np.nan
                    if confinement_time_data is not None and not confinement_time_data.empty: # Check if confinement_time_data is not None and not empty
                        tau_idx = np.abs(confinement_time_data['time_ms'] - x).idxmin()
                        tau_value = confinement_time_data.loc[tau_idx, 'tau'] * 1e6 # Convert to microseconds

                    # Append row to clipboard content
                    clipboard_text += f"{shot}\t{time_value:.2f}\t{bt_value:.2f}\t{ip_value:.2f}\t{ne_value:.2e}\t{te_0_value:.1f}\t{tau_value:.1f}\n" # Added tau_value

                # Copy data to clipboard
                pyperclip.copy(clipboard_text)

    def configure_filter(self):
        dialog = FilterConfigDialog(self.root, self)  # <-- pasa 'self' como segundo argumento
        if dialog.result:
            self.savgol_window, self.savgol_polyorder = dialog.result
            self.plot_data()

class FilterConfigDialog(simpledialog.Dialog):
    def __init__(self, parent, viewer):
        self.viewer = viewer  # referencia a TokamakDataViewer
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
        return self.window_entry  # Focus

    def apply(self):
        try:
            window = int(self.window_entry.get())
            polyorder = int(self.polyorder_entry.get())

            if window % 2 == 0 or polyorder >= window:
                raise ValueError

            self.result = (window, polyorder)
        except:
            messagebox.showerror("Invalid Input", "Ensure window is odd and polyorder < window.")
            self.result = None

if __name__ == "__main__":
    root = tk.Tk()
    app = TokamakDataViewer(root)
    root.mainloop()

