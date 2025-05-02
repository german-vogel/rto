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

class TokamakDataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("GOLEM Tokamak Data Viewer")
        self.shots = {}
        self.current_shot = None
        
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
        
        # Create a frame for the plot on the left
        self.plot_frame = tk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create a 3x2 grid of subplots with shared x-axis
        self.fig, self.axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        self.fig.subplots_adjust(left=0.07, right=0.96, top=0.95, bottom=0.09, hspace=0.25, wspace=0.15)
        
        # Bind the xlim_changed event to all subplots
        for ax_row in self.axs:
            for ax in ax_row:
                ax.callbacks.connect('xlim_changed', self.on_xlim_changed)
        
        # Embed the plot in the plot frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add the Matplotlib navigation toolbar below the plot
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas.draw()
        
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
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_right_click)


    def electron_temperature_Spitzer_eV(self, eta_measured, Z_eff=3, eps=0, coulomb_logarithm=14):
        eta_s = eta_measured / Z_eff * (1 - np.sqrt(eps))**2
        term = 1.96 * eta_s * (3 * constants.epsilon_0**2 /
                           (np.sqrt(constants.m_e) * constants.elementary_charge**2 * coulomb_logarithm))
        return term**(-2 / 3) / (constants.elementary_charge * 2 * np.pi)



    def load_shot(self):
        shot_number = simpledialog.askinteger("Input", "Enter the shot number:", parent=self.root)
        if shot_number:
            try:
                # Create a local folder for the shot if it doesn't exist
                local_folder = f"shot_{shot_number}"
                os.makedirs(local_folder, exist_ok=True)
                
                # Load Bt, Ip, U_loop, ne, and fast camera data
                bt_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/Bt.csv",
                                         f"{local_folder}/Bt.csv", ['time_ms', 'Bt'])
                ip_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/Ip.csv",
                                         f"{local_folder}/Ip.csv", ['time_ms', 'Ip'])
                u_loop_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/U_loop.csv",
                                             f"{local_folder}/U_loop.csv", ['time_ms', 'U_loop'])
                ne_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/Interferometry/ne_lav.csv",
                                         f"{local_folder}/ne.csv", ['time_ms', 'ne'])

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
                te_csv_path = f"{local_folder}/Te.csv"
                te_data.to_csv(te_csv_path, index=False)
                print(f"CSV file '{te_csv_path}' has been generated successfully.")

                # Store the data
                self.shots[shot_number] = {
                    'Bt': bt_data,
                    'Ip': ip_data,
                    'U_loop': u_loop_data,
                    'ne': ne_data,
                    'fast_camera_vertical': fast_camera_vertical_data,
                    'fast_camera_radial': fast_camera_radial_data,
                    'Te': te_data  # Add Te data to the shots dictionary
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
        local_folder = f"shot_{shot_number}"
        os.makedirs(local_folder, exist_ok=True)  # Ensure the folder exists
        output_file = f"{local_folder}/Te.csv"
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
        local_thumbnail_path = f"{local_folder}/ScreenShotAll_thumbnail.png"
        local_full_path = f"{local_folder}/ScreenShotAll_full.png"
    
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
        for ax_row in self.axs:
            for ax in ax_row:
                ax.clear()
    
        # Reset the color cycle for each plot
        color_cycle = itertools.cycle(self.color_palette)
        for shot, data in self.shots.items():
            # Get the next color from the cycle
            color = next(color_cycle)
            lighter_color = self.lighter_color(color, 1.5)  # Generate a lighter tone
        
            bt_data = data['Bt']
            ip_data = data['Ip']
            u_loop_data = data['U_loop']
            ne_data = data['ne']
            fast_camera_vertical_data = data['fast_camera_vertical']
            fast_camera_radial_data = data['fast_camera_radial']
        
            # Plot Bt data
            self.axs[0, 0].plot(bt_data['time_ms'], bt_data['Bt'], label=f'Bt ({shot})', color=color)
            self.axs[0, 0].set_ylabel('Bt [T]')
            self.axs[0, 0].legend()
            self.axs[0, 0].grid(True, color='lightgray', alpha=0.4)
            self.axs[0, 0].margins(y=0.05)  # Add small margin
        
            # Plot Ip data
            self.axs[0, 1].plot(ip_data['time_ms'], ip_data['Ip'], label=f'Ip ({shot})', color=color)
            self.axs[0, 1].set_ylabel('Ip [kA]')
            self.axs[0, 1].legend()
            self.axs[0, 1].grid(True, color='lightgray', alpha=0.4)
            self.axs[0, 1].margins(y=0.05)  # Add small margin
        
            # Plot U_loop data
            self.axs[1, 0].plot(u_loop_data['time_ms'], u_loop_data['U_loop'], label=f'U_loop ({shot})', color=color)
            self.axs[1, 0].set_ylabel('U_loop [V]')
            self.axs[1, 0].legend()
            self.axs[1, 0].grid(True, color='lightgray', alpha=0.4)
            self.axs[1, 0].margins(y=0.05)  # Add small margin
        
            # Plot ne data
            self.axs[1, 1].plot(ne_data['time_ms'], ne_data['ne'], label=f'ne ({shot})', color=color)
            self.axs[1, 1].set_ylabel('ne [m^-3]')
            self.axs[1, 1].legend()
            self.axs[1, 1].grid(True, color='lightgray', alpha=0.4)
            self.axs[1, 1].margins(y=0.05)  # Add small margin
        
            # Plot fast camera data
            self.axs[2, 0].plot(fast_camera_radial_data['time_ms'], fast_camera_radial_data['radial_displacement'], label=f'Δr ({shot})', color=color)
            self.axs[2, 0].plot(fast_camera_vertical_data['time_ms'], fast_camera_vertical_data['vertical_displacement'], label=f'Δv ({shot})', color=lighter_color)
            self.axs[2, 0].axhline(0, color='black', linestyle='--')
            self.axs[2, 0].set_ylabel('Displacement [mm]')
            self.axs[2, 0].legend()
            self.axs[2, 0].grid(True, color='lightgray', alpha=0.4)
            self.axs[2, 0].margins(y=0.05)  # Add small margin
        
            # Plot Te_0 and Te_avg_a in the 6th panel (self.axs[2, 1])
            if 'Te' in data:
                te_data = data['Te']
                self.axs[2, 1].plot(te_data['time_ms'], te_data['Te_0'], label=f'Te_0 ({shot})', color=color)
                self.axs[2, 1].plot(te_data['time_ms'], te_data['Te_avg_a'], label=f'Te_avg_a ({shot})', color=lighter_color, linestyle='--')
                self.axs[2, 1].set_ylabel('Te [eV]')
                self.axs[2, 1].legend()
                self.axs[2, 1].grid(True, color='lightgray', alpha=0.4)
                self.axs[2, 1].set_xlabel('Time [ms]')
                
                # Dynamically adjust y-axis limits for Te based on visible data (0–35 ms)
                visible_te_data = te_data[(te_data['time_ms'] >= 0) & (te_data['time_ms'] <= 35)]
                y_min = np.nanmin([visible_te_data['Te_0'].min(), visible_te_data['Te_avg_a'].min()])
                y_max = np.nanmax([visible_te_data['Te_0'].max(), visible_te_data['Te_avg_a'].max()])
                if not np.isnan(y_min) and not np.isnan(y_max):
                    self.axs[2, 1].set_ylim(y_min * 0.9, y_max * 1.1)  # Add 10% margin
        
            # Set x-label for the bottom row
            self.axs[2, 0].set_xlabel('Time [ms]')
            self.axs[2, 1].set_xlabel('Time [ms]')
    
        # Limit x-axis to 35 ms
        for ax_row in self.axs:
            for ax in ax_row:
                ax.set_xlim(0, 35)
                # Only apply relim and autoscale_view for panels without explicit limits
                if ax != self.axs[2, 1]:  # Exclude the Te panel
                    ax.relim()  # Recalculate limits based on data
                    ax.autoscale_view()  # Autoscale view
    
        self.canvas.draw()


    def toggle_cursor_dynamics(self):
        # Toggle the state of cursor dynamics
        self.cursor_dynamics_enabled = not self.cursor_dynamics_enabled
        
        if self.cursor_dynamics_enabled:
            # Enable cursor dynamics
            self.cursor_toggle_button.config(text="Disable Cursor Dynamics")
        
            # Reconnect event handlers for mouse movement and right-click
            self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
            self.canvas.mpl_connect('button_press_event', self.on_right_click)

        else:
            # Disable cursor dynamics
            self.cursor_toggle_button.config(text="Enable Cursor Dynamics")
        
            # Clear all existing cursor lines and text
            for line in self.cursor_lines:
                try:
                    line.remove()  # Remove the line from the plot
                except ValueError:
                    pass  # Ignore if the line is already removed
            self.cursor_lines.clear()

            if self.cursor_text:
                self.cursor_text.set_visible(False)  # Hide the cursor text
                self.cursor_text = None

            # Disconnect event handlers for mouse movement and right-click
            self.canvas.mpl_disconnect(self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move))
            self.canvas.mpl_disconnect(self.canvas.mpl_connect('button_press_event', self.on_right_click))

        # Redraw the canvas to reflect changes
        self.canvas.draw()

    def on_mouse_move(self, event):
        if event.inaxes and self.cursor_dynamics_enabled:
            x = event.xdata  # Current x-coordinate (time)

            # Hide existing cursor lines and text
            for line in self.cursor_lines:
                try:
                    line.remove()  # Remove the line from the plot
                except ValueError:
                    pass  # Ignore if the line is already removed
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
            header = "Shot     Time(ms)  Bt(T)   Ip(kA)   ne(m^-3)   Te_0(eV)"
            data_table = [header]  # Header row
            for shot, data in self.shots.items():
                bt_data = data['Bt']
                ip_data = data['Ip']
                ne_data = data['ne']
                te_data = data.get('Te', None)

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
                if te_data is not None:
                    te_idx = np.abs(te_data['time_ms'] - x).idxmin()
                    te_0_value = te_data.loc[te_idx, 'Te_0']

                # Format the row for this shot with fixed-width columns
                row = (
                    f"{shot:<7}  {time_value:<8.2f}  {bt_value:<6.2f}  {ip_value:<7.2f}  "
                    f"{ne_value:<9.2e}  {te_0_value:<8.1f}"
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
        Dynamically adjust y-axis limits when zooming or panning.
        """
        for ax_row in self.axs:
            for ax in ax_row:
                ax.relim()  # Recalculate limits based on visible data
                ax.autoscale_view(scalex=False, scaley=True)  # Autoscale y-axis only
        self.canvas.draw()

    def on_right_click(self, event):
        if event.button == 3 and self.cursor_dynamics_enabled:  # Right-click
            if event.inaxes:
                x = event.xdata  # Current x-coordinate (time)

                # Prepare clipboard content for all shots
                clipboard_text = "Shot\tTime(ms)\tBt(T)\tIp(kA)\tne(m^-3)\tTe_0(eV)\n"
                for shot, data in self.shots.items():
                    bt_data = data['Bt']
                    ip_data = data['Ip']
                    ne_data = data['ne']
                    te_data = data.get('Te', None)

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
                    if te_data is not None:
                        te_idx = np.abs(te_data['time_ms'] - x).idxmin()
                        te_0_value = te_data.loc[te_idx, 'Te_0']

                    # Append row to clipboard content
                    clipboard_text += f"{shot}\t{time_value:.2f}\t{bt_value:.2f}\t{ip_value:.2f}\t{ne_value:.2e}\t{te_0_value:.1f}\n"

                # Copy data to clipboard
                pyperclip.copy(clipboard_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = TokamakDataViewer(root)
    root.mainloop()