import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class SimuladorRLCPro:
    def __init__(self, root):
        self.root = root
        self.root.title("RLC Circuit")
        self.root.geometry("1150x750")
        self.root.configure(bg="#f5f5f7")

        self.frame_esquerda = ttk.Frame(self.root, padding=15)
        self.frame_esquerda.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.frame_direita = ttk.Frame(self.root, padding=10)
        self.frame_direita.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.build_interface()
        self.initialize_chart()

    def build_interface(self):
        lbl_circuito = ttk.Label(self.frame_esquerda, text="Circuit Components", font=("Arial", 11, "bold"))
        lbl_circuito.pack(anchor="w", pady=(0, 5))
        
        frame_comp = ttk.LabelFrame(self.frame_esquerda, padding=10)
        frame_comp.pack(fill="x", pady=(0, 15))

        self.val_R = self.build_entry(frame_comp, "Resistance (R) [Ω]:", "0.5") # Resistance [Ω]
        self.val_L = self.build_entry(frame_comp, "Inductance (L) [H]:", "1.0") # Inductance [H]
        self.val_C = self.build_entry(frame_comp, "Capacitance (C) [F]:", "0.01") # Capacitance [F]
        self.val_t = self.build_entry(frame_comp, "Simulation Time [s]:", "15") # Simulation Time [s]

        lbl_fonte = ttk.Label(self.frame_esquerda, text="Font Configuration", font=("Arial", 11, "bold"))
        lbl_fonte.pack(anchor="w", pady=(0, 5))

        frame_tipo_fonte = ttk.Frame(self.frame_esquerda)
        frame_tipo_fonte.pack(fill="x", pady=(0, 5))

        self.tipo_fonte = tk.StringVar(value="AC")
        rb_ac = ttk.Radiobutton(frame_tipo_fonte, text="Alternating Current (AC)", variable=self.tipo_fonte, value="AC", command=self.alternate_font)
        rb_ac.pack(side=tk.LEFT, padx=5) # Alternating Current
        rb_dc = ttk.Radiobutton(frame_tipo_fonte, text="Direct Current (DC)", variable=self.tipo_fonte, value="DC", command=self.alternate_font)
        rb_dc.pack(side=tk.LEFT, padx=5) # Direct Current

        self.frame_params_fonte = ttk.LabelFrame(self.frame_esquerda, padding=10)
        self.frame_params_fonte.pack(fill="x", pady=(0, 15))
        
        self.val_A = self.build_entry(self.frame_params_fonte, "Amplitude / Voltage [V]:", "50.0") # AC/DC - param
        self.val_w = self.build_entry(self.frame_params_fonte, "Angular Frequency (ω) [rad/s]:", "11.5") # AC - param
        self.val_offset = self.build_entry(self.frame_params_fonte, "Offset [V]:", "0.0") # AC - param

        self.btn_simular = ttk.Button(self.frame_esquerda, text="Simulate", command=self.execute_simulation)
        self.btn_simular.pack(fill="x", ipady=5)

    def build_entry(self, master, label_text, default_val):
        frame = ttk.Frame(master)
        frame.pack(fill="x", pady=3)
        lbl = ttk.Label(frame, text=label_text, width=28, anchor="w")
        lbl.pack(side=tk.LEFT)
        entry = ttk.Entry(frame, width=10)
        entry.insert(0, default_val)
        entry.pack(side=tk.RIGHT, expand=True, fill="x")
        return entry

    def alternate_font(self):
        # Disable unnecessary fields if it's a DC
        if self.tipo_fonte.get() == "DC":
            self.val_w.configure(state="disabled")
            self.val_offset.configure(state="disabled")
        else:
            self.val_w.configure(state="normal")
            self.val_offset.configure(state="normal")

    def initialize_chart(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 5), dpi=100)
        self.ax.set_title("Current x Time")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Current (A)")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_direita)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame_direita)
        self.toolbar.update()

    def f(self, t, y, R, L, C):
        q = y[0]  # Charge
        i = y[1]  # Current
        
        # Determines the voltage V(t) based on the user's choice.
        if self.tipo_fonte.get() == "AC":
            A = float(self.val_A.get())
            w = float(self.val_w.get())
            offset = float(self.val_offset.get())
            V_t = A * np.sin(w * t) + offset
        else: # DC
            V_t = float(self.val_A.get())
            
        # First-order differential equations:
        dq_dt = i
        di_dt = (V_t - R * i - (1 / C) * q) / L
        
        return np.array([dq_dt, di_dt])

    def rk4_step(self, t, y, h, R, L, C):
        k1 = self.f(t, y, R, L, C)
        k2 = self.f(t + h/2, y + (h/2) * k1, R, L, C)
        k3 = self.f(t + h/2, y + (h/2) * k2, R, L, C)
        k4 = self.f(t + h, y + h * k3, R, L, C)
        return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    def execute_simulation(self):
        try:
            # Collection of numerical parameters
            R = float(self.val_R.get())
            L = float(self.val_L.get())
            C = float(self.val_C.get())
            t_final = float(self.val_t.get())
            
            # Step Settings
            N = 15000
            h = t_final / N
            tempos = np.linspace(0, t_final, N)
            
            # Initial conditions: q(0) = 0, i(0) = 0
            y = np.zeros((N, 2))
            
            # Runge-Kutta 4 Loop
            for n in range(N - 1):
                y[n+1] = self.rk4_step(tempos[n], y[n], h, R, L, C)
                
            corrente = y[:, 1] # Electric current is our second state variable (di/dt)

            # Chart Update
            self.ax.clear()
            self.ax.plot(tempos, corrente, color="#007aff", linewidth=2, label="Corrente $i(t)$")
            self.ax.set_title("RLC Circuit Response", fontsize=12, fontweight="bold")
            self.ax.set_xlabel("Time (s)", fontsize=10)
            self.ax.set_ylabel("Current (A)", fontsize=10)
            self.ax.grid(True, linestyle="--", alpha=0.5)
            self.ax.legend(loc="upper right")
            self.fig.tight_layout()
            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Input Error", "Please ensure you fill in all fields with valid numbers.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimuladorRLCPro(root)
    root.mainloop()