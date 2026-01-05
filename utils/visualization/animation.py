import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d  # para interpolação suave

# Load data
df = pd.read_csv(sys.argv[1])

#df = df.sort_values("data").reset_index(drop=True)

# Converte datas para números (eixos do matplotlib não entendem datetime nativamente na interpolação)

df = df[df['Node ID'] ==  1]

x_raw = df["Timestamp"]
x = np.arange(len(x_raw))  # índice inteiro para cada ponto
y = df["Throughput DL"].values


# Interpolate points
num_frames = 300  
x_smooth = np.linspace(x.min(), x.max(), num_frames)
interp_func = interp1d(x, y, kind="linear") 
y_smooth = interp_func(x_smooth)

# Setup graph
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min() - 5, y.max() + 5)
ax.set_xlabel("Time (100 ms)", fontsize=16)
ax.set_ylabel("Throughtput (Mbps)", fontsize=16)

def init():

    line.set_data([], [])
    return line,

def update(frame):
    
    xdata = x_smooth[:frame + 1]
    ydata = y_smooth[:frame + 1]
    line.set_data(xdata, ydata)
    return line,

ani = FuncAnimation(fig, 
                    update, 
                    frames=num_frames, 
                    init_func=init, 
                    blit=True,
                    repeat=False,
                    interval=20)

plt.tight_layout()
plt.show()

# Save GIF
ani.save("figures/animation.gif", writer="pillow", fps=20)

