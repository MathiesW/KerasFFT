# %%
import os
os.environ["KERAS_BACKEND"] = "jax"
from src.keras_fft import derivative
import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(0, 2 * np.pi, 512, endpoint=False)
dt = np.mean(np.diff(t))

x = sum([np.cos(k * t - np.pi * np.random.rand()) for k in range(1,5)])
dxdt = derivative(x, d=dt, n=1)
ddxddt = derivative(x, d=dt, n=2)

dxdt /= dxdt.max()
ddxddt /= ddxddt.max()

fig, ax = plt.subplots(1, 2, sharex=True, sharey=False)
ax[0].plot(t, x)
ax[1].plot(t, dxdt, label="1st")
ax[1].plot(t, ddxddt, label="2nd")

ax[1].legend()

plt.show()

# %%
