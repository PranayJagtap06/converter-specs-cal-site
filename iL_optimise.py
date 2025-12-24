import numpy as np
import matplotlib.pyplot as plt

v1 = float(48)
v2 = float(230)
T = float(31.875) * pow(10, -6)
iLoad = float(12)
Lc = range(1, 400)
i_max = []
i_min = []


for L in Lc:
    i_mn = iLoad - np.divide((v2 - v1) * v1 * T, 2 * L * pow(10, -6) * v2)
    i_mx = iLoad + np.divide((v2 - v1) * v1 * T, 2 * L * pow(10, -6) * v2)
    i_min.append(i_mn)
    i_max.append(i_mx)

lc = [l for l in Lc]
plt.plot(lc, i_min, label="iLmin")
plt.plot(lc, i_max, label="iLmax")
plt.xlabel("Inductor (uH)")
plt.ylabel("Inductor Current (A)")
plt.title("Inductor Value Optimization for DCM operation")
plt.legend()
plt.grid()
plt.show()
