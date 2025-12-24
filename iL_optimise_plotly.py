import numpy as np
import plotly.graph_objects as go

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

lc = [indlc for indlc in Lc]

fig = go.Figure()
fig.add_trace(go.Scatter(x=lc, y=i_min,
                    mode='lines',
                    name='iLmin'))
fig.add_trace(go.Scatter(x=lc, y=i_max,
                    mode='lines',
                    name='iLmax'))

fig.update_layout(title='Inductor Value Optimization for DCM operation',
                   xaxis_title='Inductor (uH)', height=600, width=720,
                   yaxis_title='Inductor Current (A)')

fig.show()
