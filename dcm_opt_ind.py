import numpy as np
import plotly.graph_objects as go


def _buck_dcm_ind(vs: float, vo:float, io:float, ind: float, T: float) -> tuple[float, float]:
    iL = io
    iLmin = iL - np.divide((vs - vo) * vo * T, 2 * ind * pow(10, -6) * vs)
    iLmax = iL + np.divide((vs - vo) * vo * T, 2 * ind * pow(10, -6) * vs)
    return iLmin, iLmax


def _boost_dcm_ind(vs: float, vo:float, io:float, ind: float, T: float) -> tuple[float, float]:
    iL = np.divide(io * vo, vs)
    iLmin = iL - np.divide((vo - vs) * vs * T, 2 * ind * pow(10, -6) * vo)
    iLmax = iL + np.divide((vo - vs) * vs * T, 2 * ind * pow(10, -6) * vo)
    return iLmin, iLmax


def _buckboost_dcm_ind(vs: float, vo:float, io:float, ind: float, T: float) -> tuple[float, float]:
    iL = np.divide(io * (vs + vo), vs)
    iLmin = iL - np.divide(vo * vs * T, 2 * ind * pow(10, -6) * (vo + vs))
    iLmax = iL + np.divide(vo * vs * T, 2 * ind * pow(10, -6) * (vo + vs))
    return iLmin, iLmax


_IND_MNIMAX_FUNCS = {
    'Buck': lambda vs, vo, io, ind, T: _buck_dcm_ind(vs, vo, io, ind, T),
    'Boost': lambda vs, vo, io, ind, T: _boost_dcm_ind(vs, vo, io, ind, T),
    'BuckBoost': lambda vs, vo, io, ind, T: _buckboost_dcm_ind(vs, vo, io, ind, T),
}


def _plot_dcm_opt_ind(iLmin_list: list[float], iLmax_list: list[float]) -> tuple[go.Figure, list[float]]:
    lc = [indlc * pow(10, -6) for indlc in range(1, 400)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lc, y=iLmin_list,
                        mode='lines',
                        name='iLmin'))
    fig.add_trace(go.Scatter(x=lc, y=iLmax_list,
                        mode='lines',
                        name='iLmax'))

    fig.update_layout(title='Inductor Value Optimization for DCM operation',
                    xaxis_title='Inductor (uH)', height=600, width=720,
                    yaxis_title='Inductor Current (A)')

    return fig, lc


def dcm_opt_ind(vs: float, vo: float, io: float, T: float, mode: str) -> tuple[go.Figure, list[float], list[float], list[float]]:
    iLmin_list = []
    iLmax_list = []
    for ind in range(1, 400):
        iLmin, iLmax = _IND_MNIMAX_FUNCS[mode](vs, vo, io, ind, T)
        iLmin_list.append(iLmin)
        iLmax_list.append(iLmax)
    
    fig, lc = _plot_dcm_opt_ind(iLmin_list, iLmax_list)
    return fig, lc, iLmax_list, iLmin_list
