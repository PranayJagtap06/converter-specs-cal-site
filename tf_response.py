"""
Small-signal transfer functions for basic CCM DC-DC converters.

Reference: Erickson & Maksimovic, Table 8.2 (Section 8.2.2)
    Gvg(s) = Gg0 · 1 / (1 + s/(Q·ω₀) + (s/ω₀)²)           — Eq. 8.148
    Gvd(s) = Gd0 · (1 − s/ωz) / (1 + s/(Q·ω₀) + (s/ω₀)²)  — Eq. 8.147

Notation: D' = 1 − D, V = Vo (output voltage at operating point)
"""

import control as ct
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


# ---------------------------------------------------------------------------
# Shared denominator builder
# ---------------------------------------------------------------------------
def _second_order_den(d_prime: float, inductor: float, capacitor: float,
                      resistor: float) -> np.ndarray:
    """
    Build the standard 2nd-order denominator coefficients [1, a₁, a₀].

    For buck:       ω₀ = 1/√(LC),   Q = R√(C/L)   → a₁ = 1/(RC),  a₀ = 1/(LC)
    For boost/bb:   ω₀ = D'/√(LC),  Q = D'R√(C/L) → a₁ = 1/(RC),  a₀ = D'²/(LC)

    Parameters
    ----------
    d_prime : float
        Complement of duty cycle (1 − D).  Use 1.0 for buck.
    inductor, capacitor, resistor : float
        Component values.
    """
    a1 = 1.0 / (resistor * capacitor)
    a0 = (d_prime ** 2) / (inductor * capacitor)
    return np.array([1.0, a1, a0])


# ---------------------------------------------------------------------------
# Buck converter transfer functions
# ---------------------------------------------------------------------------
def _buck_gvg(d: float, inductor: float, capacitor: float,
              resistor: float) -> ct.TransferFunction:
    """
    Buck line-to-output TF  Gvg(s).

    Gg0 = D,  ω₀ = 1/√(LC),  ωz = ∞  (no zero)
    num = Gg0 · ω₀² = D / (LC)

    Ref: Table 8.2, row 1.
    """
    lc = inductor * capacitor
    num = np.array([d / lc])
    den = _second_order_den(1.0, inductor, capacitor, resistor)
    return ct.tf(num, den)


def _buck_gvd(vin: float, inductor: float, capacitor: float,
              resistor: float) -> ct.TransferFunction:
    """
    Buck control-to-output TF  Gvd(s).

    Gd0 = V/D = Vin (since V = Vin·D),  ωz = ∞  (no zero)
    num = Gd0 · ω₀² = Vin / (LC)

    Ref: Table 8.2, row 1.
    """
    lc = inductor * capacitor
    num = np.array([vin / lc])
    den = _second_order_den(1.0, inductor, capacitor, resistor)
    return ct.tf(num, den)


# ---------------------------------------------------------------------------
# Boost converter transfer functions
# ---------------------------------------------------------------------------
def _boost_gvg(d: float, inductor: float, capacitor: float,
               resistor: float) -> ct.TransferFunction:
    """
    Boost line-to-output TF  Gvg(s).

    Gg0 = 1/D',  ω₀ = D'/√(LC),  ωz = ∞  (no zero in Gvg)
    Gg0 · ω₀² = (1/D') · D'²/(LC) = D' / (LC)

    Ref: Table 8.2, row 2.
    """
    d_prime = 1.0 - d
    lc = inductor * capacitor
    num = np.array([d_prime / lc])
    den = _second_order_den(d_prime, inductor, capacitor, resistor)
    return ct.tf(num, den)


def _boost_gvd(d: float, vin: float, inductor: float, capacitor: float,
               resistor: float) -> ct.TransferFunction:
    """
    Boost control-to-output TF  Gvd(s).

    Gd0 = V/D' = Vin/D'²   (since V = Vin/D')
    ωz  = D'²R / L          (RHP zero)

    Gvd(s) = Gd0·ω₀²·(1 − s/ωz) / (s² + s·ω₀/Q + ω₀²)
    Gd0·ω₀² = Vin/(LC)
    Numerator polynomial: [-Vin/(D'²·R·C),  Vin/(LC)]

    Ref: Table 8.2, row 2.
    """
    d_prime = 1.0 - d
    lc = inductor * capacitor
    rc = resistor * capacitor
    num = np.array([
        -vin / (rc * d_prime ** 2),   # s¹ coefficient
         vin / lc                      # s⁰ coefficient
    ])
    den = _second_order_den(d_prime, inductor, capacitor, resistor)
    return ct.tf(num, den)


# ---------------------------------------------------------------------------
# Buck-Boost converter transfer functions
# ---------------------------------------------------------------------------
def _buckboost_gvg(d: float, inductor: float, capacitor: float,
                   resistor: float) -> ct.TransferFunction:
    """
    Buck-Boost line-to-output TF  Gvg(s).

    Gg0 = −D/D',  ω₀ = D'/√(LC),  ωz = ∞  (no zero in Gvg)
    Gg0 · ω₀² = (−D/D') · D'²/(LC) = −D·D' / (LC)

    Ref: Table 8.2, row 3.
    """
    d_prime = 1.0 - d
    lc = inductor * capacitor
    num = np.array([-(d * d_prime) / lc])
    den = _second_order_den(d_prime, inductor, capacitor, resistor)
    return ct.tf(num, den)


def _buckboost_gvd(d: float, vin: float, inductor: float, capacitor: float,
                   resistor: float) -> ct.TransferFunction:
    """
    Buck-Boost control-to-output TF  Gvd(s).

    Gd0 = V/(D·D') = −Vin/D'²   (V = −Vin·D/D' for inverting topology)
    ωz  = D'²R / (D·L)          (RHP zero)

    Gvd(s) = Gd0·ω₀²·(1 − s/ωz) / (s² + s·ω₀/Q + ω₀²)
    Gd0·ω₀² = −Vin/(LC)
    Numerator polynomial: [Vin·D/(D'²·R·C),  −Vin/(LC)]

    Ref: Table 8.2, row 3.
    """
    d_prime = 1.0 - d
    lc = inductor * capacitor
    rc = resistor * capacitor
    num = np.array([
        (vin * d) / (rc * d_prime ** 2),   # s¹ coefficient
        -vin / lc                           # s⁰ coefficient
    ])
    den = _second_order_den(d_prime, inductor, capacitor, resistor)
    return ct.tf(num, den)


# ---------------------------------------------------------------------------
# Dispatcher & plot builder
# ---------------------------------------------------------------------------
_GVG_FUNCS = {
    'Buck':      lambda d, vin, L, C, R: _buck_gvg(d, L, C, R),
    'Boost':     lambda d, vin, L, C, R: _boost_gvg(d, L, C, R),
    'BuckBoost': lambda d, vin, L, C, R: _buckboost_gvg(d, L, C, R),
}

_GVD_FUNCS = {
    'Buck':      lambda d, vin, L, C, R: _buck_gvd(vin, L, C, R),
    'Boost':     lambda d, vin, L, C, R: _boost_gvd(d, vin, L, C, R),
    'BuckBoost': lambda d, vin, L, C, R: _buckboost_gvd(d, vin, L, C, R),
}


def _build_figure(t: np.ndarray, y: np.ndarray, y_ss: float,
                  title: str, trace_name: str) -> go.Figure:
    """Create a Plotly figure for a transient response."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name=trace_name))
    fig.add_trace(go.Scatter(
        x=t, y=np.full_like(t, y_ss),
        mode='lines', name='Steady State Value'
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(exponentformat='e', tickformat='.0e', showexponent='all'),
        xaxis_title='Time (sec)',
        yaxis_title='Response (volts)',
        height=600,
        width=720,
    )
    fig.add_annotation(
        x=t[-1], y=y_ss,
        text=f'Steady State Value: {y_ss:.4f}',
        showarrow=True, arrowhead=1, ax=-40, ay=-40,
    )
    return fig


def plot_response(d: float, vin: float, inductor: float,
                  capacitor: float, resistor: float,
                  mode: str) -> tuple:
    """
    Compute line-to-output & control-to-output step responses and return
    Plotly figures along with the raw time/response arrays and TF objects.

    Returns
    -------
    (fig_gvg, fig_gvd, sys_g, sys_d, tg, yg, td, yd)
    """
    if mode not in _GVG_FUNCS:
        raise ValueError(f"Unknown converter mode: {mode!r}")

    sys_g = _GVG_FUNCS[mode](d, vin, inductor, capacitor, resistor)
    sys_d = _GVD_FUNCS[mode](d, vin, inductor, capacitor, resistor)

    tg, yg = ct.step_response(sys_g)
    td, yd = ct.step_response(sys_d)

    yg_ss = float(yg[-1])
    yd_ss = float(yd[-1])

    fig_gvg = _build_figure(
        tg, yg, yg_ss,
        title=f'Line-to-Output Transient Response: Mode-{mode}, Vin-{vin}, D-{d}',
        trace_name='Transient Response Gvg(s)',
    )
    fig_gvd = _build_figure(
        td, yd, yd_ss,
        title=f'Control-to-Output Transient Response: Mode-{mode}, Vin-{vin}, D-{d}',
        trace_name='Transient Response Gvd(s)',
    )

    return fig_gvg, fig_gvd, sys_g, sys_d, tg, yg, td, yd
