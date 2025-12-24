from ConverterPackage import bckbstlc, bucklc, boostlc
from fpdf import FPDF
import os
import sys
import base64
import logging
import tempfile
import numpy as np
import pandas as pd
import plotly.io as pio
import tf_response as tfr
import streamlit as st
pio.renderers.default = "browser"


# Setup logging
logger_str: str = """[%(asctime)s %(name)s] %(levelname)s: %(module)s : %(message)s"""

log_dir: str = "Streamlit_logs"
api_log_path = os.path.join(log_dir, "StreamlitApp-logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logger_str,
    handlers=[
        logging.FileHandler(api_log_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('StreamlitApp')


# Use Streamlit session state for history
if "history_df" not in st.session_state:
    st.session_state["history_df"] = pd.DataFrame(columns=["Specs", "line_to_op_resp", "ctrl_to_op_resp"])

def save_to_history(specs: str, line_plot, ctrl_plot) -> None:
    # Use session state to persist history
    st.session_state["history_df"] = pd.concat([
        st.session_state["history_df"],
        pd.DataFrame({
            "Specs": [specs],
            "line_to_op_resp": [line_plot],
            "ctrl_to_op_resp": [ctrl_plot]
        })
    ], ignore_index=True)


def download_history_as_pdf_with_plots():
    if st.session_state["history_df"].empty:
        st.warning("No calculation history to download.")
        return

    with st.spinner("Generating PDF, please wait..."):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)

        for idx, row in st.session_state["history_df"].iterrows():
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "DC-DC Converter Calculation History", ln=True, align="C")
            pdf.cell(0, 8, f"Calculation {idx + 1}", ln=True, align="C")
            pdf.set_font("Arial", size=10)
            # Add specs text, line by line
            for line in row["Specs"].splitlines():
                pdf.multi_cell(0, 6, line)
            pdf.ln(2)

            # Save plots as temporary images and add to PDF
            for plot, title in zip(
                [row["line_to_op_resp"], row["ctrl_to_op_resp"]],
                ["Line to Output Response", "Control to Output Response"]
            ):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as imgfile:
                    plot.write_image(imgfile.name, format="png", scale=2)
                    # pio.kaleido.write_image(plot, file=imgfile.name, format="png", scale=2)
                    pdf.ln(2)
                    pdf.set_font("Arial", "B", 12)
                    # pdf.cell(0, 6, title, ln=True, align="C")
                    pdf.image(imgfile.name, w=150)
                    pdf.ln(2)

        # Save PDF to a temporary file and offer download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            pdf.output(tmpfile.name)
            tmpfile.seek(0)
            st.download_button(
                label="Download Calculation History as PDF",
                data=tmpfile.read(),
                file_name="calculation_history.pdf",
                mime="application/pdf"
            )
        st.success("PDF generated successfully!", icon="✅")


def tf_to_latex(tf, var='s', name='H'):
    """Convert a control.TransferFunction to a LaTeX string."""
    def poly_to_latex(coeffs):
        terms = []
        order = len(coeffs) - 1
        for i, c in enumerate(coeffs):
            if abs(c) < 1e-12:
                continue
            power = order - i
            c_str = f"{c:.3g}" if abs(c) != 1 or power == 0 else ("-" if c == -1 else "")
            if power == 0:
                terms.append(f"{c_str}")
            elif power == 1:
                terms.append(f"{c_str}{var}")
            else:
                terms.append(f"{c_str}{var}^{{{power}}}")
        return " + ".join(terms) if terms else "0"
    num = tf.num[0][0] if hasattr(tf.num[0], '__iter__') else tf.num
    den = tf.den[0][0] if hasattr(tf.den[0], '__iter__') else tf.den
    num_str = poly_to_latex(num)
    den_str = poly_to_latex(den)
    return rf"{name}({var}) = \frac{{{num_str}}}{{{den_str}}}"


# Setup page
about: str = """# DC-DC Converter Specs Calculator Site
This is a web application built using Streamlit to calculate and visualize the specifications of various DC-DC converters including Buck, Boost, and Buck-Boost converters.
"""

st.set_page_config(
    page_title="DC-DC Converter Specs Calculator",
    page_icon="🚀",
    layout="wide",
    menu_items={
        "About": about
    }
)

st.title(body="Get Your DC-DC Converter Specs Calculated! 🚀")
st.markdown(body="This application helps you calculate specifications and visualize the steady state response of *Buck*, *Boost*, and *Buck-Boost* converters based on your input parameters.")


# User input section
st.markdown("-----")
st.header("Please fill in the details below to get the converter specifications & plots:")
st.markdown("-----")

# Select dc-dc converter type
converter_type: str = str(st.pills(
    "Select DC-DC Converter Type",
    options=["Buck", "Boost", "BuckBoost"],
    default="Buck",
    help="Choose the DC-DC converter type you want to analyze.",
))

# Input Voltage
vin: float = st.number_input(
    label="Input Voltage (Vin) in Volts",
    min_value=0.1,
    max_value=1000.0,
    value=12.0,
    step=1.0,
    help="Enter the input voltage for the converter in Volts."
)

# Output Voltage
vo: float = st.number_input(
    label="Output Voltage (Vo) in Volts",
    min_value=0.1,
    max_value=1000.0,
    value=5.0,
    step=1.0,
    help="Enter the desired output voltage for the converter in Volts."
)

# Output Resistance
rload: float = st.number_input(
    label="Load Resistance (Rload) in Ohms",
    min_value=0.0,
    max_value=100000.0,
    value=10.0,
    step=1.0,
    help="Enter the load resistance for the converter in Ohms."
)

# Operating Frequency
fsw: float = st.number_input(
    label="Switching Frequency (fsw) in Hz",
    min_value=0.0,
    max_value=1000000.0,
    value=10000.0,
    step=100.0,
    help="Enter the switching frequency for the converter in Hertz."
)

# Percentage Inductor Ripple Current
ripl_crnt: float = st.number_input(
    label="Percentage i/p Inductor Ripple Current (ΔIL) in Amperes",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
    help="Enter the desired percentage i/p inductor ripple current for the converter in Amperes. E.g.: 10 for 10%"
)

# Percentage Output Voltage Ripple
ripl_vout: float = st.number_input(
    label="Percentage o/p Voltage Ripple (ΔVo) in Volts",
    min_value=0.0,
    max_value=100.0,
    value=1.0,
    step=0.1,
    help="Enter the desired percentage o/p voltage ripple for the converter in Volts. E.g.: 1 for 1%"
)

# Columns for buttons
col1, col2 = st.columns(2, gap="small", width=700)

with col1:
    # Calculate button
    cal_btn = st.button(label="**Calculate Converter Specs & Plot Response**")
with col2:
    # History button
    hist_btn = st.button(label="**Show Calculation History**")

if cal_btn:
    try:
        assert vo < vin if converter_type in ["Buck", "BuckBoost"] else vo > vin, "Output voltage must be less than input voltage for Buck and Buck-Boost converters, and greater for Boost converters."
        assert rload > 0, "Load resistance must be greater than zero."
        assert fsw > 0, "Switching frequency must be greater than zero."
        assert 0 < ripl_crnt < 100, "Percentage i/p inductor ripple current must be between 0 and 100."
        assert 0 < ripl_vout < 100, "Percentage o/p voltage ripple must be between 0 and 100."
        
        # Perform calculations based on converter type
        # duty_cycle: float
        # op_crnt: float 
        # ind_crnt: float
        # ip_crnt: float
        # ip_power: float
        # op_power: float
        # crt_ind: str
        # crt_ind_rpl_crnt: float
        # ind_ripl_crnt: str
        # ind: str
        # maxind_crnt: float
        # minind_crnt: float
        # cap: str
        # esr: float
        match converter_type:
            case "Buck":
                duty_cycle: float = bucklc.bck_duty_cycle(vo, vin)
                op_crnt: float = np.round((vo / rload), decimals=3)
                ip_crnt: float = bucklc.bck_ind_current(duty_cycle, op_crnt)
                ind_crnt: float = op_crnt
                ip_power: float = np.round((vin * ip_crnt), decimals=3)
                op_power: float = np.round((vo * op_crnt), decimals=3)
                crt_ind: str = bucklc.bck_cr_ind(duty_cycle, rload, fsw)
                crt_ind_rpl_crnt: float = bucklc.bck_ind_ripl_(vo, duty_cycle, fsw, crt_ind)
                ind_ripl_crnt: str = bucklc.bck_ripl_current(ind_crnt, ripl_crnt)
                ind: str = bucklc.bck_cont_ind(vo, duty_cycle, fsw, ind_ripl_crnt)
                maxind_crnt: float = np.round((ind_crnt + (float(ind_ripl_crnt) / 2)), decimals=3)
                minind_crnt: float = np.round((ind_crnt - (float(ind_ripl_crnt) / 2)), decimals=3)
                cap: str = bucklc.bck_cap_val(duty_cycle, ind, ripl_vout, fsw)
                esr: float = bucklc.Esr(ripl_vout, vo, maxind_crnt)
            case "Boost":
                duty_cycle: float = boostlc.bst_duty_cycle(vo, vin)
                op_crnt: float = np.round((vo / rload), decimals=3)
                ind_crnt: float = boostlc.bst_ind_current(duty_cycle, op_crnt)
                ip_crnt: float = ind_crnt
                ip_power: float = np.round((vin * ip_crnt), decimals=3)
                op_power: float = np.round((vo * op_crnt), decimals=3)
                crt_ind: str = boostlc.bst_cr_ind(duty_cycle, rload, fsw)
                crt_ind_rpl_crnt: float = boostlc.bst_ind_ripl_(vin, duty_cycle, fsw, crt_ind)
                ind_ripl_crnt: str = boostlc.bst_ripl_current(ind_crnt, ripl_crnt)
                ind: str = boostlc.bst_cont_ind(vin, duty_cycle, fsw, ind_ripl_crnt)
                maxind_crnt: float = np.round((ind_crnt + (float(ind_ripl_crnt) / 2)), decimals=3)
                minind_crnt: float = np.round((ind_crnt - (float(ind_ripl_crnt) / 2)), decimals=3)
                cap: str = boostlc.bst_cap_val(duty_cycle, rload, ripl_vout, fsw)
                esr: float = boostlc.Esr(ripl_vout, vo, maxind_crnt)
            case "BuckBoost":
                duty_cycle: float = bckbstlc.bckbst_duty_cycle(vo, vin)
                op_crnt: float = np.round((vo / rload), decimals=3)
                ind_crnt: float = bckbstlc.bckbst_ind_current(duty_cycle, op_crnt)
                ip_crnt: float = np.round((ind_crnt * duty_cycle), decimals=3)
                ip_power: float = np.round((vin * ip_crnt), decimals=3)
                op_power: float = np.round((vo * op_crnt), decimals=3)
                crt_ind: str = bckbstlc.bckbst_cr_ind(duty_cycle, rload, fsw)
                crt_ind_rpl_crnt: float = bckbstlc.bckbst_ind_ripl_(vin, duty_cycle, fsw, crt_ind)
                ind_ripl_crnt: str = bckbstlc.bckbst_ripl_current(ip_crnt, ripl_crnt)
                ind: str = bckbstlc.bckbst_cont_ind(vin, duty_cycle, fsw, ind_ripl_crnt)
                maxind_crnt: float = np.round((ind_crnt + (float(ind_ripl_crnt) / 2)), decimals=3)
                minind_crnt: float = np.round((ind_crnt - (float(ind_ripl_crnt) / 2)), decimals=3)
                cap: str = bckbstlc.bckbst_cap_val(duty_cycle, rload, ripl_vout, fsw)
                esr: float = bckbstlc.Esr(ripl_vout, vo, maxind_crnt)

        with st.spinner("Calculating specs and plotting responses, please wait..."):
            # Plot response
            line_to_op_resp, ctrl_to_op_resp, sys_g, sys_d = tfr.plot_response(duty_cycle, vin, float(ind), float(cap), rload, converter_type)
            gvg_latex = tf_to_latex(sys_g, name='G_{vg}')
            gvd_latex = tf_to_latex(sys_d, name='G_{vd}')

            # Display results
            op_string = f"""
            #### Converter Parameters
            \tMode = {converter_type} Converter
            \tVin = {vin}V
            \tVo = {vo}V
            \tR = {rload}Ohms
            \tfsw = {fsw}Hz
            \tIrp = {ripl_crnt}%
            \tVrp = {ripl_vout}%

            #### Converter Parameters
            \tDuty Cycle = {duty_cycle}
            \tPower Input = {ip_power}W
            \tPower output = {op_power}W
            \tOutput Current = {op_crnt}A
            \tInductor Current = {ind_crnt}A
            \tInput Current = {ip_crnt}A
            \tCritical Inductance Value(Lcr)= {crt_ind}H
            \tRipple Current due to Lcr = {crt_ind_rpl_crnt}A
            \tContinuous Conduction Inductor Value (L) = {ind}H
            \tRipple Current due to L = {ind_ripl_crnt}A
            \tMaximum inductor ripple current = {maxind_crnt}A
            \tMinimum inductor ripple current = {minind_crnt}A
            \tOutput Capacitor = {cap}F
            \tCapacitor ESR = {esr}Ohms

            #### Transfer Functions

            $$ {gvg_latex} $$

            $$ {gvd_latex} $$
            """

            # print('Line-to-Output Transfer Function: ', sys_g['num'], '/', sys_g['den'])
            st.write(op_string)
            st.plotly_chart(line_to_op_resp)
            st.plotly_chart(ctrl_to_op_resp)

        logger.info(f"Calculated specs for {converter_type} converter with Vin={vin}V, Vo={vo}V, Rload={rload}Ohms, fsw={fsw}Hz, Irp={ripl_crnt}%, Vrp={ripl_vout}%")

        # pio.kaleido.write_image(line_to_op_resp, file="line_to_op_resp.png", format="png")
        # pio.kaleido.write_image(ctrl_to_op_resp, file="ctrl_to_op_resp.png", format="png")

        # Save to history
        save_to_history(op_string, line_to_op_resp, ctrl_to_op_resp)
    except AssertionError as ae:
        st.error(f"Input Error: {ae}")
        logger.error(f"Input Error: {ae}")
    except Exception as e:
        st.error(f"An error occurred during calculation: {e}")
        logger.error(f"Calculation Error: {e}")


if hist_btn:
    st.markdown("## Calculation History")
    if st.session_state["history_df"].empty:
        st.markdown("No calculations performed yet.")
    else:
        # for index, row in st.session_state["history_df"].iterrows():
        #     st.markdown(f"### Calculation {index + 1}")
        #     st.markdown(row["Specs"])
        #     st.plotly_chart(row["line_to_op_resp"])
        #     st.plotly_chart(row["ctrl_to_op_resp"])

        try:
            for index, row in st.session_state["history_df"].iterrows():
                col_specs, col_line, col_ctrl = st.columns(3, vertical_alignment="center", border=True)
                with col_specs:
                    st.markdown(f"### Calculation {index + 1}")
                    st.markdown(row["Specs"])
                with col_line:
                        st.plotly_chart(row["line_to_op_resp"])
                with col_ctrl:
                        st.plotly_chart(row["ctrl_to_op_resp"])
        except Exception as e:
            st.error(f"An error occurred while displaying history: {e}")
            logger.error(f"History Display Error: {e}")
            
        download_history_as_pdf_with_plots()

def get_image_base64(image_path):
    """Read image file."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# st.markdown("""---""")

# Get the base64 string of the image
img_base64 = get_image_base64("assets/pranay-jolly-sq.jpg")

# Create the HTML for the circular image
st.markdown(
    """
    ------
    <style>
        a.author {
            text-decoration: none;
            color: #F14848;
        }
        a.author:hover {
            text-decoration: none;
            color: #14a3ee;
        }
    </style>
    <p><em>Created with</em> ❤️‍🔥 <em>by <a class='author' href='https://pranayjagtap.netlify.app' rel=noopener noreferrer' target='_blank'><b>Pranay Jagtap</b></a></em></p>
    """,
    unsafe_allow_html=True
)
html_code = """
<style>
    .circular-image {
        width: 125px;
        height: 125px;
        border-radius: 55%;
        overflow: hidden;
        display: inline-block;
    }
    .circular-image img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .author-headline {
        color: #14a3ee
    }
</style>
""" + f"""
<div class="circular-image">
    <a href="https://pranayjagtap.netlify.app" target="_blank" rel="noopener noreferrer">
        <img src="data:image/jpeg;base64,{img_base64}" alt="Pranay Jagtap">
    </a>
</div>
<p class=author-headline><b>Machine Learning Enthusiast | Electrical Engineer<br>📍Nagpur, Maharashtra, India<b></p>
"""

# Display the circular image
st.markdown(html_code, unsafe_allow_html=True)
# End of the Streamlit app code