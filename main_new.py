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
import matplotlib.pyplot as plt
from io import BytesIO

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
    st.session_state["history_df"] = pd.DataFrame(columns=[
        "Specs", "line_to_op_tf", "line_to_op_resp", "ctrl_to_op_tf", "ctrl_to_op_resp",
        "sys_g_obj", "sys_d_obj"  # Store actual transfer function objects
    ])

if "show_history" not in st.session_state:
    st.session_state["show_history"] = False

if "pdf_data" not in st.session_state:
    st.session_state["pdf_data"] = None


def save_to_history(specs: str, gvg_latex: str, gvd_latex: str, line_plot, ctrl_plot, sys_g, sys_d) -> None:
    """Save calculation results to history including transfer function objects"""
    st.session_state["history_df"] = pd.concat([
        st.session_state["history_df"],
        pd.DataFrame({
            "Specs": [specs],
            "line_to_op_tf": [gvg_latex],
            "line_to_op_resp": [line_plot],
            "ctrl_to_op_tf": [gvd_latex],
            "ctrl_to_op_resp": [ctrl_plot],
            "sys_g_obj": [sys_g],  # Store the actual system object
            "sys_d_obj": [sys_d]   # Store the actual system object
        })
    ], ignore_index=True)
    
    st.session_state["pdf_data"] = None


def latex_to_image(latex_string, fontsize=12, dpi=300):
    """Convert LaTeX string to PNG image bytes"""
    fig = plt.figure(figsize=(10, 0.8))
    fig.patch.set_facecolor('white')
    
    # Render LaTeX
    fig.text(0.5, 0.5, f'${latex_string}$', 
             fontsize=fontsize, 
             ha='center', 
             va='center')
    
    # Save to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    
    return buf


def tf_to_latex(sys, var='s', name='H'):
    """Format transfer function for LaTeX display with proper sign handling"""
    num = sys.num[0][0]
    den = sys.den[0][0]
    
    def format_polynomial(coeffs):
        """Format polynomial coefficients with proper signs"""
        terms = []
        for i, coef in enumerate(coeffs):
            if abs(coef) < 1e-10:  # Skip near-zero coefficients
                continue
                
            power = len(coeffs) - i - 1
            
            # Format coefficient
            coef_val = abs(coef)
            if coef_val == 1 and power > 0:
                coef_str = ""
            else:
                coef_str = f"{coef_val:.4g}"
            
            # Format power
            if power == 0:
                term = f"{coef_val:.4g}"
            elif power == 1:
                term = f"{coef_str}{var}"
            else:
                term = f"{coef_str}{var}^{{{power}}}"
            
            # Add sign
            if len(terms) == 0:  # First term
                if coef < 0:
                    term = "-" + term
            else:  # Subsequent terms
                if coef < 0:
                    term = " - " + term
                else:
                    term = " + " + term
            
            terms.append(term)
        
        return "".join(terms) if terms else "0"
    
    num_latex = format_polynomial(num)
    den_latex = format_polynomial(den)
    
    return rf"{name}({var}) = \frac{{{num_latex}}}{{{den_latex}}}"


def download_history_as_pdf_with_plots():
    """Generate PDF with all calculation history including transfer functions"""
    if st.session_state["history_df"].empty:
        st.info("No calculation history to download.")
        return

    if st.button("Generate PDF Report", icon="📄"):
        with st.spinner("Generating PDF, please wait..."):
            try:
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
        
                for idx, row in st.session_state["history_df"].iterrows():
                    pdf.add_page()
            
                    # Title
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, "DC-DC Converter Analysis Report", ln=True, align="R")
                    pdf.ln(2)
                    
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 8, f"Calculation #{idx + 1}", ln=True, align="C")
                    pdf.ln(5)
                    
                    # Specs section
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 8, "Converter Specifications:", ln=True)
                    pdf.set_font("Arial", "", 9)
                    
                    # Add specs text, line by line
                    for line in row["Specs"].splitlines():
                        if line.strip():  # Skip empty lines
                            clean_line = line.strip().encode('latin-1', 'replace').decode('latin-1')
                            pdf.multi_cell(0, 6.5, clean_line)
                    pdf.ln(20)
                    
                    # Transfer Functions Section
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 8, "Transfer Functions:", ln=True)
                    pdf.ln(2)
                    
                    # Line-to-Output Transfer Function
                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(0, 6, "Line-to-Output Transfer Function:", ln=True)
                    
                    # Convert LaTeX to image and add to PDF
                    try:
                        gvg_img = latex_to_image(row["line_to_op_tf"], fontsize=11)
                        fd, tmp_gvg = tempfile.mkstemp(suffix=".png")
                        os.close(fd)
                        with open(tmp_gvg, "wb") as f:
                            f.write(gvg_img.getvalue())
                        pdf.image(tmp_gvg, x=11, w=110)
                        os.unlink(tmp_gvg)
                    except Exception as e:
                        logger.error(f"Error rendering Gvg LaTeX: {e}")
                        pdf.set_font("Arial", "", 8)
                        pdf.multi_cell(0, 5, f"Gvg(s): {str(row['sys_g_obj'])}")
                    
                    pdf.ln(3)
                    
                    # Control-to-Output Transfer Function
                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(0, 6, "Control-to-Output Transfer Function:", ln=True)
                    
                    try:
                        gvd_img = latex_to_image(row["ctrl_to_op_tf"], fontsize=11)
                        fd, tmp_gvd = tempfile.mkstemp(suffix=".png")
                        os.close(fd)
                        with open(tmp_gvd, "wb") as f:
                            f.write(gvd_img.getvalue())
                        pdf.image(tmp_gvd, x=11, w=110)
                        os.unlink(tmp_gvd)
                    except Exception as e:
                        logger.error(f"Error rendering Gvd LaTeX: {e}")
                        pdf.set_font("Arial", "", 8)
                        pdf.multi_cell(0, 5, f"Gvd(s): {str(row['sys_d_obj'])}")
                    
                    pdf.ln(25)
                    
                    # Response Plots
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 8, "Transient Response Plots:", ln=True)
                    pdf.ln(2)
                    
                    # Line-to-Output Response
                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(0, 6, "Line-to-Output Response:", ln=True)
                    try:
                        fd, tmp_img_path = tempfile.mkstemp(suffix=".png")
                        os.close(fd)
                        row["line_to_op_resp"].write_image(tmp_img_path, format="png", width=800, height=500, scale=2)
                        pdf.image(tmp_img_path, x=10, w=190)
                        os.unlink(tmp_img_path)
                    except Exception as e:
                        logger.error(f"Plotly image error: {e}")
                        pdf.set_font("Arial", "", 8)
                        pdf.multi_cell(0, 5, "[Plot image generation failed or not supported in this environment]")
                    pdf.ln(3)
                    
                    # Control-to-Output Response
                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(0, 6, "Control-to-Output Response:", ln=True)
                    try:
                        fd, tmp_img_path = tempfile.mkstemp(suffix=".png")
                        os.close(fd)
                        row["ctrl_to_op_resp"].write_image(tmp_img_path, format="png", width=800, height=500, scale=2)
                        pdf.image(tmp_img_path, x=10, w=190)
                        os.unlink(tmp_img_path)
                    except Exception as e:
                        logger.error(f"Plotly image error: {e}")
                        pdf.set_font("Arial", "", 8)
                        pdf.multi_cell(0, 5, "[Plot image generation failed or not supported in this environment]")
                
                fd, tmpfile_path = tempfile.mkstemp(suffix=".pdf")
                os.close(fd)
                pdf.output(tmpfile_path)
                
                with open(tmpfile_path, "rb") as f:
                    st.session_state["pdf_data"] = f.read()
                
                os.unlink(tmpfile_path)
                
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")
                logger.error(f"PDF generation failed: {e}")

    if st.session_state.get("pdf_data") is not None:
        st.download_button(
            label="📥 Download Calculation History as PDF",
            data=st.session_state["pdf_data"],
            file_name="dc_dc_converter_analysis_history.pdf",
            mime="application/pdf",
            type="primary"
        )


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
    label="Input Voltage (V)",
    min_value=0.0,
    max_value=1000.0,
    value=12.0,
    step=0.1,
    help="Enter the input voltage for the converter."
)

# Output Voltage
vo: float = st.number_input(
    label="Output Voltage (V)",
    min_value=0.0,
    max_value=1000.0,
    value=5.0,
    step=0.1,
    help="Enter the desired output voltage."
)

# Load Resistance
rload: float = st.number_input(
    label="Load Resistance (Ω)",
    min_value=0.1,
    max_value=10000.0,
    value=10.0,
    step=0.1,
    help="Enter the load resistance value."
)

# Switching Frequency
fsw: float = st.number_input(
    label="Switching Frequency (Hz)",
    min_value=1000.0,
    max_value=1000000.0,
    value=100000.0,
    step=1000.0,
    help="Enter the switching frequency."
)

# Ripple Current Percentage
ripl_crnt: float = st.number_input(
    label="Ripple Current (%)",
    min_value=0.1,
    max_value=100.0,
    value=30.0,
    step=0.1,
    help="Enter the acceptable ripple current percentage."
)

# Ripple Voltage Percentage
ripl_vout: float = st.number_input(
    label="Output Voltage Ripple (%)",
    min_value=0.1,
    max_value=100.0,
    value=1.0,
    step=0.1,
    help="Enter the acceptable output voltage ripple percentage."
)

# Buttons
col1, col2 = st.columns(2)
with col1:
    calc_btn = st.button("Calculate Specs", type="primary", use_container_width=True)
with col2:
    hist_btn = st.button("Toggle History", use_container_width=True)

if hist_btn:
    st.session_state["show_history"] = not st.session_state["show_history"]

# Calculate and display results
if calc_btn:
    st.session_state["show_history"] = False
    try:
        # Variable declarations
        duty_cycle: float
        ip_crnt: float
        op_crnt: float
        ind_crnt: float
        ip_power: float
        op_power: float
        crt_ind: str
        crt_ind_rpl_crnt: float
        ind_ripl_crnt: str
        ind: str
        maxind_crnt: float
        minind_crnt: float
        cap: str
        esr: float
        
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
            line_to_op_resp, ctrl_to_op_resp, sys_g, sys_d = tfr.plot_response(
                duty_cycle, vin, float(ind), float(cap), rload, converter_type
            )
            
            # Generate LaTeX equations
            gvg_latex = tf_to_latex(sys_g, name='G_{vg}')
            gvd_latex = tf_to_latex(sys_d, name='G_{vd}')

            # Display results
            op_string = f"""
#### Input Parameters
- Mode = {converter_type} Converter
- Vin = {vin}V
- Vo = {vo}V
- R = {rload}Ohm
- fsw = {fsw}Hz
- Irp = {ripl_crnt}%
- Vrp = {ripl_vout}%

#### Calculated Parameters
- Duty Cycle = {duty_cycle}
- Power Input = {ip_power}W
- Power output = {op_power}W
- Output Current = {op_crnt}A
- Inductor Current = {ind_crnt}A
- Input Current = {ip_crnt}A
- Critical Inductance Value (Lcr) = {crt_ind}H
- Ripple Current due to Lcr = {crt_ind_rpl_crnt}A
- Continuous Conduction Inductor Value (L) = {ind}H
- Ripple Current due to L = {ind_ripl_crnt}A
- Maximum inductor ripple current = {maxind_crnt}A
- Minimum inductor ripple current = {minind_crnt}A
- Output Capacitor = {cap}F
- Capacitor ESR = {esr}Ohm
"""

            st.write(op_string)
            
            # Display Transfer Functions
            st.markdown("#### Transfer Functions")
            col_tf1, col_tf2 = st.columns(2)
            
            with col_tf1:
                st.markdown("**Line-to-Output Transfer Function:**")
                st.latex(gvg_latex)
                st.plotly_chart(line_to_op_resp, use_container_width=True)
                
            with col_tf2:
                st.markdown("**Control-to-Output Transfer Function:**")
                st.latex(gvd_latex)
                st.plotly_chart(ctrl_to_op_resp, use_container_width=True)

        logger.info(f"Calculated specs for {converter_type} converter with Vin={vin}V, Vo={vo}V")

        # Save to history - now includes sys_g and sys_d objects
        save_to_history(op_string, gvg_latex, gvd_latex, line_to_op_resp, ctrl_to_op_resp, sys_g, sys_d)
        
    except AssertionError as ae:
        st.error(f"Input Error: {ae}")
        logger.error(f"Input Error: {ae}")
    except Exception as e:
        st.error(f"An error occurred during calculation: {e}")
        logger.error(f"Calculation Error: {e}")


# History Display
if st.session_state["show_history"]:
    st.markdown("## Calculation History")
    if st.session_state["history_df"].empty:
        st.info("No calculations performed yet.")
    else:
        try:
            for index, row in st.session_state["history_df"].iterrows():
                with st.expander(f"📊 Calculation #{index + 1}", expanded=False):
                    # Specs
                    st.markdown("### Specifications")
                    st.markdown(row["Specs"])
                    
                    # Transfer Functions
                    st.markdown("### Transfer Functions")
                    col_tf1, col_tf2 = st.columns(2)
                    
                    with col_tf1:
                        st.markdown("**Line-to-Output:**")
                        st.latex(row["line_to_op_tf"])
                        
                    with col_tf2:
                        st.markdown("**Control-to-Output:**")
                        st.latex(row["ctrl_to_op_tf"])
                    
                    # Response Plots
                    st.markdown("### Response Plots")
                    col_plot1, col_plot2 = st.columns(2)
                    
                    with col_plot1:
                        st.plotly_chart(row["line_to_op_resp"], use_container_width=True)
                        
                    with col_plot2:
                        st.plotly_chart(row["ctrl_to_op_resp"], use_container_width=True)
                        
        except Exception as e:
            st.error(f"An error occurred while displaying history: {e}")
            logger.error(f"History Display Error: {e}")
            
        # PDF Export Button
        st.markdown("---")
        download_history_as_pdf_with_plots()


def get_image_base64(image_path):
    """Read image file."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# Footer
st.markdown("""---""")

# Get the base64 string of the image
try:
    img_base64 = get_image_base64("assets/pranay-jolly-sq.jpg")
    
    # Create the HTML for the circular image
    st.markdown(
        """
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
except Exception as e:
    logger.warning(f"Could not load author image: {e}")

# End of the Streamlit app code