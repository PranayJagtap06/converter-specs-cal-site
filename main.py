from ConverterPackage import bckbstlc, bucklc, boostlc
from dcm_opt_ind import dcm_opt_ind
from fpdf import FPDF
import os
import sys
import base64
import logging
import tempfile
import numpy as np
import plotly.io as pio
import tf_response as tfr
import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
from functools import lru_cache
from typing import Optional

pio.renderers.default = "browser"


# Setup logging
logger_str: str = """[%(asctime)s %(name)s] %(levelname)s: %(module)s : %(message)s"""

log_dir: str = "gradio_logs"
api_log_path = os.path.join(log_dir, "gradioapp-logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logger_str,
    handlers=[
        logging.FileHandler(api_log_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('gradioapp')

IDEL_STRING = """
### Input Parameters
- Mode = converter_type Converter
- Vin = input_voltage V
- Vo = output_voltage V
- R = output_resistance Ohm
- fsw = frequency Hz
- Irp = ripl_current %
- Vrp = ripl_voltage %

### Calculated Parameters
- Duty Cycle = duty_cycle
- Power Input = ip_power W
- Power output = op_power W
- Output Current = op_crnt A
- Inductor Current = ind_crnt A
- Input Current = ip_crnt A
- Critical Inductance Value (Lcr) = crt_ind H
- Ripple Current due to Lcr = crt_ind_rpl_crnt A
- Continuous Conduction Inductor Value (L) = ind H
- Ripple Current due to L = ind_ripl_crnt A
- Maximum inductor ripple current = maxind_crnt A
- Minimum inductor ripple current = minind_crnt A
- Output Capacitor = cap F
- Capacitor ESR = esr Ohm

### Transfer Functions
$$ line-to-output\ transfer\ function $$

$$ control-to-output\ transfer\ function $$
"""


def latex_to_image(latex_string, fontsize=12, dpi=300):
    """Convert LaTeX string to PNG image bytes"""
    fig = plt.figure(figsize=(10, 0.8), dpi=dpi)
    fig.patch.set_facecolor('white')

    try:
        fig.text(0.5, 0.5, f'${latex_string}$',
                 fontsize=fontsize,
                 ha='center',
                 va='center')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none', pad_inches=0.1)
        buf.seek(0)
        return buf
    finally:
        plt.close(fig)


def tf_to_latex(sys, var='s', name='H') -> str:
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


def generate_history_pdf(specs_hist: list):
    """Generate PDF from calculation history with all equations and plots"""
    if not specs_hist:
        raise gr.Error("No calculation history to download.")
    
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        for idx, row in enumerate(reversed(specs_hist)):
            original_idx = len(specs_hist) - 1 - idx
            pdf.add_page()
            
            # Title
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "DC-DC Converter Analysis Report", ln=True, align="R")
            pdf.ln(2)
            
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, f"Calculation #{original_idx + 1}", ln=True, align="C")
            pdf.ln(5)
            
            # Specifications
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, "Converter Specifications:", ln=True)
            pdf.set_font("Arial", "", 9)
            
            # Parse specs and add to PDF
            for line in row["Specs"].splitlines():
                if line.strip().startswith("### Transfer Functions"):
                    break  # Stop writing raw text since equations get rendered as beautiful images below
                elif line.strip():  # Skip empty lines
                    clean_line = line.strip().encode('latin-1', 'replace').decode('latin-1')
                    pdf.multi_cell(0, 6.5, clean_line)
            
            pdf.ln(20)
            
            # Transfer Functions
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, "Transfer Functions:", ln=True)
            pdf.ln(2)
            
            # Line-to-Output Transfer Function
            pdf.set_font("Arial", "B", 10)
            # pdf.cell(0, 6, "Line-to-Output Transfer Function:", ln=True)
            
            # Convert LaTeX to image
            try:
                gvg_img = latex_to_image(row["line_to_op_tf"], fontsize=11)
                fd, tmp_gvg = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                with open(tmp_gvg, "wb") as f:
                    f.write(gvg_img.getvalue())
                pdf.image(tmp_gvg, x=11, w=116, h=16)
                os.unlink(tmp_gvg)
            except Exception as e:
                logger.error(f"Error rendering Gvg LaTeX: {e}")
                pdf.set_font("Arial", "", 8)
                pdf.multi_cell(0, 5, f"Gvg(s): {str(row['sys_g_obj'])}")
            
            pdf.ln(3)
            
            # Control-to-Output Transfer Function
            pdf.set_font("Arial", "B", 10)
            # pdf.cell(0, 6, "Control-to-Output Transfer Function:", ln=True)
            
            try:
                gvd_img = latex_to_image(row["ctrl_to_op_tf"], fontsize=11)
                fd, tmp_gvd = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                with open(tmp_gvd, "wb") as f:
                    f.write(gvd_img.getvalue())
                pdf.image(tmp_gvd, x=11, w=116, h=16)
                os.unlink(tmp_gvd)
            except Exception as e:
                logger.error(f"Error rendering Gvd LaTeX: {e}")
                pdf.set_font("Arial", "", 8)
                pdf.multi_cell(0, 5, f"Gvd(s): {str(row['sys_d_obj'])}")
            
            pdf.ln(25)
            
            # DCM Inductor Value Optimization
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 6, "DCM Inductor Value Optimization:", ln=True)
            try:
                fd, tmp_img_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                
                # Hybrid approach: Use Matplotlib strictly for PDF export
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(row["lc"], row["iLmin"], color='#1f77b4', label='iLmin')
                ax.plot(row["lc"], row["iLmax"], color='#ff7f0e', label='iLmax')
                ax.set_title("Inductor Value Optimization for DCM operation")
                ax.set_xlabel("Inductor (uH)")
                ax.set_ylabel("Inductor Current (A)")
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                ax.grid(True, linestyle=':', alpha=0.7)
                ax.legend()
                fig.savefig(tmp_img_path, format="png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                
                pdf.image(tmp_img_path, x=10, w=190)
                os.unlink(tmp_img_path)
            except Exception as e:
                logger.error(f"Matplotlib image error: {e}")
                pdf.set_font("Arial", "", 8)
                pdf.multi_cell(0, 5, "[Plot image generation failed or not supported in this environment]")
            pdf.ln(3)
            
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
                
                # Hybrid approach: Use Matplotlib strictly for PDF export
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(row["tg"], row["yg"], color='#1f77b4', label='Transient Response')
                ax.axhline(y=row["yg"][-1], color='#ff7f0e', linestyle='--', label='Steady State Value')
                ax.set_title("Line-to-Output Transient Response")
                ax.set_xlabel("Time (sec)")
                ax.set_ylabel("Response (volts)")
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                ax.grid(True, linestyle=':', alpha=0.7)
                ax.legend()
                fig.savefig(tmp_img_path, format="png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                
                pdf.image(tmp_img_path, x=10, w=190)
                os.unlink(tmp_img_path)
            except Exception as e:
                logger.error(f"Matplotlib image error: {e}")
                pdf.set_font("Arial", "", 8)
                pdf.multi_cell(0, 5, "[Plot image generation failed or not supported in this environment]")
            pdf.ln(3)
            
            # Control-to-Output Response
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 6, "Control-to-Output Response:", ln=True)
            try:
                fd, tmp_img_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(row["td"], row["yd"], color='#2ca02c', label='Transient Response')
                ax.axhline(y=row["yd"][-1], color='#d62728', linestyle='--', label='Steady State Value')
                ax.set_title("Control-to-Output Transient Response")
                ax.set_xlabel("Time (sec)")
                ax.set_ylabel("Response (volts)")
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                ax.grid(True, linestyle=':', alpha=0.7)
                ax.legend()
                fig.savefig(tmp_img_path, format="png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                
                pdf.image(tmp_img_path, x=10, w=190)
                os.unlink(tmp_img_path)
            except Exception as e:
                logger.error(f"Matplotlib image error: {e}")
                pdf.set_font("Arial", "", 8)
                pdf.multi_cell(0, 5, "[Plot image generation failed or not supported in this environment]")
        
        # Save PDF
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, "DC_DC_Converter_Analysis_History.pdf")
        pdf.output(pdf_path)
        
        return pdf_path
    
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        gr.Error(f"Failed to generate PDF: {e}")


@lru_cache(maxsize=32)
def _calculate_core_specs(converter_type: str, input_voltage: float, output_voltage: float, output_resistance: float, frequency: float, ripl_current: float, ripl_voltage: float):
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
            duty_cycle = bucklc.bck_duty_cycle(output_voltage, input_voltage)
            op_crnt = np.round((output_voltage / output_resistance), decimals=3)
            ip_crnt = bucklc.bck_ind_current(duty_cycle, op_crnt)
            ind_crnt = op_crnt
            ip_power = np.round((input_voltage * ip_crnt), decimals=3)
            op_power = np.round((output_voltage * op_crnt), decimals=3)
            crt_ind = bucklc.bck_cr_ind(duty_cycle, output_resistance, frequency)
            crt_ind_rpl_crnt = bucklc.bck_ind_ripl_(
                output_voltage, duty_cycle, frequency, crt_ind)
            ind_ripl_crnt = bucklc.bck_ripl_current(
                ind_crnt, ripl_current)
            ind = bucklc.bck_cont_ind(
                output_voltage, duty_cycle, frequency, ind_ripl_crnt)
            maxind_crnt = np.round(
                (ind_crnt + (float(ind_ripl_crnt) / 2)), decimals=3)
            minind_crnt = np.round(
                (ind_crnt - (float(ind_ripl_crnt) / 2)), decimals=3)
            cap = bucklc.bck_cap_val(duty_cycle, ind, ripl_voltage, frequency)
            esr = bucklc.Esr(ripl_voltage, output_voltage, maxind_crnt)
        case "Boost":
            duty_cycle = boostlc.bst_duty_cycle(output_voltage, input_voltage)
            op_crnt = np.round((output_voltage / output_resistance), decimals=3)
            ind_crnt = boostlc.bst_ind_current(duty_cycle, op_crnt)
            ip_crnt = ind_crnt
            ip_power = np.round((input_voltage * ip_crnt), decimals=3)
            op_power = np.round((output_voltage * op_crnt), decimals=3)
            crt_ind = boostlc.bst_cr_ind(duty_cycle, output_resistance, frequency)
            crt_ind_rpl_crnt = boostlc.bst_ind_ripl_(
                input_voltage, duty_cycle, frequency, crt_ind)
            ind_ripl_crnt = boostlc.bst_ripl_current(
                ind_crnt, ripl_current)
            ind = boostlc.bst_cont_ind(
                input_voltage, duty_cycle, frequency, ind_ripl_crnt)
            maxind_crnt = np.round(
                (ind_crnt + (float(ind_ripl_crnt) / 2)), decimals=3)
            minind_crnt = np.round(
                (ind_crnt - (float(ind_ripl_crnt) / 2)), decimals=3)
            cap = boostlc.bst_cap_val(
                duty_cycle, output_resistance, ripl_voltage, frequency)
            esr = boostlc.Esr(ripl_voltage, output_voltage, maxind_crnt)
        case "BuckBoost":
            duty_cycle = bckbstlc.bckbst_duty_cycle(output_voltage, input_voltage)
            op_crnt = np.round((output_voltage / output_resistance), decimals=3)
            ind_crnt = bckbstlc.bckbst_ind_current(
                duty_cycle, op_crnt)
            ip_crnt = np.round((ind_crnt * duty_cycle), decimals=3)
            ip_power = np.round((input_voltage * ip_crnt), decimals=3)
            op_power = np.round((output_voltage * op_crnt), decimals=3)
            crt_ind = bckbstlc.bckbst_cr_ind(duty_cycle, output_resistance, frequency)
            crt_ind_rpl_crnt = bckbstlc.bckbst_ind_ripl_(
                input_voltage, duty_cycle, frequency, crt_ind)
            ind_ripl_crnt = bckbstlc.bckbst_ripl_current(
                ip_crnt, ripl_current)
            ind = bckbstlc.bckbst_cont_ind(
                input_voltage, duty_cycle, frequency, ind_ripl_crnt)
            maxind_crnt = np.round(
                (ind_crnt + (float(ind_ripl_crnt) / 2)), decimals=3)
            minind_crnt = np.round(
                (ind_crnt - (float(ind_ripl_crnt) / 2)), decimals=3)
            cap = bckbstlc.bckbst_cap_val(
                duty_cycle, output_resistance, ripl_voltage, frequency)
            esr = bckbstlc.Esr(ripl_voltage, output_voltage, maxind_crnt)

    # Plot response
    line_to_op_resp, ctrl_to_op_resp, sys_g, sys_d, tg, yg, td, yd = tfr.plot_response(
        duty_cycle, input_voltage, float(ind), float(cap), output_resistance, converter_type
    )
    dcm_opt_ind_fig, lc, iLmax, iLmin = dcm_opt_ind(input_voltage, output_voltage, op_crnt, 1/frequency, converter_type)

    # Generate LaTeX equations
    gvg_latex = tf_to_latex(sys_g, name='Line-to-Output\ Transfer\ Function\ G_{vg}')
    gvd_latex = tf_to_latex(sys_d, name='Contol-to-Output\ Transfer\ Function\ G_{vd}')

    # Display results
    op_string = f"""
### Input Parameters
- Mode = {converter_type} Converter
- Vin = {input_voltage}V
- Vo = {output_voltage}V
- R = {output_resistance}Ohm
- fsw = {frequency}Hz
- Irp = {ripl_current}%
- Vrp = {ripl_voltage}%

### Calculated Parameters
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

### Transfer Functions
$$ {gvg_latex} $$

$$ {gvd_latex} $$
"""
    return op_string, gvg_latex, gvd_latex, line_to_op_resp, ctrl_to_op_resp, dcm_opt_ind_fig, sys_g, sys_d, tg, yg, td, yd, lc, iLmax, iLmin


def dc_dc_converter(converter_type: str, input_voltage: float, output_voltage: float, output_resistance: float, frequency: float, ripl_current: float, ripl_voltage: float, specs_hist: list) -> Optional[tuple]:
    if converter_type in ["Buck", "BuckBoost"] and output_voltage >= input_voltage:
        raise gr.Error(f"Output voltage must be less than input voltage for {converter_type} converter.")
    if converter_type == "Boost" and output_voltage <= input_voltage:
        raise gr.Error("Output voltage must be greater than input voltage for Boost converter.")
        
    try:
        op_string, gvg_latex, gvd_latex, line_to_op_resp, ctrl_to_op_resp, dcm_opt_ind_fig, sys_g, sys_d, tg, yg, td, yd, lc, iLmax, iLmin = _calculate_core_specs(
            converter_type, input_voltage, output_voltage, output_resistance, frequency, ripl_current, ripl_voltage
        )
    except Exception as e:
        logger.error(f"Calculation Error: {e}")
        raise gr.Error(f"Calculation Error: {e}")
        
    new_row = {
        "Specs": op_string,
        "line_to_op_tf": gvg_latex,
        "line_to_op_resp": line_to_op_resp,
        "ctrl_to_op_tf": gvd_latex,
        "ctrl_to_op_resp": ctrl_to_op_resp,
        "dcm_opt_ind_fig": dcm_opt_ind_fig,
        "sys_g_obj": sys_g,
        "sys_d_obj": sys_d,
        "tg": tg,
        "yg": yg,
        "td": td,
        "yd": yd,
        "lc": lc,
        "iLmax": iLmax,
        "iLmin": iLmin
    }
    
    if specs_hist is None:
        specs_hist = []
        
    specs_hist.append(new_row)

    return (
        op_string,
        dcm_opt_ind_fig,
        gr.update(visible=True),
        line_to_op_resp,
        ctrl_to_op_resp,
        gr.update(visible=True),
        gr.update(visible=True),
        specs_hist
    )


def get_image_base64(image_path):
    """Read image file."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


description = "## Get Your DC-DC Converter Specs Calculated! 🚀\n This application helps you calculate specifications and visualize the steady state response of *Buck*, *Boost*, and *Buck-Boost* converters based on your input parameters."

with gr.Blocks(delete_cache=(86400, 86400)) as demo:        
    gr.Markdown("# 🦄 DC-DC Converter Specs Calculator")
    gr.Markdown(description)

    # Initialize State
    history_state = gr.State([])

    with gr.Row():
        with gr.Column(variant="panel"):
            ctype = gr.Dropdown(value="Buck", choices=["Buck", "Boost", "BuckBoost"], label="Select Converter Type", show_label=True, 
                                info="Choose the DC-DC converter type you want to analyze.", interactive=True)
            vin = gr.Number(value=12, placeholder="12", info="Enter the input voltage for the converter.",
                            minimum=0.0, maximum=1000.0, step=0.1, label="Input Voltage (V)", show_label=True)
            vo = gr.Number(value=5, placeholder="5", info="Enter the output voltage for the converter.",
                            minimum=0.0, maximum=1000.0, step=0.1, label="Output Voltage (V)", show_label=True)
            rload = gr.Number(value=5, placeholder="5", info="Enter the load resistance value for the converter.",
                            minimum=0.0, maximum=100000.0, step=5, label="Output Resistance (Ohm)", show_label=True)
            fsw = gr.Number(value=10000, placeholder="10000", info="Enter the operating frequency for the converter. Enter only numeric value. E.g: 10000 for 10k", label="Operating Frequency (Hz)", minimum=0.0, maximum=500000.0, step=10, show_label=True)
            ripl_crnt = gr.Number(value=10, placeholder="10", info="Enter the acceptable percentage ripple current for the converter. Enter only numeric value. E.g: 10 for 10%", label="Percentage Ripple Curent (%)", minimum=0.0, maximum=100.0, step=0.1, show_label=True)
            ripl_vout = gr.Number(value=0.5, placeholder="0.5", info="Enter the acceptable percentage ripple output voltage for the converter. Enter only numeric value. E.g: 10 for 10%", label="Percentage Ripple Output Voltage (%)", minimum=0.0, maximum=100.0, step=0.1, show_label=True)
        
        with gr.Row(variant="compact"):
            calc_button = gr.Button("Calculate Specs", variant="primary")
            hist_button = gr.Button("View History", variant="secondary")
            download_pdf_button = gr.Button("📥 Download History PDF", variant="secondary")
            
            # Results section
            with gr.Column():
                gr.Markdown("### DC-DC Coverter Specs & Transfer Functions")
                op_label = gr.Markdown(value=IDEL_STRING, show_label=False, line_breaks=True, header_links=True, visible=True, elem_id="op-string")

                with gr.Column(variant="panel", visible=False, elem_id="tf-resp-col") as results_col:
                    dcm_opt_ind_fig = gr.Plot(label="Inductor Value Optimization for DCM operation", show_label=False, visible=True, elem_id="dcm-opt-ind")
                    tf_resp_ttl = gr.Markdown("### Transfer Function Responses", visible=False, elem_id="tf-resp-ttl")
                    ltop_resp_ttl = gr.Markdown("$$  Line\ to\ Output\ Transient\ Response\ (G_{vg})  $$", visible=False, elem_id="ltop-tf-resp-ttl")
                    ltop_resp_plt = gr.Plot(label="Line-to-Output Response (G_{vg})", show_label=False, visible=True, elem_id="ltop-tf-resp")
                    ctop_resp_ttl = gr.Markdown("$$  Control\ to\ Output\ Transient\ Response\ (G_{vd})  $$", visible=False, elem_id="ctop-tf-resp-ttl")
                    ctop_resp_plt = gr.Plot(label="Control-to-Output Response (G_{vd})", show_label=False, visible=True, elem_id="ctop-tf-resp")

    # History section (shown when viewing history)
    with gr.Column(visible=False, elem_id="history-col") as history_col:
        gr.Markdown("### 📚 Calculation History")
        
        @gr.render(inputs=[history_state])
        def render_history(history_list):
            if not history_list:
                gr.Markdown("No calculation history available.")
                return
            
            for idx, row in enumerate(reversed(history_list)):
                original_idx = len(history_list) - 1 - idx
                with gr.Accordion(label=f"📊 Calculation #{original_idx + 1}", open=False):
                    gr.Markdown("## Specifications")
                    gr.Markdown(row['Specs'])
                    
                    # gr.Markdown("### Transfer Functions")
                    # gr.Markdown(f"$$ {row['line_to_op_tf']} $$")
                    # gr.Markdown(f"$$ {row['ctrl_to_op_tf']} $$")
                    
                    gr.Markdown("#### Response Plots")
                    with gr.Row():
                        gr.Plot(value=row['dcm_opt_ind_fig'], label="DCM Inductor Value Optimization")
                        gr.Plot(value=row['line_to_op_resp'], label="Line-to-Output Response")
                        gr.Plot(value=row['ctrl_to_op_resp'], label="Control-to-Output Response")
                        
    # PDF Download Section
    with gr.Column(visible=False, elem_id="pdf-col") as pdf_col:
        gr.Markdown("### 📄 Calculation History Report")
        pdf_file = gr.File(label="Download PDF", interactive=False)
    
    calc_button.click(
        dc_dc_converter,
        inputs=[ctype, vin, vo, rload, fsw, ripl_crnt, ripl_vout, history_state],
        outputs=[op_label, dcm_opt_ind_fig, tf_resp_ttl, ltop_resp_plt, ctop_resp_plt, ltop_resp_ttl, ctop_resp_ttl, history_state]
    ).success(
        lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
        inputs=[],
        outputs=[results_col, history_col, pdf_col]
    )

    hist_button.click(
        lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)),
        inputs=[],
        outputs=[results_col, history_col, pdf_col]
    )
    
    # Download PDF button
    download_pdf_button.click(
        lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[results_col, history_col, pdf_col]
    ).then(
        generate_history_pdf,
        inputs=[history_state],
        outputs=[pdf_file]
    )

    # Footer
    gr.Markdown("---")

    try:
        img_base64 = get_image_base64("assets/pranayj_sqlook.png")
        
        html_code = f"""
        <style>
            a.author {{
                text-decoration: none;
                color: #EA580C;
            }}
            a.author:hover {{
                text-decoration: none;
                color: #F14848;
            }}
            .circular-image {{
                width: 125px;
                height: 125px;
                border-radius: 55%;
                overflow: hidden;
                display: inline-block;
            }}
            .circular-image img {{
                width: 100%;
                height: 100%;
                object-fit: cover;
            }}
            .author-headline {{
                color: #14a3ee
            }}
        </style>
        <p><em>Created with</em> ❤️‍🔥 <em>by <a class='author' href='https://pranayjagtap.netlify.app' rel='noopener noreferrer' target='_blank'><b>Pranay Jagtap</b></a></em></p>
        <div class="circular-image">
            <a href="https://pranayjagtap.netlify.app" target="_blank" rel="noopener noreferrer">
                <img src="data:image/jpeg;base64,{img_base64}" alt="Pranay Jagtap">
            </a>
        </div>
        <p class="author-headline"><b>Machine Learning Enthusiast | Electrical Engineer<br>📍Nagpur, Maharashtra, India</b></p>
        """
        gr.HTML(html_code, apply_default_css=False)
    except Exception as e:
        logger.warning(f"Could not load author image: {e}")

demo.launch(server_name="0.0.0.0", server_port=7860)
