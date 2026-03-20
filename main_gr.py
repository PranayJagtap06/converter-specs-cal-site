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
import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
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


global specs_hist
specs_hist = pd.DataFrame(["Specs", "line_to_op_tf", "line_to_op_resp", "ctrl_to_op_tf", "ctrl_to_op_resp",
    "sys_g_obj", "sys_d_obj"])

IDEL_STRING = """
#### Input Parameters
\tMode = converter_type Converter
\tVin = input_voltage V
\tVo = output_voltage V
\tR = output_resistance Ohm
\tfsw = frequency Hz
\tIrp = ripl_current %
\tVrp = ripl_voltage %

#### Calculated Parameters
\tDuty Cycle = duty_cycle
\tPower Input = ip_power W
\tPower output = op_power W
\tOutput Current = op_crnt A
\tInductor Current = ind_crnt A
\tInput Current = ip_crnt A
\tCritical Inductance Value (Lcr) = crt_ind H
\tRipple Current due to Lcr = crt_ind_rpl_crnt A
\tContinuous Conduction Inductor Value (L) = ind H
\tRipple Current due to L = ind_ripl_crnt A
\tMaximum inductor ripple current = maxind_crnt A
\tMinimum inductor ripple current = minind_crnt A
\tOutput Capacitor = cap F
\tCapacitor ESR = esr Ohm

#### Transfer Functions
$$ line-to-output\ transfer\ function $$

$$ control-to-output\ transfer\ function $$
"""

def save_to_history(specs: str, gvg_latex: str, gvd_latex: str, line_plot, ctrl_plot, sys_g, sys_d) -> pd.DataFrame:
    """Save calculation specs and plots to history DataFrame"""
    global specs_hist
    specs_hist = pd.concat([
        specs_hist, 
        pd.DataFrame({
            "Specs": [specs],
            "line_to_op_tf": [gvg_latex],
            "line_to_op_resp": [line_plot],
            "ctrl_to_op_tf": [gvd_latex],
            "ctrl_to_op_resp": [ctrl_plot],
            "sys_g_obj": [sys_g],
            "sys_d_obj": [sys_d]
        })
    ], ignore_index=True)

    return specs_hist


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
            coef_str = f"{abs(coef):.4g}"

            # Format power
            if power == 0:
                term = coef_str
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


# def update_convt_specs(*args):
#     # pass
#     results = dc_dc_converter(*args)
#     return gr.update(elem_id="op-string", visible=True), gr.update(elem_id="tf-resp-ttl", visible=True), gr.update(elem_id="ltop-tf-resp-ttl", visible=True), gr.update(elem_id="ltop-tf-resp", visible=True), gr.update(elem_id="ctop-tf-resp-ttl", visible=True), gr.update(elem_id="ctop-tf-resp", visible=True)


def display_history():
    # gr.Markdown(value="### Calculation History")
    # if specs_hist.empty or len(specs_hist) == 0:
    #     gr.Warning(message="No calculation history available.")
    # else:
    #     try:
    #         for idx, row in specs_hist.iterrows():
    #             with gr.Accordion(label=f"📊 Calculation #{idx + 1}", open=False):
    #                 gr.Markdown("### Specifications")
    #                 gr.Markdown(value=row["Specs"], show_label=False, line_breaks=True, header_links=True)

    #                 gr.Markdown("### Transfer Functions")
    #                 with gr.Row():
    #                     with gr.Column():
    #                         gr.Markdown("#### Line-to-Output Transfer Function")
    #                         try:
    #                             gvg_img = latex_to_image(row["line_to_op_tf"], fontsize=11)
    #                             gr.Image(value=gvg_img, format="png", label="Gvg(s) Transfer Function", show_label=True)
    #                         except Exception as e:
    #                             logger.error(f"Error rendering Gvg LaTeX: {e}")
    #                             gr.Markdown(value=f"Gvg(s): {str(row['sys_g_obj'])}", show_label=False, line_breaks=True, header_links=True)

    #                     with gr.Column():
    #                         gr.Markdown("#### Control-to-Output Transfer Function")
    #                         try:
    #                             gvd_img = latex_to_image(row["ctrl_to_op_tf"], fontsize=11)
    #                             gr.Image(value=gvd_img, format="png", label="Gvd(s) Transfer Function", show_label=True)
    #                         except Exception as e:
    #                             logger.error(f"Error rendering Gvd LaTeX: {e}")
    #                             gr.Markdown(value=f"Gvd(s): {str(row['sys_d_obj'])}", show_label=False, line_breaks=True, header_links=True)

    #                 gr.Markdown("### Transient Response Plots")
    #                 with gr.Row():
    #                     with gr.Column():
    #                         gr.Markdown("#### Line-to-Output Response")
    #                         gr.Plot(value=row["line_to_op_resp"], label="Line-to-Output Response (Gvg)", show_label=True)

    #                     with gr.Column():
    #                         gr.Markdown("#### Control-to-Output Response")
    #                         gr.Plot(value=row["ctrl_to_op_resp"], label="Control-to-Output Response (Gvd)", show_label=True)
    #     except Exception as e:
    #         gr.Error(message="An error occurred while displaying the history.\n{e}")
    #         logger.error(f"Error displaying history: {e}")

    # if specs_hist.empty or len(specs_hist) == 0:
    #     return "<p>No calculation history available.</p>"
    
    # html_output = "<h3>📊 Calculation History</h3>"
    
    # for idx, row in specs_hist.iterrows():
    #     if idx == 0:  # Skip header row
    #         continue
            
    #     html_output += f"""
    #     <details>
    #         <summary><strong>Calculation #{idx}</strong></summary>
    #         <div style="padding: 10px;">
    #             <h4>Specifications</h4>
    #             <pre>{row['Specs']}</pre>
                
    #             <h4>Transfer Functions</h4>
    #             <p><strong>Line-to-Output:</strong></p>
    #             <pre>{latex_to_image(row['line_to_op_tf'])}</pre>
                
    #             <p><strong>Control-to-Output:</strong></p>
    #             <pre>{latex_to_image(row['ctrl_to_op_tf'])}</pre>

    #             <h4>Transient Response Plots</h4>
    #             <p><strong>Line-to-Output Response:</strong></p>
    #             <div>{row['line_to_op_resp']}</div>

    #             <p><strong>Control-to-Output Response:</strong></p>
    #             <div>{row['ctrl_to_op_resp']}</div>
    #         </div>
    #     </details>
    #     <hr>
    #     """
    
    # return html_output

    """Return list of components to display history with plots and equations"""
    global specs_hist
    
    if specs_hist.empty or len(specs_hist) <= 1:  # Check if only header row exists
        return [], []
    
    history_items = []
    
    for idx, row in specs_hist.iterrows():
        if idx == 0:  # Skip header row
            continue
        
        # Create accordion content for each calculation
        accordion_content = f"""
### 📊 Calculation #{idx}

#### Specifications
```
{row['Specs']}
```

#### Line-to-Output Transfer Function
$$ {row['line_to_op_tf']} $$

#### Control-to-Output Transfer Function
$$ {row['ctrl_to_op_tf']} $$
        """
        
        history_items.append({
            'markdown': accordion_content,
            'line_plot': row['line_to_op_resp'],
            'ctrl_plot': row['ctrl_to_op_resp']
        })
    
    return history_items


def plotly_to_image_bytes(fig, width=800, height=600):
    """Convert Plotly figure to PNG bytes using kaleido"""
    try:
        img_bytes = pio.to_image(fig, format="png", width=width, height=height, engine="kaleido")
        return img_bytes
    except Exception as e:
        logger.error(f"Error converting plotly to image: {e}")
        return None


def generate_history_pdf():
    """Generate PDF from calculation history with all equations and plots"""
    global specs_hist
    
    if specs_hist.empty or len(specs_hist) <= 1:
        return None
    
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Title page
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'DC-DC Converter Calculation History', 0, 1, 'C')
        pdf.ln(10)
        
        for idx, row in specs_hist.iterrows():
            if idx == 0:  # Skip header row
                continue
            
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, f'Calculation #{idx}', 0, 1, 'L')
            pdf.ln(5)
            
            # Specifications
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Specifications:', 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            
            # Parse specs and add to PDF
            specs_lines = row['Specs'].split('\n')
            for line in specs_lines:
                if line.strip():
                    pdf.multi_cell(0, 5, line.strip())
            
            pdf.ln(5)
            
            # Transfer Functions
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Transfer Functions:', 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            
            # Line-to-Output TF
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 8, 'Line-to-Output Transfer Function:', 0, 1, 'L')
            pdf.set_font('Courier', '', 9)
            
            # Convert LaTeX to image
            latex_img = latex_to_image(row['line_to_op_tf'], fontsize=10, dpi=150)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(latex_img.read())
                tmp_path = tmp_file.name
            
            pdf.image(tmp_path, x=10, w=190)
            os.unlink(tmp_path)
            pdf.ln(5)
            
            # Control-to-Output TF
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 8, 'Control-to-Output Transfer Function:', 0, 1, 'L')
            pdf.set_font('Courier', '', 9)
            
            latex_img = latex_to_image(row['ctrl_to_op_tf'], fontsize=10, dpi=150)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(latex_img.read())
                tmp_path = tmp_file.name
            
            pdf.image(tmp_path, x=10, w=190)
            os.unlink(tmp_path)
            pdf.ln(10)
            
            # Line-to-Output Plot
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Line-to-Output Response:', 0, 1, 'L')
            
            if row['line_to_op_resp'] is not None:
                img_bytes = plotly_to_image_bytes(row['line_to_op_resp'], width=1000, height=600)
                if img_bytes:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        tmp_file.write(img_bytes)
                        tmp_path = tmp_file.name
                    
                    pdf.image(tmp_path, x=10, w=190)
                    os.unlink(tmp_path)
            
            pdf.ln(10)
            
            # Control-to-Output Plot
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Control-to-Output Response:', 0, 1, 'L')
            
            if row['ctrl_to_op_resp'] is not None:
                img_bytes = plotly_to_image_bytes(row['ctrl_to_op_resp'], width=1000, height=600)
                if img_bytes:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        tmp_file.write(img_bytes)
                        tmp_path = tmp_file.name
                    
                    pdf.image(tmp_path, x=10, w=190)
                    os.unlink(tmp_path)
        
        # Save PDF
        output_dir = "gradio_logs"
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, "converter_history.pdf")
        pdf.output(pdf_path)
        
        return pdf_path
    
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        gr.Error(f"Failed to generate PDF: {e}")
        return None


def download_hist_as_pdf(specs_hist: pd.DataFrame):
    if specs_hist.empty:
        gr.Warning(message="No calculation history to download.")
        return
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for idx, row in specs_hist.iterrows():
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
                pdf.multi_cell(0, 6.5, line.strip())
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
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_gvg:
                tmp_gvg.write(gvg_img.getvalue())
                tmp_gvg.flush()
                pdf.image(tmp_gvg.name, x=11, w=110)
                os.unlink(tmp_gvg.name)
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
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_gvd:
                tmp_gvd.write(gvd_img.getvalue())
                tmp_gvd.flush()
                pdf.image(tmp_gvd.name, x=11, w=110)
                os.unlink(tmp_gvd.name)
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
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as imgfile:
            row["line_to_op_resp"].write_image(imgfile.name, format="png", width=800, height=500, scale=2)
            pdf.image(imgfile.name, x=10, w=190)
            os.unlink(imgfile.name)
        pdf.ln(3)
        
        # Control-to-Output Response
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 6, "Control-to-Output Response:", ln=True)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as imgfile:
            row["ctrl_to_op_resp"].write_image(imgfile.name, format="png", width=800, height=500, scale=2)
            pdf.image(imgfile.name, x=10, w=190)
            os.unlink(imgfile.name)
        
        # Add page break between calculations (except last one)
        # if idx < len(st.session_state["history_df"]) - 1:
        #     pdf.add_page()

    # Save PDF to a temporary file and offer download
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        
        # with open(tmpfile.name, "rb") as f:
        #     pdf_data = f.read()
        
        # st.download_button(
        #     label="📥 Download Calculation History as PDF",
        #     data=pdf_data,
        #     file_name="dc_dc_converter_analysis_history.pdf",
        #     mime="application/pdf",
        #     type="primary"
        # )
        gr.DownloadButton(
            label="📥 Download Calculation History as PDF",
            value=tmpfile.name,
        )
        
        os.unlink(tmpfile.name)
        
    gr.Success("✅ PDF generated successfully!")


def dc_dc_converter(converter_type: str, input_voltage: float, output_voltage: float, output_resistance: float, frequency: float, ripl_current: float, ripl_voltage: float) -> Optional[tuple]:
    global specs_hist
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
                duty_cycle: float = bucklc.bck_duty_cycle(output_voltage, input_voltage)
                op_crnt: float = np.round((output_voltage / output_resistance), decimals=3)
                ip_crnt: float = bucklc.bck_ind_current(duty_cycle, op_crnt)
                ind_crnt: float = op_crnt
                ip_power: float = np.round((input_voltage * ip_crnt), decimals=3)
                op_power: float = np.round((output_voltage * op_crnt), decimals=3)
                crt_ind: str = bucklc.bck_cr_ind(duty_cycle, output_resistance, frequency)
                crt_ind_rpl_crnt: float = bucklc.bck_ind_ripl_(
                    output_voltage, duty_cycle, frequency, crt_ind)
                ind_ripl_crnt: str = bucklc.bck_ripl_current(
                    ind_crnt, ripl_current)
                ind: str = bucklc.bck_cont_ind(
                    output_voltage, duty_cycle, frequency, ind_ripl_crnt)
                maxind_crnt: float = np.round(
                    (ind_crnt + (float(ind_ripl_crnt) / 2)), decimals=3)
                minind_crnt: float = np.round(
                    (ind_crnt - (float(ind_ripl_crnt) / 2)), decimals=3)
                cap: str = bucklc.bck_cap_val(duty_cycle, ind, ripl_voltage, frequency)
                esr: float = bucklc.Esr(ripl_voltage, output_voltage, maxind_crnt)
            case "Boost":
                duty_cycle: float = boostlc.bst_duty_cycle(output_voltage, input_voltage)
                op_crnt: float = np.round((output_voltage / output_resistance), decimals=3)
                ind_crnt: float = boostlc.bst_ind_current(duty_cycle, op_crnt)
                ip_crnt: float = ind_crnt
                ip_power: float = np.round((input_voltage * ip_crnt), decimals=3)
                op_power: float = np.round((output_voltage * op_crnt), decimals=3)
                crt_ind: str = boostlc.bst_cr_ind(duty_cycle, output_resistance, frequency)
                crt_ind_rpl_crnt: float = boostlc.bst_ind_ripl_(
                    input_voltage, duty_cycle, frequency, crt_ind)
                ind_ripl_crnt: str = boostlc.bst_ripl_current(
                    ind_crnt, ripl_current)
                ind: str = boostlc.bst_cont_ind(
                    input_voltage, duty_cycle, frequency, ind_ripl_crnt)
                maxind_crnt: float = np.round(
                    (ind_crnt + (float(ind_ripl_crnt) / 2)), decimals=3)
                minind_crnt: float = np.round(
                    (ind_crnt - (float(ind_ripl_crnt) / 2)), decimals=3)
                cap: str = boostlc.bst_cap_val(
                    duty_cycle, output_resistance, ripl_voltage, frequency)
                esr: float = boostlc.Esr(ripl_voltage, output_voltage, maxind_crnt)
            case "BuckBoost":
                duty_cycle: float = bckbstlc.bckbst_duty_cycle(output_voltage, input_voltage)
                op_crnt: float = np.round((output_voltage / output_resistance), decimals=3)
                ind_crnt: float = bckbstlc.bckbst_ind_current(
                    duty_cycle, op_crnt)
                ip_crnt: float = np.round((ind_crnt * duty_cycle), decimals=3)
                ip_power: float = np.round((input_voltage * ip_crnt), decimals=3)
                op_power: float = np.round((output_voltage * op_crnt), decimals=3)
                crt_ind: str = bckbstlc.bckbst_cr_ind(duty_cycle, output_resistance, frequency)
                crt_ind_rpl_crnt: float = bckbstlc.bckbst_ind_ripl_(
                    input_voltage, duty_cycle, frequency, crt_ind)
                ind_ripl_crnt: str = bckbstlc.bckbst_ripl_current(
                    ip_crnt, ripl_current)
                ind: str = bckbstlc.bckbst_cont_ind(
                    input_voltage, duty_cycle, frequency, ind_ripl_crnt)
                maxind_crnt: float = np.round(
                    (ind_crnt + (float(ind_ripl_crnt) / 2)), decimals=3)
                minind_crnt: float = np.round(
                    (ind_crnt - (float(ind_ripl_crnt) / 2)), decimals=3)
                cap: str = bckbstlc.bckbst_cap_val(
                    duty_cycle, output_resistance, ripl_voltage, frequency)
                esr: float = bckbstlc.Esr(ripl_voltage, output_voltage, maxind_crnt)

        # Plot response
        line_to_op_resp, ctrl_to_op_resp, sys_g, sys_d = tfr.plot_response(
            duty_cycle, input_voltage, float(ind), float(cap), output_resistance, converter_type
        )

        # Generate LaTeX equations
        gvg_latex = tf_to_latex(sys_g, name='Line-to-Output\ Transfer\ Function\ G_{vg}')
        gvd_latex = tf_to_latex(sys_d, name='Contol-to-Output\ Transfer\ Function\ G_{vd}')

        # Display results
        op_string = f"""
        #### Input Parameters
        \tMode = {converter_type} Converter
        \tVin = {input_voltage}V
        \tVo = {output_voltage}V
        \tR = {output_resistance}Ohm
        \tfsw = {frequency}Hz
        \tIrp = {ripl_current}%
        \tVrp = {ripl_voltage}%

        #### Calculated Parameters
        \tDuty Cycle = {duty_cycle}
        \tPower Input = {ip_power}W
        \tPower output = {op_power}W
        \tOutput Current = {op_crnt}A
        \tInductor Current = {ind_crnt}A
        \tInput Current = {ip_crnt}A
        \tCritical Inductance Value (Lcr) = {crt_ind}H
        \tRipple Current due to Lcr = {crt_ind_rpl_crnt}A
        \tContinuous Conduction Inductor Value (L) = {ind}H
        \tRipple Current due to L = {ind_ripl_crnt}A
        \tMaximum inductor ripple current = {maxind_crnt}A
        \tMinimum inductor ripple current = {minind_crnt}A
        \tOutput Capacitor = {cap}F
        \tCapacitor ESR = {esr}Ohm

        #### Transfer Functions
        $$ {gvg_latex} $$

        $$ {gvd_latex} $$
        """

        save_to_history(op_string, gvg_latex, gvd_latex, line_to_op_resp, ctrl_to_op_resp, sys_g, sys_d)
        print("Specs-Hist: \n", specs_hist)
        # update_convt_specs(op_string, line_to_op_resp, ctrl_to_op_resp, gvg_latex, gvd_latex)
        # return op_string, line_to_op_resp, ctrl_to_op_resp, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        # return op_string, gr.update(visible=True), line_to_op_resp, ctrl_to_op_resp, gr.update(visible=True), gr.update(visible=True)

        print("NaN count per column:")
        print(specs_hist.isna().sum())
        print("\nData types:")
        print(specs_hist.dtypes)
        print("\nFirst few rows:")
        print(specs_hist.head())
        print("\nRows with any NaN:")
        print(specs_hist[specs_hist.isna().any(axis=1)])

        return (
            op_string, 
            gr.update(visible=True), 
            line_to_op_resp, 
            ctrl_to_op_resp, 
            gr.update(visible=True), 
            gr.update(visible=True),
            gr.update(visible=False)  # Hide history column
        )
    except AssertionError as ae:
        gr.Error(title="Input Error: ", message=f"{ae}")
        logger.error(f"Input Error: {ae}")
        return None
    except Exception as e:
        gr.Error(title="Calculation Error: ", message=f"{e}")
        logger.error(f"Calculation Error: {e}")
        return None


description = "## Get Your DC-DC Converter Specs Calculated! 🚀\n This application helps you calculate specifications and visualize the steady state response of *Buck*, *Boost*, and *Buck-Boost* converters based on your input parameters."

# demo = gr.Interface(
#     fn=dc_dc_converter,
#     inputs=["text", "number", "number", "number", "number", "number", "number"],
#     outputs=[gr.TextArea(value="No calculation performed yet.", label="DC-DC Converter Specs & Plots", scale=1, show_label=True)],
#     api_name="dc_dc_converter_specs",
#     title="🦄 DC-DC Converter Specs Calculator",
#     description=description
# )

with gr.Blocks(delete_cache=(86400, 86400)) as demo:        
    gr.Markdown("# 🦄 DC-DC Converter Specs Calculator")
    gr.Markdown(description)
    with gr.Row(equal_height=True, variant="compact"):
        with gr.Column(variant="compact"):
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
        
        with gr.Row():
            calc_button = gr.Button("Calculate Specs", variant="primary")
            hist_button = gr.Button("View History", variant="secondary")
            download_pdf_button = gr.Button("📥 Download History PDF", variant="secondary")
            with gr.Column():
                gr.Markdown("### DC-DC Coverter Specs & Transfer Functions")
                op_label = gr.Markdown(value=IDEL_STRING, show_label=False, line_breaks=True, header_links=True, visible=True, elem_id="op-string")

    # Results section
    with gr.Column(variant="panel", visible=True, elem_id="tf-resp-col") as results_col:
        tf_resp_ttl = gr.Markdown("### Transfer Function Responses", visible=False, elem_id="tf-resp-ttl")
        ltop_resp_ttl = gr.Markdown("$$  Line\ to\ Output\ Transient\ Response\ (G_{vg})  $$", visible=False, elem_id="ltop-tf-resp-ttl")
        ltop_resp_plt = gr.Plot(label="Line-to-Output Response (G_{vg})", show_label=False, visible=True, elem_id="ltop-tf-resp")
        ctop_resp_ttl = gr.Markdown("$$  Control\ to\ Output\ Transient\ Response\ (G_{vd})  $$", visible=False, elem_id="ctop-tf-resp-ttl")
        ctop_resp_plt = gr.Plot(label="Control-to-Output Response (G_{vd})", show_label=False, visible=True, elem_id="ctop-tf-resp")

    # History section (shown when viewing history)
    with gr.Column(visible=False, elem_id="history-col") as history_col:
        gr.Markdown("### 📚 Calculation History")
        history_accordion = gr.Accordion(label="History Items", open=True)
        with history_accordion:
            # Dynamic history content will be rendered here
            history_container = gr.Column()
    
    calc_button.click(
        dc_dc_converter,
        inputs=[ctype, vin, vo, rload, fsw, ripl_crnt, ripl_vout],
        outputs=[op_label, tf_resp_ttl, ltop_resp_plt, ctop_resp_plt, ltop_resp_ttl, ctop_resp_ttl, history_col]
    ).then(
        lambda: gr.update(visible=True),
        inputs=[],
        outputs=[results_col]
    )

    # with gr.Column(visible=False, elem_id="history-col") as history_col:
    #     history_output = gr.HTML(label="History", elem_id="history-output")

    # View history button click
    def show_history():
        history_items = display_history()
        
        if not history_items:
            return gr.update(visible=False), gr.update(visible=True), "No history available"
        
        # Build history display
        history_html = ""
        plots_line = []
        plots_ctrl = []
        
        for item in history_items:
            history_html += item['markdown']
            history_html += "\n\n---\n\n"
            plots_line.append(item['line_plot'])
            plots_ctrl.append(item['ctrl_plot'])
        
        return (
            gr.update(visible=False),  # Hide results
            gr.update(visible=True),   # Show history
            history_html,
            plots_line,
            plots_ctrl
        )
    
    # Rebuild history display dynamically
    def rebuild_history_display():
        history_items = display_history()
        
        if not history_items:
            return [gr.Markdown("No calculation history available.")]
        
        components = []
        for idx, item in enumerate(history_items):
            with gr.Accordion(label=f"📊 Calculation #{idx + 1}", open=False):
                gr.Markdown(item['markdown'])
                gr.Markdown("#### Response Plots")
                gr.Plot(value=item['line_plot'], label="Line-to-Output Response")
                gr.Plot(value=item['ctrl_plot'], label="Control-to-Output Response")
        
        return components
    
    hist_button.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[results_col, history_col]
    ).then(
        rebuild_history_display,
        inputs=[],
        outputs=[history_container]
    )
    
    # Download PDF button
    download_pdf_button.click(
        generate_history_pdf,
        inputs=[],
        outputs=[gr.File(label="Download PDF")]
    )
    
    # hist_button.click(
    #     display_history,
    #     inputs=[],
    #     outputs=[history_output]
    # ).then(
    #     lambda: gr.update(visible=True),
    #     inputs=[],
    #     outputs=[history_col]
    # )

    # demo.load(fn=calc_button.click(),
    #           inputs=[ctype, vin, vo, rload, fsw, ripl_crnt, ripl_vout],
    #           outputs=op_label
    # )

demo.launch(server_name="0.0.0.0", server_port=7860)