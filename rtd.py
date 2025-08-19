import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import io

# ----------------------
# Page Setup
# ----------------------
st.set_page_config(page_title="RTD Analyzer", layout="wide")
st.title("🧪 Packed Bed Reactor RTD Analyzer")

# ----------------------
# Instructions Section
# ----------------------
with st.expander("📘 Click here for Instructions"):
    st.markdown(
        """
        ### 🔍 Objective
        This tool analyzes residence time distribution (RTD) data for packed bed reactors.

        You’ll get:
        - **Mean residence time**
        - **Variance**
        - **Axial dispersion coefficient**
        - **Peclet number (Pe)**
        - **Space time**
        - **Reynolds number**

        ---

        ### 🧾 File Format
        CSV must include these 4 columns:
        1. `Time (s)`
        2. `Concentration (C)`
        3. `Flow Rate (mL/min)`
        4. `Run`

        Example (CSV):
            Time (s),Concentration (C),Flow Rate (mL/min),Run  
            0,0.000,30,1  
            1,0.012,30,1  

        ---

        ### 🛠 Instructions
        1. **Enter reactor and fluid parameters** in the sidebar.
        2. **Upload your CSV file**.
        3. **View and download results**.
        """
    )

# ----------------------
# Sidebar Inputs
# ----------------------
st.sidebar.header("🔧 Reactor and Fluid Parameters")
D = st.sidebar.number_input("Reactor Diameter (m)", value=0.01680, format="%.5f")
L = st.sidebar.number_input("Reactor Length (m)", value=0.1000, format="%.4f")
epsilon = st.sidebar.number_input("Porosity (ε)", value=0.33, min_value=0.0, max_value=1.0, step=0.01)
Dp = st.sidebar.number_input("Particle Diameter (m)", value=300e-6, format="%.1e")
mu = st.sidebar.number_input("Viscosity (Pa·s)", value=0.0010, format="%.4f")
rho = st.sidebar.number_input("Density (kg/m³)", value=1000.0, format="%.1f")
A = np.pi * (D / 2) ** 2  # Cross-sectional area (m²)

# ----------------------
# File Upload
# ----------------------
uploaded_file = st.file_uploader("📤 Upload your RTD data file (CSV format)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]  # Strip whitespace

    required_cols = ['Time (s)', 'Concentration (C)', 'Flow Rate (mL/min)', 'Run']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ CSV file must contain the following columns: {', '.join(required_cols)}")
    else:
        results = []
        plots = {}

        for run_id in df['Run'].unique():
            run_data = df[df['Run'] == run_id].copy()
            t = run_data['Time (s)'].values
            C = np.clip(run_data['Concentration (C)'].values, 0, None)  # Clip negatives
            Q_mL_min = run_data['Flow Rate (mL/min)'].values[0]
            Q = Q_mL_min / (1e6 * 60)  # Convert to m³/s
            v = Q / A  # Superficial velocity (m/s)

            E = C / trapezoid(C, t) if trapezoid(C, t) != 0 else np.zeros_like(C)
            tau = trapezoid(t * E, t)
            sigma_squared = trapezoid((t - tau) ** 2 * E, t)
            D_ax = (sigma_squared * v ** 3) / 2 if v != 0 else np.nan
            Pe = (L * v) / D_ax if D_ax != 0 else np.nan
            tau_0 = (epsilon * A * L) / Q if Q != 0 else np.nan
            Re = (Dp * v * rho) / ((1 - epsilon) * mu) if mu != 0 else np.nan

            results.append({
                "Run": run_id,
                "Flow Rate (mL/min)": Q_mL_min,
                "Mean Residence Time (s)": round(tau, 2),
                "Variance (s²)": round(sigma_squared, 2),
                "Axial Dispersion Coefficient (m²/s)": f"{D_ax:.2e}",
                "Peclet Number": f"{Pe:.2f}",
                "Space Time τ₀ (s)": round(tau_0, 2),
                "Reynolds Number": f"{Re:.2f}"
            })

            # Plot E(t)
            fig, ax = plt.subplots()
            ax.plot(t, E, label=f"Run {run_id}")
            ax.axvline(tau, color='red', linestyle='--', label=f"τ = {tau:.2f} s")
            ax.set_title(f"RTD Curve for Run {run_id}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("E(t)")
            ax.legend()
            fig.tight_layout()
            plots[run_id] = fig

        # ----------------------
        # Display Results
        # ----------------------
        results_df = pd.DataFrame(results)
        st.subheader("📊 Summary Table")
        st.dataframe(results_df)

        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results as CSV", csv_data, "RTD_results.csv", "text/csv")

        # ----------------------
        # Show and Download Plots
        # ----------------------
        st.subheader("📈 RTD Plots")
        for run_id, fig in plots.items():
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label=f"⬇️ Download Plot for Run {run_id}",
                data=buf.getvalue(),
                file_name=f"RTD_Run_{run_id}.png",
                mime="image/png"
            )
