import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import io
import base64

# ----------------------
# Page Setup
# ----------------------
st.set_page_config(page_title="RTD Analyzer", layout="wide")
st.title("🧪 Packed Bed Reactor RTD Analyzer")

# ----------------------
# Instructions Section
# ----------------------
with st.expander("📘 Click here for Instructions"):
    st.markdown("""
### 🔍 Objective
This tool helps you analyze residence time distribution (RTD) data for packed bed reactors.

You’ll upload your experimental data and receive:
- **Mean residence time**
- **Variance**
- **Axial dispersion coefficient**
- **Peclet number (Pe)**
- **Space time**

---

### 🧾 File Format
Your CSV must include **4 columns**:
1. `Time (s)` — Time in seconds  
2. `Concentration (C)` — Measured concentration  
3. `Flow Rate (mL/min)` — Constant flow rate for that run  
4. `Run` — Run number (e.g., 1, 2...)

Example:
```
Time (s),Concentration (C),Flow Rate (mL/min),Run
0,0.000,30,1
1,0.012,30,1
```

---

### 🛠 Instructions
1. **Enter reactor parameters** (diameter, length, porosity) in the sidebar.
2. **Upload your CSV file** using the uploader below.
3. **View results**: A summary table and plots for each run will appear.
4. **Download results and plots** using the provided buttons.
""")

# ----------------------
# Sidebar Inputs
# ----------------------
st.sidebar.header("🔧 Reactor Parameters")
D = st.sidebar.number_input("Diameter (m)", value=0.01680, format="%.5f")
L = st.sidebar.number_input("Length (m)", value=0.1000, format="%.4f")
epsilon = st.sidebar.number_input("Porosity (ε)", value=0.33, min_value=0.0, max_value=1.0, step=0.01)
A = np.pi * (D / 2) ** 2  # Cross-sectional area (m^2)

# ----------------------
# File Upload
# ----------------------
uploaded_file = st.file_uploader("📤 Upload your RTD data file (CSV format)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if not all(col in df.columns for col in ['Time (s)', 'Concentration (C)', 'Flow Rate (mL/min)', 'Run']):
        st.error("❌ CSV file must contain the following columns: Time (s), Concentration (C), Flow Rate (mL/min), Run")
    else:
        results = []
        plots = {}

        for run_id in df['Run'].unique():
            run_data = df[df['Run'] == run_id].copy()
            t = run_data['Time (s)'].values
            C = run_data['Concentration (C)'].values
            Q_mL_min = run_data['Flow Rate (mL/min)'].values[0]
            Q = Q_mL_min / (1000 * 1000 * 60)  # Convert to m^3/s

            E = C / trapezoid(C, t)  # Normalize to E(t)
            tau = trapezoid(t * E, t)
            sigma_squared = trapezoid((t - tau) ** 2 * E, t)
            D_ax = (sigma_squared * (Q / A) ** 3) / 2
            Pe = (L * (Q / A)) / D_ax
            tau_0 = (epsilon * A * L) / Q

            results.append({
                "Run": run_id,
                "Flow Rate (mL/min)": Q_mL_min,
                "Mean Residence Time (s)": round(tau, 2),
                "Variance (s²)": round(sigma_squared, 2),
                "Axial Dispersion Coefficient (m²/s)": f"{D_ax:.2e}",
                "Peclet Number": f"{Pe:.2f}",
                "Space Time τ₀ (s)": round(tau_0, 2)
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