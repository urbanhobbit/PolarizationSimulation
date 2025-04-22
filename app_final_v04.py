# --- Imports ---
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from agents import Agents
from scenarios import SCENARIOS
from config import *
from polarisation import calculate_polarisation_metrics, calculate_social_choice_winners
from deliberation import deliberation_step_matched
import io
import inspect
import math
import time

# --- Simulation Function ---
@st.cache_data
def run_simulation(N_AGENTS, T, N_FRAMES, party_positions, delta_matrix, preference_update_mode="dynamic"):
    np.random.seed(42)
    N_PARTIES = len(party_positions)
    N_PROFILES = math.factorial(N_PARTIES)

    # Validate delta_matrix shape before proceeding
    if delta_matrix.shape != (N_PROFILES, N_PROFILES):
        raise ValueError(f"delta_matrix shape {delta_matrix.shape} does not match required shape ({N_PROFILES}, {N_PROFILES}) for {N_PARTIES} parties.")

    agents = Agents(N_AGENTS, OPINION_SPACE_SIZE)
    profiles = np.random.choice(N_PROFILES, N_AGENTS)  # Initialize profiles after delta_matrix is validated
    agents.pref_indices = profiles

    step_interval = max(1, T // N_FRAMES)
    record_iters = list(range(step_interval, T + 1, step_interval))

    positions_record = []
    polarisation_records = []
    voting_records = []
    social_choice_records = []

    for t in range(1, T + 1):
        if st.session_state.get('cancel_simulation', False):
            return None, None, None, None

        agents.positions, agents.pref_indices = deliberation_step_matched(
            agents.positions, agents.pref_indices, delta_matrix, t, party_positions,
            opinion_space_size=OPINION_SPACE_SIZE,
            mu_a=MU_ATTRACTION,
            mu_r=MU_REACTION,
            discount_coeff=DISCOUNT_COEFF,
            interaction_rate=1.0
        )

        if t in record_iters:
            preferences = agents.update_preferences(party_positions)
            first_choices = preferences[:, 0]
            pref_indices = agents.pref_indices

            frame = pd.DataFrame({
                "x": agents.positions[:, 0],
                "y": agents.positions[:, 1],
                "FirstChoice": first_choices,
                "PrefIndex": pref_indices,
                "Iteration": t
            })

            party_centers = []
            for i in range(N_PARTIES):
                supporters = agents.positions[first_choices == i]
                center = supporters.mean(axis=0) if len(supporters) > 0 else np.array([np.nan, np.nan])
                party_centers.append(center)

            society_center = agents.positions.mean(axis=0)

            frame_party_centers = pd.DataFrame(party_centers, columns=["x", "y"])
            frame_party_centers["Party"] = [str(i) for i in range(N_PARTIES)]
            frame_party_centers["Iteration"] = t

            frame_society_center = pd.DataFrame({
                "x": [society_center[0]],
                "y": [society_center[1]],
                "Iteration": [t]
            })

            sig = inspect.signature(calculate_polarisation_metrics)
            if len(sig.parameters) != 7:
                return None, None, None, None

            try:
                party_polar, pref_polar, binary_polar, kemeny_polar = calculate_polarisation_metrics(
                    agents.positions, preferences, first_choices, party_positions, N_AGENTS, OPINION_SPACE_SIZE, agents.pref_indices
                )
            except TypeError:
                return None, None, None, None

            polarisation_records.append({
                "Iteration": t,
                "PartyPolarisation": party_polar,
                "PrefPolarisation": pref_polar,
                "BinaryPolarisation": binary_polar,
                "KemenyPolarisation": kemeny_polar
            })

            voting_share = pd.Series(first_choices).value_counts(normalize=True).sort_index()
            voting_records.append({"Iteration": t, **voting_share.to_dict()})

            social_choice_results = calculate_social_choice_winners(preferences, N_AGENTS, N_PARTIES)
            social_choice_record = {
                "Iteration": t,
                "PluralityWinner": social_choice_results["plurality_winner"],
                "BordaWinner": social_choice_results["borda_winner"],
                "MajCompWinner": social_choice_results["maj_comp_winner"],
                "CopelandWinner": social_choice_results["copeland_winner"]
            }
            for i in range(N_PARTIES):
                social_choice_record[f"PluralityVotes{i}"] = social_choice_results["plurality_votes"][i]
                social_choice_record[f"BordaScores{i}"] = social_choice_results["borda_scores"][i]
                social_choice_record[f"MajCompScores{i}"] = social_choice_results["maj_comp_scores"][i]
                social_choice_record[f"CopelandScores{i}"] = social_choice_results["copeland_scores"][i]
            social_choice_records.append(social_choice_record)

            positions_record.append((frame, frame_party_centers, frame_society_center))

    polarisation_df = pd.DataFrame(polarisation_records)
    voting_df = pd.DataFrame(voting_records).fillna(0)
    social_choice_df = pd.DataFrame(social_choice_records)

    return positions_record, polarisation_df, voting_df, social_choice_df

# --- Streamlit Setup ---
st.set_page_config(page_title="Polarisation Simulation", layout="wide")
st.title("🥉 Agent-Based Polarisation Simulation")
st.subheader("POlarization viewed from a SOcial choice Perspective")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Settings")

# First, determine N_PARTIES and party_positions
party_input_mode = st.sidebar.radio(
    "How to define party positions?",
    options=["Preset Scenario", "Manual Entry"],
    index=0
)

if party_input_mode == "Preset Scenario":
    selected_scenario = st.sidebar.selectbox("Select Scenario", list(SCENARIOS.keys()))
    party_positions = np.array(SCENARIOS[selected_scenario]["party_positions"])
    N_PARTIES = len(party_positions)
else:
    N_PARTIES = st.sidebar.slider("Number of Parties", 2, 5, 3)
    manual_positions = []
    for i in range(N_PARTIES):
        x = st.sidebar.number_input(f"Party {i+1} - X Coordinate", value=float(i*2-2), step=0.1)
        y = st.sidebar.number_input(f"Party {i+1} - Y Coordinate", value=0.0, step=0.1)
        manual_positions.append([x, y])
    party_positions = np.array(manual_positions)

# Compute N_PROFILES after N_PARTIES is determined
N_PROFILES = math.factorial(N_PARTIES)

# Now load or define delta_matrix
if party_input_mode == "Preset Scenario":
    delta_matrix = SCENARIOS[selected_scenario]["delta_matrix"]
    # Validate delta_matrix shape
    if delta_matrix.shape != (N_PROFILES, N_PROFILES):
        st.sidebar.warning(f"Selected scenario's delta_matrix is {delta_matrix.shape}, but {N_PARTIES} parties require a {N_PROFILES}x{N_PROFILES} matrix. Using default identity matrix.")
        delta_matrix = np.eye(N_PROFILES)
else:
    # For manual entry, let the user define a custom delta_matrix or use a default
    st.sidebar.info("Since party positions are manually defined, a default identity delta matrix is used.")
    delta_matrix = np.eye(N_PROFILES)
    # Optionally, allow advanced users to define a custom delta_matrix (not implemented here for simplicity)

with st.sidebar.expander("🔍 View Selected Configuration"):
    st.subheader("Party Positions")
    party_positions_df = pd.DataFrame(
        party_positions,
        columns=["X", "Y"],
        index=[f"Party {i+1}" for i in range(N_PARTIES)]
    )
    st.dataframe(party_positions_df)

    st.subheader("Delta Matrix")
    delta_matrix_df = pd.DataFrame(
        delta_matrix,
        columns=[f"Profile {i}" for i in range(N_PROFILES)],
        index=[f"Profile {i}" for i in range(N_PROFILES)]
    )
    st.dataframe(delta_matrix_df)

frame_duration = st.sidebar.slider("Animation Speed (ms)", 100, 2000, 500, step=100)
N_AGENTS = st.sidebar.slider("Number of Agents", 50, 2000, 200, step=10)
T = st.sidebar.slider("Number of Iterations", 50, 1000, 300, step=50)
N_FRAMES = st.sidebar.slider("Number of Frames", 10, 100, 30, step=5)

# Fixed to dynamic to align with MATLAB
preference_update_mode = "dynamic"

# Initialize saved simulations list
if 'saved_simulations' not in st.session_state:
    st.session_state['saved_simulations'] = []

# Placeholder for progress bar and status text
progress_container = st.empty()
status_container = st.empty()

# Simulation control buttons
col1, col2 = st.sidebar.columns(2)
if col1.button("▶️ Start", key="start_simulation"):
    progress_bar = progress_container.progress(0.0)
    status_container.write("Starting simulation...")
    st.session_state['cancel_simulation'] = False
    st.session_state['simulation_running'] = True
    try:
        positions_record, polarisation_df, voting_df, social_choice_df = run_simulation(
            N_AGENTS, T, N_FRAMES, party_positions, delta_matrix, preference_update_mode
        )
        progress_update_interval = max(1, T // 50)
        for t in range(1, T + 1):
            if t % progress_update_interval == 0 or t == T:
                progress_bar.progress(t / T)
                status_container.write(f"Running iteration {t} of {T}")
                time.sleep(0.05)
        if positions_record is not None:
            status_container.write("Simulation completed!")
            st.session_state.positions_record = positions_record
            st.session_state.polarisation_df = polarisation_df
            st.session_state.voting_df = voting_df
            st.session_state.social_choice_df = social_choice_df
            st.session_state.N_PARTIES = N_PARTIES
            st.session_state.party_positions = party_positions
            st.session_state.current_scenario = selected_scenario if party_input_mode == "Preset Scenario" else "Manual"
            st.session_state.current_delta_matrix = delta_matrix
            st.session_state.simulation_running = False
        else:
            status_container.write("Simulation cancelled.")
            progress_bar.progress(0.0)
            st.session_state.simulation_running = False
    except ValueError as e:
        st.error(f"Simulation failed: {str(e)}")
        status_container.write("Simulation failed due to configuration error.")
        progress_bar.progress(0.0)
        st.session_state.simulation_running = False

if col2.button("⏹️ Stop/Cancel", key="stop_cancel_simulation"):
    if st.session_state.get('simulation_running', False):
        st.session_state['cancel_simulation'] = True
        status_container.write("Cancelling simulation...")
    else:
        st.session_state.positions_record = None
        st.session_state.party_positions = None
        progress_container.empty()
        status_container.empty()

# Reset Parameters button
if st.sidebar.button("🔄 Reset Parameters", key="reset_parameters"):
    st.session_state['N_AGENTS'] = 200
    st.session_state['T'] = 300
    st.session_state['N_FRAMES'] = 30
    st.session_state['frame_duration'] = 500
    st.rerun()

# Results Management
with st.sidebar.expander("💾 Results Management"):
    if 'positions_record' in st.session_state and st.session_state.positions_record is not None:
        simulation_name = st.text_input("Simulation Name", value=f"Run {len(st.session_state['saved_simulations']) + 1}")
        if st.button("Save Results", key="save_results"):
            if simulation_name:
                saved_simulation = {
                    "name": simulation_name,
                    "scenario": st.session_state.current_scenario,
                    "party_positions": st.session_state.party_positions,
                    "delta_matrix": st.session_state.current_delta_matrix,
                    "polarisation_df": st.session_state.polarisation_df,
                    "voting_df": st.session_state.voting_df,
                    "social_choice_df": st.session_state.social_choice_df
                }
                st.session_state['saved_simulations'].append(saved_simulation)
                st.success(f"Saved simulation: {simulation_name}")
            else:
                st.warning("Please enter a simulation name.")

    st.write(f"Saved Simulations: {len(st.session_state['saved_simulations'])}")
    if st.session_state['saved_simulations']:
        saved_names = [sim["name"] for sim in st.session_state['saved_simulations']]
        st.write("Saved Runs:", ", ".join(saved_names))
        st.info("Note: Excel sheet names are truncated to fit the 31-character limit (e.g., 'Scenario 1 - Ideal W_party_pos').")

        # Checkboxes for selecting simulations to export
        st.write("Select Simulations to Export:")
        selected_simulations = []
        for idx, sim in enumerate(st.session_state['saved_simulations']):
            if st.checkbox(f"Export '{sim['name']}'", value=True, key=f"export_sim_{idx}"):
                selected_simulations.append(sim)

        if st.button("Export Selected Results", key="export_selected_results"):
            if not selected_simulations:
                st.warning("Please select at least one simulation to export.")
            else:
                excel_buffer = io.BytesIO()
                try:
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        metadata_rows = []
                        for sim in selected_simulations:
                            base_name = sim['name'][:21]
                            metadata_rows.append({
                                "Simulation Name": sim["name"],
                                "Scenario": sim["scenario"]
                            })
                            party_positions_df = pd.DataFrame(
                                sim["party_positions"],
                                columns=["X", "Y"],
                                index=[f"Party {i+1}" for i in range(len(sim["party_positions"]))]
                            )
                            sheet_name = f"{base_name}_party_pos"
                            party_positions_df.to_excel(writer, sheet_name=sheet_name)
                            delta_matrix_df = pd.DataFrame(
                                sim["delta_matrix"],
                                columns=[f"Profile {i}" for i in range(sim["delta_matrix"].shape[0])],
                                index=[f"Profile {i}" for i in range(sim["delta_matrix"].shape[0])]
                            )
                            sheet_name = f"{base_name}_delta"
                            delta_matrix_df.to_excel(writer, sheet_name=sheet_name)
                            sim["polarisation_df"].to_excel(writer, sheet_name=f"{base_name}_polar", index=False)
                            sim["voting_df"].to_excel(writer, sheet_name=f"{base_name}_voting", index=False)
                            sim["social_choice_df"].to_excel(writer, sheet_name=f"{base_name}_social", index=False)
                        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="Metadata", index=False)
                except ModuleNotFoundError as e:
                    if "xlsxwriter" in str(e):
                        st.warning("The 'xlsxwriter' library is not installed. Attempting to use 'openpyxl' instead.")
                        try:
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                metadata_rows = []
                                for sim in selected_simulations:
                                    base_name = sim['name'][:21]
                                    metadata_rows.append({
                                        "Simulation Name": sim["name"],
                                        "Scenario": sim["scenario"]
                                    })
                                    party_positions_df = pd.DataFrame(
                                        sim["party_positions"],
                                        columns=["X", "Y"],
                                        index=[f"Party {i+1}" for i in range(len(sim["party_positions"]))]
                                    )
                                    sheet_name = f"{base_name}_party_pos"
                                    party_positions_df.to_excel(writer, sheet_name=sheet_name)
                                    delta_matrix_df = pd.DataFrame(
                                        sim["delta_matrix"],
                                        columns=[f"Profile {i}" for i in range(sim["delta_matrix"].shape[0])],
                                        index=[f"Profile {i}" for i in range(sim["delta_matrix"].shape[0])]
                                    )
                                    sheet_name = f"{base_name}_delta"
                                    delta_matrix_df.to_excel(writer, sheet_name=sheet_name)
                                    sim["polarisation_df"].to_excel(writer, sheet_name=f"{base_name}_polar", index=False)
                                    sim["voting_df"].to_excel(writer, sheet_name=f"{base_name}_voting", index=False)
                                    sim["social_choice_df"].to_excel(writer, sheet_name=f"{base_name}_social", index=False)
                                pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="Metadata", index=False)
                        except ModuleNotFoundError:
                            st.error("Neither 'xlsxwriter' nor 'openpyxl' is installed. Please install one of them to export Excel files.\n"
                                     "Install xlsxwriter: `conda install xlsxwriter` or `pip install xlsxwriter`\n"
                                     "Install openpyxl: `conda install openpyxl` or `pip install openpyxl`")
                            excel_buffer = None
                    else:
                        st.error(f"Error during export: {str(e)}")
                        excel_buffer = None
                
                if excel_buffer is not None:
                    excel_buffer.seek(0)
                    st.download_button(
                        label="Download Selected Results",
                        data=excel_buffer,
                        file_name="selected_simulation_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_selected_results"
                    )

        st.write("---")
        st.write("Clear Saved Simulations")
        confirm_reset = st.checkbox("Confirm Clear All Simulations")
        if st.button("Clear Saved Simulations", key="clear_simulations"):
            if confirm_reset:
                st.session_state['saved_simulations'] = []
                st.success("All saved simulations have been cleared.")
            else:
                st.warning("Please confirm to clear all saved simulations.")

# Documentation Section
with st.sidebar.expander("📜 Documentation"):
    st.write("Download the user manual for this simulation dashboard.")
    manual_content = """
# Agent-Based Polarisation Simulation Dashboard User Manual

## Polarization viewed from a Social choice Perspective

The Agent-Based Polarisation Simulation Dashboard is a Streamlit-based tool designed to model opinion dynamics in a 2D opinion space. It simulates how agents interact, form preferences, and evolve their opinions over time, allowing users to analyze polarization, voting behavior, and social choice outcomes. The dashboard provides interactive visualizations, parameter controls, and result export options to facilitate research and analysis of social dynamics.

This manual is for version `app_final_v04.py` (updated April 21, 2025).

## Getting Started

### Prerequisites
- **Python Environment**: Ensure Python 3.12 is installed with the following dependencies:
  - `streamlit==1.37.1`
  - `numpy==1.26.4`
  - `pandas==2.2.2`
  - `plotly==5.22.0`
  - `xlsxwriter==3.2.0`
  - `openpyxl==3.1.5`
- **Project Files**: Place `app_final_v04.py`, `agents.py`, `scenarios.py`, `config.py`, `polarisation.py`, and `deliberation.py` in the same directory.
- **Run the App**: Execute the following command in your terminal:
  ```
  streamlit run app_final_v04.py
  ```

### Dashboard Layout
- **Sidebar**: Contains simulation controls, results management, and documentation.
- **Main Interface**: Displays simulation results in three tabs (Scatter Animation, Polarisation Metrics, Voting Results) and a section for downloading individual result tables.

## Simulation Settings

- **Party Positions**:
  - Select "Preset Scenario" to use a predefined scenario and its party positions.
  - Select "Manual Entry" to define 2–5 parties with custom X and Y coordinates.
- **Delta Matrix**:
  - If using a preset scenario, the delta matrix is loaded from the scenario.
  - If using manual entry, a default identity matrix is used.
- **Sliders**:
  - **Animation Speed (ms)**: Adjust the speed of the animation (100–2000 ms, default: 500 ms).
  - **Number of Agents**: Set the number of agents (50–2000, default: 200).
  - **Number of Iterations**: Set the total iterations (50–1000, default: 300).
  - **Number of Frames**: Set the number of animation frames (10–100, default: 30).
- **View Selected Configuration**:
  - Expand this section to see the party positions and delta matrix as tables.

### Running the Simulation
- **Start**: Click the "▶️ Start" button to run the simulation.
- **Stop/Cancel**: Click "⏹️ Stop/Cancel" to interrupt the simulation or clear results.
- **Reset Parameters**: Click "🔄 Reset Parameters" to revert to default settings.

## Visualizations

### Tab 1: Scatter Animation
- Displays an interactive animation of agents, party positions, party centers, and the society center in a 2D opinion space.
- **Agents**: Colored dots based on their first-choice party.
- **Party Positions**: Shown as stars.
- **Party Centers**: Shown as diamonds.
- **Society Center**: Shown as a black square.

### Tab 2: Polarisation Metrics
- Shows four polarization metrics over time: Party Polarisation, Preference Polarisation, Binary Polarisation, and Kemeny Polarisation.
- Use the dropdown to switch between metrics or view all.

### Tab 3: Voting Results
- **Party Support Over Time**: Line plot of voting shares.
- **Final Support Distribution**: Bar chart of final voting shares.
- **Preference Profile Distribution**: Line plot of preference profiles over time.
- **Social Choice Results**: Line plots for social choice winners and scores.
- **Final Social Choice Winners**: Table listing the final winners.

## Results Management

### Saving Results
- Enter a name in the "Simulation Name" field and click "Save Results" to store the simulation.

### Exporting Results
- Select simulations using checkboxes and click "Export Selected Results" to download an Excel file.
- **Excel File Structure**:
  - **Metadata Sheet**: Lists simulation names and scenarios.
  - **Per-Simulation Sheets**: Party positions, delta matrix, polarization metrics, voting results, and social choice results.

### Clearing Saved Simulations
- Check "Confirm Clear All Simulations" and click "Clear Saved Simulations" to remove all saved simulations.

## Downloading Individual Results
- Expand the "Download Simulation Results" section to download individual tables as Excel files.

## Accessing the User Manual
- Expand the "Documentation" section and click "Download User Manual" to download this manual as `user_manual.md`.

## Troubleshooting
- **Excel Export Fails**: Ensure `xlsxwriter` or `openpyxl` is installed.
- **Simulation Fails**: Check for errors in the sidebar (e.g., mismatched delta matrix dimensions) and adjust parameters accordingly.

## Contact and Support
For support, contact the developer at [insert contact information].

---

**developed by Emre Erdoğan, Pınar Uyan-Semerci, Ayça Ebru Giritligil, Onur Doğan & Giray Girengir-2025**
    """
    st.download_button(
        label="Download User Manual",
        data=manual_content,
        file_name="user_manual.md",
        mime="text/markdown",
        key="download_manual"
    )

    st.write("Download the technical document (DOCX) for detailed formulae and methodology.")
    try:
        with open("technical_document.docx", "rb") as docx_file:
            docx_data = docx_file.read()
        st.download_button(
            label="Download Technical Document (DOCX)",
            data=docx_data,
            file_name="technical_document.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_technical_doc"
        )
    except FileNotFoundError:
        st.warning("Technical document DOCX not found. Please ensure 'technical_document.docx' is in the project directory.")
        st.info("To generate the DOCX file:\n"
                "1. Save the Markdown file 'technical_document.md' provided in the documentation.\n"
                "2. Install Pandoc (download from pandoc.org/installing.html) and ensure it's added to your PATH.\n"
                "3. Open a terminal in the project directory and run:\n"
                "   `pandoc technical_document.md -o technical_document.docx --mathml`\n"
                "4. The file 'technical_document.docx' will be created in the directory.\n"
                "Alternatively, use an online Markdown-to-DOCX converter or manually convert the Markdown to DOCX in Microsoft Word, inserting equations as needed.")

# Advanced Controls
with st.sidebar.expander("⚙️ Advanced Controls"):
    if st.button("🧹 Clear Cache", key="clear_cache"):
        st.cache_data.clear()
        st.rerun()

# --- Visualization ---
if 'positions_record' in st.session_state and st.session_state.positions_record is not None:
    positions_record = st.session_state.positions_record
    polarisation_df = st.session_state.polarisation_df
    voting_df = st.session_state.voting_df
    social_choice_df = st.session_state.social_choice_df
    N_PARTIES = st.session_state.N_PARTIES
    party_positions = st.session_state.get('party_positions', party_positions)

    tab1, tab2, tab3 = st.tabs(["Scatter Animation", "Polarisation Metrics", "Voting Results"])

    with tab1:
        st.subheader("Scatter Animation of Agents, Party Positions, Party Centers, and Society Center")
        party_colors = px.colors.qualitative.Plotly[:N_PARTIES]
        agent_frames, party_center_frames, society_center_frames = zip(*positions_record)
        positions_df = pd.concat(agent_frames, ignore_index=True)
        positions_df["FirstChoice"] = positions_df["FirstChoice"].astype(str)
        party_centers_df = pd.concat(party_center_frames, ignore_index=True)
        society_centers_df = pd.concat(society_center_frames, ignore_index=True)

        frames = []
        unique_iterations = sorted(positions_df['Iteration'].unique())

        initial_data = []
        for party in [str(i) for i in range(N_PARTIES)]:
            df_party = positions_df[positions_df["Iteration"] == unique_iterations[0]]
            df_party = df_party[df_party["FirstChoice"] == party]
            initial_data.append(go.Scatter(
                x=df_party["x"], y=df_party["y"],
                mode="markers",
                marker=dict(size=6, color=party_colors[int(party)], symbol="circle"),
                name=f"Party {int(party)+1} Agents"
            ))
        initial_data.append(go.Scatter(
            x=party_positions[:, 0], y=party_positions[:, 1],
            mode="markers",
            marker=dict(size=12, color=party_colors[:N_PARTIES], symbol="star", line=dict(width=2, color="black")),
            name="Party Positions"
        ))
        df_centers = party_centers_df[party_centers_df["Iteration"] == unique_iterations[0]]
        for i in range(N_PARTIES):
            center = df_centers[df_centers["Party"] == str(i)]
            initial_data.append(go.Scatter(
                x=center["x"], y=center["y"],
                mode="markers",
                marker=dict(size=10, color=party_colors[i], symbol="diamond", line=dict(width=1, color="black")),
                name=f"Party {i+1} Center"
            ))
        df_society = society_centers_df[society_centers_df["Iteration"] == unique_iterations[0]]
        initial_data.append(go.Scatter(
            x=df_society["x"], y=df_society["y"],
            mode="markers",
            marker=dict(size=12, color="black", symbol="square", line=dict(width=2, color="white")),
            name="Society Center"
        ))

        for iteration in unique_iterations:
            frame_data = []
            df_iter = positions_df[positions_df['Iteration'] == iteration]
            for party in [str(i) for i in range(N_PARTIES)]:
                df_party = df_iter[df_iter["FirstChoice"] == party]
                frame_data.append(go.Scatter(
                    x=df_party["x"], y=df_party["y"],
                    mode="markers",
                    marker=dict(size=6, color=party_colors[int(party)], symbol="circle"),
                    name=f"Party {int(party)+1} Agents"
                ))
            frame_data.append(go.Scatter(
                x=party_positions[:, 0], y=party_positions[:, 1],
                mode="markers",
                marker=dict(size=12, color=party_colors[:N_PARTIES], symbol="star", line=dict(width=2, color="black")),
                name="Party Positions"
            ))
            df_centers = party_centers_df[party_centers_df["Iteration"] == iteration]
            for i in range(N_PARTIES):
                center = df_centers[df_centers["Party"] == str(i)]
                frame_data.append(go.Scatter(
                    x=center["x"], y=center["y"],
                    mode="markers",
                    marker=dict(size=10, color=party_colors[i], symbol="diamond", line=dict(width=1, color="black")),
                    name=f"Party {i+1} Center"
                ))
            df_society = society_centers_df[society_centers_df["Iteration"] == iteration]
            frame_data.append(go.Scatter(
                x=df_society["x"], y=df_society["y"],
                mode="markers",
                marker=dict(size=12, color="black", symbol="square", line=dict(width=2, color="white")),
                name="Society Center"
            ))
            frames.append(go.Frame(
                data=frame_data,
                name=str(iteration),
                layout=go.Layout(title=f"Opinion Space: Iteration {iteration}")
            ))

        fig = go.Figure(data=initial_data, frames=frames)
        fig.update_layout(
            xaxis=dict(range=[-OPINION_SPACE_SIZE/2 - 0.5, OPINION_SPACE_SIZE/2 + 0.5], title="X Opinion"),
            yaxis=dict(range=[-OPINION_SPACE_SIZE/2 - 0.5, OPINION_SPACE_SIZE/2 + 0.5], title="Y Opinion"),
            width=800,
            height=800,
            title=f"Opinion Space: Iteration {unique_iterations[0]}",
            showlegend=True,
            shapes=[
                dict(
                    type="rect",
                    x0=-OPINION_SPACE_SIZE/2,
                    y0=-OPINION_SPACE_SIZE/2,
                    x1=OPINION_SPACE_SIZE/2,
                    y1=OPINION_SPACE_SIZE/2,
                    line=dict(color="black", width=2, dash="dash"),
                    layer="below"
                )
            ],
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "▶️ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "⏸️ Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]
                    }
                ]
            }]
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Polarisation Metrics Over Time")
        fig = go.Figure()
        metrics = ["PartyPolarisation", "PrefPolarisation", "BinaryPolarisation", "KemenyPolarisation"]
        for metric in metrics:
            fig.add_trace(go.Scatter(
                x=polarisation_df["Iteration"],
                y=polarisation_df[metric],
                name=metric,
                visible=(metric == metrics[0])
            ))
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            label=metric,
                            method="update",
                            args=[{"visible": [m == metric for m in metrics]}]
                        ) for metric in metrics
                    ] + [dict(
                        label="All Metrics",
                        method="update",
                        args=[{"visible": [True] * len(metrics)}]
                    )],
                    direction="down",
                    showactive=True,
                )
            ],
            xaxis_title="Iteration",
            yaxis_title="Polarisation Value",
            showlegend=True,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Party Support Over Time")
        fig = go.Figure()
        for i in range(N_PARTIES):
            party = f"Party {i+1}"
            col = i if i in voting_df.columns else None
            if col is not None:
                fig.add_trace(go.Scatter(
                    x=voting_df["Iteration"],
                    y=voting_df[i],
                    name=party,
                    line=dict(color=party_colors[i]),
                    visible=(i == 0)
                ))
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            label=f"Party {i+1}",
                            method="update",
                            args=[{"visible": [j == i for j in range(N_PARTIES)]}]
                        ) for i in range(N_PARTIES)
                    ] + [dict(
                        label="All Parties",
                        method="update",
                        args=[{"visible": [True] * N_PARTIES}]
                    )],
                    direction="down",
                    showactive=True,
                )
            ],
            xaxis_title="Iteration",
            yaxis_title="Voting Share",
            showlegend=True,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Final Support Distribution")
        final_support = voting_df[voting_df["Iteration"] == voting_df["Iteration"].max()].drop(columns=["Iteration"]).T
        final_support.columns = ["Support"]
        final_support.index = [f"Party {i+1}" for i in range(N_PARTIES)]
        fig_bar = px.bar(
            final_support,
            x=final_support.index,
            y="Support",
            title="Final Party Support",
            color=final_support.index,
            color_discrete_sequence=party_colors[:N_PARTIES]
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Preference Profile Distribution Over Time")
        pref_dist = positions_df.groupby(["Iteration", "PrefIndex"]).size().reset_index(name="Count")
        pref_dist["PrefIndex"] = pref_dist["PrefIndex"].astype(str)
        fig_pref = px.line(pref_dist, x="Iteration", y="Count", color="PrefIndex",
                           title="Preference Profile Support Over Time")
        st.plotly_chart(fig_pref, use_container_width=True)

        st.subheader("Social Choice Results Over Time")
        winner_df = social_choice_df[["Iteration", "PluralityWinner", "BordaWinner", "MajCompWinner", "CopelandWinner"]]
        winner_df = winner_df.rename(columns={
            "PluralityWinner": "Plurality",
            "BordaWinner": "Borda",
            "MajCompWinner": "Maj. Comp.",
            "CopelandWinner": "Copeland"
        })
        fig_winners = px.line(
            winner_df.set_index("Iteration"),
            title="Social Choice Winners Over Time",
            labels={"value": "Winning Party", "variable": "Rule"}
        )
        fig_winners.update_yaxes(tickvals=list(range(N_PARTIES)), ticktext=[f"Party {i+1}" for i in range(N_PARTIES)])
        st.plotly_chart(fig_winners, use_container_width=True)

        for rule, score_prefix in [
            ("Plurality", "PluralityVotes"),
            ("Borda", "BordaScores"),
            ("Maj. Comp.", "MajCompScores"),
            ("Copeland", "CopelandScores")
        ]:
            score_cols = {f"{score_prefix}{i}": f"Party {i+1}" for i in range(N_PARTIES)}
            score_df = social_choice_df[["Iteration"] + [f"{score_prefix}{i}" for i in range(N_PARTIES)]]
            score_df = score_df.rename(columns=score_cols)
            fig_scores = px.line(
                score_df.set_index("Iteration"),
                title=f"{rule} Scores Over Time",
                labels={"value": "Score", "variable": "Party"},
                color_discrete_sequence=party_colors[:N_PARTIES]
            )
            st.plotly_chart(fig_scores, use_container_width=True)

        st.subheader("Final Social Choice Winners")
        final_winners = social_choice_df[social_choice_df["Iteration"] == social_choice_df["Iteration"].max()]
        final_winners = final_winners[["PluralityWinner", "BordaWinner", "MajCompWinner", "CopelandWinner"]].iloc[0]
        final_winners.index = ["Plurality", "Borda", "Maj. Comp.", "Copeland"]
        final_winners = final_winners.apply(lambda x: f"Party {int(x)+1}")
        st.table(final_winners.rename("Winner"))

    with st.expander("📥 Download Simulation Results"):
        excel_buffer_polar = io.BytesIO()
        polarisation_df.to_excel(excel_buffer_polar, index=False)
        excel_buffer_polar.seek(0)
        st.download_button(
            label="Download Polarisation Results",
            data=excel_buffer_polar,
            file_name="polarisation_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_polarisation_results"
        )

        excel_buffer_vote = io.BytesIO()
        voting_df.to_excel(excel_buffer_vote, index=False)
        excel_buffer_vote.seek(0)
        st.download_button(
            label="Download Voting Results",
            data=excel_buffer_vote,
            file_name="voting_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_voting_results"
        )

        excel_buffer_social = io.BytesIO()
        social_choice_df.to_excel(excel_buffer_social, index=False)
        excel_buffer_social.seek(0)
        st.download_button(
            label="Download Social Choice Results",
            data=excel_buffer_social,
            file_name="social_choice_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_social_choice_results"
        )

# Footer with Credits
st.divider()
st.markdown(
    "<div style='text-align: center; padding: 10px;'>"
    "developed by Emre Erdoğan, Pınar Uyan-Semerci, Ayça Ebru Giritligil, Onur Doğan & Giray Girengir-2025"
    "</div>",
    unsafe_allow_html=True
)
