# Agent-Based Polarisation Simulation Dashboard User Manual

## Polarization viewed from a Social choice Perspective

The Agent-Based Polarisation Simulation Dashboard is a Streamlit-based tool designed to model opinion dynamics in a 2D opinion space. It simulates how agents interact, form preferences, and evolve their opinions over time, allowing users to analyze polarization, voting behavior, and social choice outcomes. The dashboard provides interactive visualizations, parameter controls, and result export options to facilitate research and analysis of social dynamics.

This manual is for version `app_final_v06.py` (updated April 21, 2025), which includes features like selective export of saved simulations and a downloadable user manual.

## Getting Started

### Prerequisites
- **Python Environment**: Ensure Python 3.11 is installed with the following dependencies (as specified in `requirements.txt` or `environment.yml`):
  - `streamlit==1.37.1`
  - `numpy==1.26.4`
  - `pandas==2.2.2`
  - `plotly==5.22.0`
  - `xlsxwriter==3.2.0`
  - `openpyxl==3.1.5`
- **Project Files**: Place `app_final_v06.py`, `agents.py`, `scenarios.py`, `config.py`, `polarisation.py`, and `deliberation.py` in the same directory.
- **Run the App**: Execute the following command in your terminal:
  ```
  streamlit run app_final_v06.py
  ```
  The app will open in your default web browser.

### Dashboard Layout
- **Sidebar**: Contains simulation controls, results management, documentation, and advanced settings.
- **Main Interface**: Displays simulation results in three tabs (Scatter Animation, Polarisation Metrics, Voting Results) and a section for downloading individual result tables.
- **Screenshot Placeholder**: [Insert a screenshot of the dashboard’s home screen showing the sidebar and main interface.]

## Simulation Settings

The sidebar’s "Simulation Settings" section allows you to configure the simulation parameters.

- **Select Scenario for Delta Matrix**:
  - Choose a predefined scenario (e.g., "Scenario 1 - Ideal World") from the dropdown to set the interaction matrix (`delta_matrix`).
  - Scenarios define how agents with different preference profiles interact.
- **Party Positions**:
  - Select "Preset Scenario" to use party positions from the chosen scenario.
  - Select "Manual Entry" to define 2–5 parties with custom X and Y coordinates using sliders.
- **Sliders**:
  - **Animation Speed (ms)**: Adjust the speed of the animation (100–2000 ms, default: 500 ms).
  - **Number of Agents**: Set the number of agents (50–2000, default: 200).
  - **Number of Iterations**: Set the total iterations (50–1000, default: 300).
  - **Number of Frames**: Set the number of animation frames (10–100, default: 30).
- **View Selected Configuration**:
  - Expand this section to see the party positions and delta matrix as tables.
  - **Screenshot Placeholder**: [Insert a screenshot of the expanded "View Selected Configuration" section showing the party positions and delta matrix tables.]

### Running the Simulation
- **Start**: Click the "▶️ Start" button to run the simulation with the configured parameters.
- **Stop/Cancel**: Click "⏹️ Stop/Cancel" to interrupt the simulation or clear results.
- **Reset Parameters**: Click "🔄 Reset Parameters" to revert to default settings (200 agents, 300 iterations, 30 frames, 500 ms animation speed).

## Visualizations

After running a simulation, results are displayed in three tabs in the main interface.

### Tab 1: Scatter Animation
- Displays an interactive animation of agents, party positions, party centers, and the society center in a 2D opinion space.
- **Agents**: Represented as colored dots based on their first-choice party.
- **Party Positions**: Shown as stars.
- **Party Centers**: Shown as diamonds, indicating the mean position of agents supporting each party.
- **Society Center**: Shown as a black square, representing the mean position of all agents.
- **Controls**: Use the "▶️ Play" and "⏸️ Pause" buttons to control the animation.
- **Screenshot Placeholder**: [Insert a screenshot of the Scatter Animation tab with the animation playing.]

### Tab 2: Polarisation Metrics
- Shows four polarization metrics over time: Party Polarisation, Preference Polarisation, Binary Polarisation, and Kemeny Polarisation.
- Use the dropdown to switch between metrics or select "All Metrics" to view them together.
- **Screenshot Placeholder**: [Insert a screenshot of the Polarisation Metrics tab showing the dropdown and a metric plot.]

### Tab 3: Voting Results
- **Party Support Over Time**: Line plot showing the voting share of each party over iterations. Use the dropdown to switch between parties or view all.
- **Final Support Distribution**: Bar chart showing the final voting share of each party.
- **Preference Profile Distribution**: Line plot showing the distribution of preference profiles over time.
- **Social Choice Results**: Line plots for social choice winners (Plurality, Borda, Majority Comparison, Copeland) and their scores over time.
- **Final Social Choice Winners**: Table listing the final winners under each rule.
- **Screenshot Placeholder**: [Insert a screenshot of the Voting Results tab showing the party support plot and final winners table.]

## Results Management

The "Results Management" section in the sidebar allows you to save, export, and clear simulation results.

### Saving Results
- After running a simulation, enter a name in the "Simulation Name" field (e.g., "Run 1").
- Click "Save Results" to store the simulation in memory.
- Saved simulations are listed below as "Saved Runs."

### Exporting Results
- **Selective Export**:
  - Checkboxes appear for each saved simulation (e.g., "Export 'Run 1'").
  - Select the simulations you want to export by checking their boxes (by default, all are selected).
  - Click "Export Selected Results" to download an Excel file (`selected_simulation_results.xlsx`).
  - If no simulations are selected, a warning appears: "Please select at least one simulation to export."
- **Excel File Structure**:
  - **Metadata Sheet**: Lists the names and scenarios of the selected simulations.
  - **Per-Simulation Sheets**: For each selected simulation, sheets are created for:
    - Party positions (`_party_pos`)
    - Delta matrix (`_delta`)
    - Polarization metrics (`_polar`)
    - Voting results (`_voting`)
    - Social choice results (`_social`)
  - **Note**: Sheet names are truncated to fit Excel’s 31-character limit (e.g., `Scenario 1 - Ideal W_party_pos`).
- **Screenshot Placeholder**: [Insert a screenshot of the "Results Management" expander showing the checkboxes and export button.]

### Clearing Saved Simulations
- Check "Confirm Clear All Simulations" to enable the clear action.
- Click "Clear Saved Simulations" to remove all saved simulations from memory.
- If the confirmation checkbox is not checked, a warning appears: "Please confirm to clear all saved simulations."

## Downloading Individual Results
- Expand the "Download Simulation Results" section at the bottom of the main interface.
- Download individual tables as Excel files:
  - Polarisation Results (`polarisation_results.xlsx`)
  - Voting Results (`voting_results.xlsx`)
  - Social Choice Results (`social_choice_results.xlsx`)

## Accessing the User Manual
- In the sidebar, expand the "Documentation" section.
- Click "Download User Manual" to download this manual as `user_manual.md`.
- The manual can be opened in any Markdown viewer or text editor.
- **Screenshot Placeholder**: [Insert a screenshot of the "Documentation" expander showing the download button.]

## Advanced Controls
- Expand the "Advanced Controls" section in the sidebar.
- Click "🧹 Clear Cache" to clear the app’s cache and refresh the simulation.

## Troubleshooting
- **Excel Export Fails**: If the export fails, ensure `xlsxwriter` or `openpyxl` is installed:
  ```
  conda install xlsxwriter
  conda install openpyxl
  ```
- **Slow Performance**: For large simulations (e.g., 2000 agents, 1000 iterations), reduce the number of agents, iterations, or frames to improve performance.
- **Animation Not Displaying**: Ensure the simulation has completed successfully. Check the progress bar and status messages in the main interface.
- **MATLAB Alignment Issues**: Compare `agents.positions` and `agents.pref_indices` with MATLAB’s `cagentnew` and `pprofshnew` using identical inputs to ensure consistency.

## Contact and Support
For additional support, please contact the developer at [insert contact information]. Provide details about your issue, including any error messages and the simulation parameters used.

---

**developed by Emre Erdoğan, Pınar Uyan-Semerci, Ayça Ebru Giritligil, Onur Doğan & Giray Girengir-2025**