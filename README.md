 This project implements an agent-based simulation to model polarisation dynamics in a two-dimensional opinion space, replicating a MATLAB model in Python using Streamlit.

 ## Installation

 1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
 2. Activate your Conda environment:
    ```bash
    conda activate miniconda31
    ```
 3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
 4. Run the application:
    ```bash
    streamlit run app_final_v04.py
    ```

 ## Dependencies

 See `requirements.txt` for a list of required Python packages.

 ## Usage

 - Select a scenario or manually define party positions in the sidebar.
 - Adjust simulation parameters (number of agents, iterations, frames).
 - Click "Start Simulation" to run the simulation and view results in three tabs:
   - **Scatter Animation**: Visualizes agent movements and party centers.
   - **Polarisation Metrics**: Displays polarisation metrics over time.
   - **Voting Results**: Shows party support and preference profile distributions.

 ## Files

 - `app_final_v04.py`: Main Streamlit application.
 - `polarisation.py`: Calculates polarisation metrics.
 - `deliberation.py`: Implements deliberation step with unit circle reaction vectors.
 - `agents.py`: Defines the Agents class with preference profile indexing.
 - `config.py`: Configuration parameters.
 - `scenarios.py`: Predefined scenarios for delta matrices and party positions.
