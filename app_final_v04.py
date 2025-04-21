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

# --- Simulation Function ---
def run_simulation(N_AGENTS, T, N_FRAMES, party_positions, delta_matrix, progress_bar, status_text, preference_update_mode="dynamic"):
    np.random.seed(42)
    N_PARTIES = len(party_positions)

    agents = Agents(N_AGENTS, OPINION_SPACE_SIZE)
    profiles = np.random.choice(6, N_AGENTS)  # 0-based indices {0, 1, 2, 3, 4, 5} for 6 profiles
    agents.pref_indices = profiles

    step_interval = max(1, T // N_FRAMES)
    record_iters = list(range(step_interval, T + 1, step_interval))
    
    # Define progress update interval (e.g., every 2% of T or at record_iters)
    progress_update_interval = max(1, T // 50)  # Update every ~2% of iterations

    positions_record = []
    polarisation_records = []
    voting_records = []
    social_choice_records = []

    for t in range(1, T + 1):
        agents.positions, agents.pref_indices = deliberation_step_matched(
            agents.positions, agents.pref_indices, delta_matrix, t, party_positions,
            opinion_space_size=OPINION_SPACE_SIZE,
            mu_a=MU_ATTRACTION,
            mu_r=MU_REACTION,
            discount_coeff=DISCOUNT_COEFF,
            interaction_rate=1.0
        )

        # Update progress bar and status text only at intervals
        if t % progress_update_interval == 0 or t == T:
            progress = t / T
            progress_bar.progress(progress)
            status_text.write(f"Running iteration {t} of {T}")

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
                st.error("calculate_polarisation_metrics fonksiyon parametrelerini kontrol et.")
                return None, None, None, None

            try:
                party_polar, pref_polar, binary_polar, kemeny_polar = calculate_polarisation_metrics(
                    agents.positions, preferences, first_choices, party_positions, N_AGENTS, OPINION_SPACE_SIZE, agents.pref_indices
                )
            except TypeError as e:
                st.error(f"calculate_polarisation_metrics hatası: {str(e)}")
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

    # Finalize progress bar and status text
    progress_bar.progress(1.0)
    status_text.write("Simulation completed!")

    polarisation_df = pd.DataFrame(polarisation_records)
    voting_df = pd.DataFrame(voting_records).fillna(0)
    social_choice_df = pd.DataFrame(social_choice_records)

    return positions_record, polarisation_df, voting_df, social_choice_df

# --- Streamlit Setup ---
st.set_page_config(page_title="Polarisation Simulation", layout="wide")
st.title("🥉 Agent-Based Polarisation Simulation")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Settings")

if st.sidebar.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.rerun()

selected_scenario = st.sidebar.selectbox("Select Scenario for Delta Matrix", list(SCENARIOS.keys()))
delta_matrix = SCENARIOS[selected_scenario]["delta_matrix"]

party_input_mode = st.sidebar.radio(
    "How to define party positions?",
    options=["Preset Scenario", "Manual Entry"],
    index=0
)

if party_input_mode == "Preset Scenario":
    party_positions = np.array(SCENARIOS[selected_scenario]["party_positions"])
else:
    N_PARTIES = st.sidebar.slider("Number of Parties", 2, 5, 3)
    manual_positions = []
    for i in range(N_PARTIES):
        x = st.sidebar.number_input(f"Party {i+1} - X Coordinate", value=float(i*2-2), step=0.1)
        y = st.sidebar.number_input(f"Party {i+1} - Y Coordinate", value=0.0, step=0.1)
        manual_positions.append([x, y])
    party_positions = np.array(manual_positions)

with st.sidebar.expander("🔍 View Selected Configuration"):
    st.subheader("Party Positions")
    party_positions_df = pd.DataFrame(
        party_positions,
        columns=["X", "Y"],
        index=[f"Party {i+1}" for i in range(len(party_positions))]
    )
    st.dataframe(party_positions_df)

    st.subheader("Delta Matrix")
    delta_matrix_df = pd.DataFrame(
        delta_matrix,
        columns=[f"Profile {i}" for i in range(6)],
        index=[f"Profile {i}" for i in range(6)]
    )
    st.dataframe(delta_matrix_df)

frame_duration = st.sidebar.slider("Animation Speed (ms)", 100, 2000, 500, step=100)
N_AGENTS = st.sidebar.slider("Number of Agents", 50, 2000, 200, step=10)
T = st.sidebar.slider("Number of Iterations", 50, 1000, 300, step=50)
N_FRAMES = st.sidebar.slider("Number of Frames", 10, 100, 30, step=5)

# Fixed to dynamic to align with MATLAB
preference_update_mode = "dynamic"

# Placeholder for progress bar and status text
progress_container = st.empty()
status_container = st.empty()

col1, col2 = st.sidebar.columns(2)
if col1.button("▶️ Start Simulation"):
    # Initialize progress bar and status text
    progress_bar = progress_container.progress(0.0)
    status_container.write("Starting simulation...")

    positions_record, polarisation_df, voting_df, social_choice_df = run_simulation(
        N_AGENTS, T, N_FRAMES, party_positions, delta_matrix, progress_bar, status_container, preference_update_mode
    )
    if positions_record is not None:
        st.session_state.positions_record = positions_record
        st.session_state.polarisation_df = polarisation_df
        st.session_state.voting_df = voting_df
        st.session_state.social_choice_df = social_choice_df
        st.session_state.N_PARTIES = len(party_positions)
        st.session_state.party_positions = party_positions
if col2.button("⏹️ Stop Simulation"):
    st.session_state.positions_record = None
    st.session_state.party_positions = None
    progress_container.empty()
    status_container.empty()

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
        for metric in ["PartyPolarisation", "PrefPolarisation", "BinaryPolarisation", "KemenyPolarisation"]:
            fig = px.line(polarisation_df, x="Iteration", y=metric, title=f"{metric} Over Time")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Party Support Over Time")
        voting_df_renamed = voting_df.rename(columns={i: f"Party {i+1}" for i in range(N_PARTIES)})
        voting_fig = px.line(
            voting_df_renamed.set_index("Iteration"),
            title="Party Voting Share Over Time",
            labels={"value": "Voting Share", "variable": "Party"}
        )
        st.plotly_chart(voting_fig, use_container_width=True)

        st.subheader("Final Support Distribution")
        final_support = voting_df[voting_df["Iteration"] == voting_df["Iteration"].max()].drop(columns=["Iteration"]).T
        final_support.columns = ["Support"]
        final_support.index = [f"Party {i+1}" for i in range(N_PARTIES)]
        fig_bar = px.bar(final_support, x=final_support.index, y="Support", title="Final Party Support")
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
                labels={"value": "Score", "variable": "Party"}
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
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        excel_buffer_vote = io.BytesIO()
        voting_df.to_excel(excel_buffer_vote, index=False)
        excel_buffer_vote.seek(0)
        st.download_button(
            label="Download Voting Results",
            data=excel_buffer_vote,
            file_name="voting_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        excel_buffer_social = io.BytesIO()
        social_choice_df.to_excel(excel_buffer_social, index=False)
        excel_buffer_social.seek(0)
        st.download_button(
            label="Download Social Choice Results",
            data=excel_buffer_social,
            file_name="social_choice_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Click 'Start Simulation' to run the simulation and view results.")
