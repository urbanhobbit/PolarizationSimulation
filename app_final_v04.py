# --- Imports ---
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from agents import Agents
from scenarios import SCENARIOS
from config import *
from polarisation import calculate_polarisation_metrics
from deliberation import deliberation_step_matched
import io
import inspect

# --- Simulation Function ---
# @st.cache_data  # Geçici olarak kaldırıldı, testten sonra geri eklenebilir
def run_simulation(N_AGENTS, T, N_FRAMES, party_positions, delta_matrix):
    np.random.seed(42)
    N_PARTIES = len(party_positions)

    agents = Agents(N_AGENTS, OPINION_SPACE_SIZE)
    profiles = np.random.choice(6, N_AGENTS)  # 6 profil, sabit

    step_interval = max(1, T // N_FRAMES)
    record_iters = list(range(step_interval, T + 1, step_interval))

    positions_record = []
    polarisation_records = []
    voting_records = []

    for t in range(1, T + 1):
        agents.positions = deliberation_step_matched(
            agents.positions, profiles, delta_matrix, t,
            opinion_space_size=OPINION_SPACE_SIZE,
            mu_a=MU_ATTRACTION,
            mu_r=MU_REACTION,
            discount_coeff=DISCOUNT_COEFF,
            interaction_rate=INTERACTION_RATE
        )

        if t in record_iters:
            preferences = agents.update_preferences(party_positions)
            first_choices = preferences[:, 0]
            pref_indices = agents.pref_indices  # Tercih indeksleri

            frame = pd.DataFrame({
                "x": agents.positions[:, 0],
                "y": agents.positions[:, 1],
                "FirstChoice": first_choices,
                "PrefIndex": pref_indices,  # Yeni sütun
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

            # Hata ayıklama: Fonksiyon imzasını kontrol et
            sig = inspect.signature(calculate_polarisation_metrics)
            expected_args = len(sig.parameters)
            if expected_args != 7:
                st.error(f"calculate_polarisation_metrics fonksiyonu {expected_args} argüman bekliyor, ancak 7 argüman gerekiyor. Lütfen polarisation.py dosyasını kontrol edin.")
                return None, None, None

            try:
                party_polar, pref_polar, binary_polar, kemeny_polar = calculate_polarisation_metrics(
                    agents.positions, preferences, first_choices, party_positions, N_AGENTS, OPINION_SPACE_SIZE, agents.pref_indices
                )
            except TypeError as e:
                st.error(f"calculate_polarisation_metrics çağrısında hata: {str(e)}. polarisation.py dosyasını kontrol edin.")
                return None, None, None

            polarisation_records.append({
                "Iteration": t,
                "PartyPolarisation": party_polar,
                "PrefPolarisation": pref_polar,
                "BinaryPolarisation": binary_polar,
                "KemenyPolarisation": kemeny_polar
            })

            voting_share = pd.Series(first_choices).value_counts(normalize=True).sort_index()
            voting_records.append({"Iteration": t, **voting_share.to_dict()})

            positions_record.append((frame, frame_party_centers, frame_society_center))

    polarisation_df = pd.DataFrame(polarisation_records)
    voting_df = pd.DataFrame(voting_records).fillna(0)

    return positions_record, polarisation_df, voting_df

# --- Streamlit Setup ---
st.set_page_config(page_title="Polarisation Simulation", layout="wide")
st.title("🥉 Agent-Based Polarisation Simulation")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Settings")

if st.sidebar.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.rerun()

# Senaryo seçimi her zaman yapılacak
selected_scenario = st.sidebar.selectbox("Select Scenario for Delta Matrix", list(SCENARIOS.keys()))
delta_matrix = SCENARIOS[selected_scenario]["delta_matrix"]  # 6x6 delta matrix

party_input_mode = st.sidebar.radio(
    "How to define party positions?",
    options=["Preset Scenario", "Manual Entry"],
    index=0
)

if party_input_mode == "Preset Scenario":
    party_positions = np.array(SCENARIOS[selected_scenario]["party_positions"])
else:
    N_PARTIES = st.sidebar.slider("Number of Parties", 2, 5, 3)  # 5 ile sınırlı
    if N_PARTIES > 5:
        st.sidebar.warning("Number of parties limited to 5 to manage preference profile complexity.")
    manual_positions = []
    for i in range(N_PARTIES):
        x = st.sidebar.number_input(f"Party {i+1} - X Coordinate", value=float(i*2-2), step=0.1)
        y = st.sidebar.number_input(f"Party {i+1} - Y Coordinate", value=0.0, step=0.1)
        manual_positions.append([x, y])
    party_positions = np.array(manual_positions)

# Seçilen konfigürasyonu göster
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

col1, col2 = st.sidebar.columns(2)
if col1.button("▶️ Start Simulation"):
    positions_record, polarisation_df, voting_df = run_simulation(N_AGENTS, T, N_FRAMES, party_positions, delta_matrix)
    if positions_record is not None:
        st.session_state.positions_record = positions_record
        st.session_state.polarisation_df = polarisation_df
        st.session_state.voting_df = voting_df
        st.session_state.N_PARTIES = len(party_positions)
        st.session_state.party_positions = party_positions  # Parti pozisyonlarını sakla
if col2.button("⏹️ Stop Simulation"):
    st.session_state.positions_record = None
    st.session_state.party_positions = None  # Temizle

# --- Visualization ---
if 'positions_record' in st.session_state and st.session_state.positions_record is not None:
    positions_record = st.session_state.positions_record
    polarisation_df = st.session_state.polarisation_df
    voting_df = st.session_state.voting_df
    N_PARTIES = st.session_state.N_PARTIES
    # party_positions'ı session_state'ten al, yoksa mevcut party_positions'ı kullan
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

        # Initial data: Ajanlar, parti pozisyonları, parti merkezleri, toplum merkezi
        initial_data = []
        # Ajanlar
        for party in [str(i) for i in range(N_PARTIES)]:
            df_party = positions_df[positions_df["Iteration"] == unique_iterations[0]]
            df_party = df_party[df_party["FirstChoice"] == party]
            initial_data.append(go.Scatter(
                x=df_party["x"], y=df_party["y"],
                mode="markers",
                marker=dict(size=6, color=party_colors[int(party)], symbol="circle"),
                name=f"Party {int(party)+1} Agents"
            ))
        # Parti pozisyonları (sabit)
        initial_data.append(go.Scatter(
            x=party_positions[:, 0], y=party_positions[:, 1],
            mode="markers",
            marker=dict(size=12, color=party_colors[:N_PARTIES], symbol="star", line=dict(width=2, color="black")),
            name="Party Positions"
        ))
        # Parti merkezleri
        df_centers = party_centers_df[party_centers_df["Iteration"] == unique_iterations[0]]
        for i in range(N_PARTIES):
            center = df_centers[df_centers["Party"] == str(i)]
            initial_data.append(go.Scatter(
                x=center["x"], y=center["y"],
                mode="markers",
                marker=dict(size=10, color=party_colors[i], symbol="diamond", line=dict(width=1, color="black")),
                name=f"Party {i+1} Center"
            ))
        # Toplum merkezi
        df_society = society_centers_df[society_centers_df["Iteration"] == unique_iterations[0]]
        initial_data.append(go.Scatter(
            x=df_society["x"], y=df_society["y"],
            mode="markers",
            marker=dict(size=12, color="black", symbol="square", line=dict(width=2, color="white")),
            name="Society Center"
        ))

        # Frames: Her iterasyon için ajanlar, parti pozisyonları, parti merkezleri, toplum merkezi
        for iteration in unique_iterations:
            frame_data = []
            # Ajanlar
            df_iter = positions_df[positions_df['Iteration'] == iteration]
            for party in [str(i) for i in range(N_PARTIES)]:
                df_party = df_iter[df_iter["FirstChoice"] == party]
                frame_data.append(go.Scatter(
                    x=df_party["x"], y=df_party["y"],
                    mode="markers",
                    marker=dict(size=6, color=party_colors[int(party)], symbol="circle"),
                    name=f"Party {int(party)+1} Agents"
                ))
            # Parti pozisyonları (sabit)
            frame_data.append(go.Scatter(
                x=party_positions[:, 0], y=party_positions[:, 1],
                mode="markers",
                marker=dict(size=12, color=party_colors[:N_PARTIES], symbol="star", line=dict(width=2, color="black")),
                name="Party Positions"
            ))
            # Parti merkezleri
            df_centers = party_centers_df[party_centers_df["Iteration"] == iteration]
            for i in range(N_PARTIES):
                center = df_centers[df_centers["Party"] == str(i)]
                frame_data.append(go.Scatter(
                    x=center["x"], y=center["y"],
                    mode="markers",
                    marker=dict(size=10, color=party_colors[i], symbol="diamond", line=dict(width=1, color="black")),
                    name=f"Party {i+1} Center"
                ))
            # Toplum merkezi
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
                # Görüş uzayını çevreleyen çerçeve
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
        # Sütun adlarını Parti 1, Parti 2, ... olarak yeniden adlandır
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

    # --- Download Section ---
    with st.expander("📥 Download Simulation Results"):
        # Polarisation Results
        excel_buffer_polar = io.BytesIO()
        polarisation_df.to_excel(excel_buffer_polar, index=False)
        excel_buffer_polar.seek(0)
        st.download_button(
            label="Download Polarisation Results",
            data=excel_buffer_polar,
            file_name="polarisation_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Voting Results
        excel_buffer_vote = io.BytesIO()
        voting_df.to_excel(excel_buffer_vote, index=False)
        excel_buffer_vote.seek(0)
        st.download_button(
            label="Download Voting Results",
            data=excel_buffer_vote,
            file_name="voting_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Click 'Start Simulation' to run the simulation and view results.")