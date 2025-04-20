import numpy as np
from scipy.special import comb
from math import factorial, floor

def calculate_polarisation_metrics(agent_positions, preferences, first_choices, party_positions, N_AGENTS, OPINION_SPACE_SIZE, pref_indices):
    N_PARTIES = len(party_positions)

    # Majority Matrix
    majority_matrix = np.zeros((N_PARTIES, N_PARTIES))
    for i in range(N_AGENTS):
        ranks = preferences[i]
        for a in range(N_PARTIES):
            for b in range(a+1, N_PARTIES):
                if ranks.tolist().index(a) < ranks.tolist().index(b):
                    majority_matrix[a, b] += 1
                else:
                    majority_matrix[b, a] += 1

    # Party Polarisation
    support_counts = np.bincount(first_choices, minlength=N_PARTIES)
    party_centers = []
    for i in range(N_PARTIES):
        supporters = agent_positions[first_choices == i]
        if len(supporters) > 0:
            center = supporters.mean(axis=0)
        else:
            center = np.array([np.nan, np.nan])
        party_centers.append(center)

    partypolar = 0
    for i in range(N_PARTIES-1):
        for j in range(i+1, N_PARTIES):
            if support_counts[i] > 0 and support_counts[j] > 0:
                if not np.any(np.isnan(party_centers[i])) and not np.any(np.isnan(party_centers[j])):
                    dist = np.linalg.norm(party_centers[i] - party_centers[j])
                    partypolar += dist * support_counts[i] * support_counts[j] / (N_AGENTS ** 2)
    partypolar /= comb(N_PARTIES, 2)

    # Preference Polarisation
    n_prefs = factorial(N_PARTIES)
    nprefsup = np.zeros(n_prefs)
    pref_centers = [np.array([np.nan, np.nan])] * n_prefs
    for r in range(n_prefs):
        cluster = agent_positions[pref_indices == r + 1]  # MATLAB indeksleri 1'den başlar
        if len(cluster) > 0:
            pref_centers[r] = cluster.mean(axis=0)
            nprefsup[r] = len(cluster)

    prefpolar = 0
    for r in range(n_prefs-1):
        for q in range(r+1, n_prefs):
            if nprefsup[r] > 0 and nprefsup[q] > 0:
                if not np.any(np.isnan(pref_centers[r])) and not np.any(np.isnan(pref_centers[q])):
                    dist = np.linalg.norm(pref_centers[r] - pref_centers[q])
                    prefpolar += dist * nprefsup[r] * nprefsup[q] / (N_AGENTS ** 2)
    prefpolar /= comb(n_prefs, 2)

    # Binary Polarisation
    binarypolar = 0
    for i in range(N_PARTIES-1):
        for j in range(i+1, N_PARTIES):
            if majority_matrix[i, j] < floor(N_AGENTS/2):
                binarypolar += majority_matrix[i, j]
            else:
                binarypolar += N_AGENTS - majority_matrix[i, j]
    binarypolar /= comb(N_PARTIES, 2) * floor(N_AGENTS/2)

    # Kemeny Polarisation
    kemenypolar = 0
    for i in range(N_PARTIES-1):
        for j in range(i+1, N_PARTIES):
            kemenypolar += majority_matrix[i, j] * (N_AGENTS - majority_matrix[i, j])
    kemenypolar /= comb(N_PARTIES, 2) * floor(N_AGENTS/2) * (N_AGENTS - floor(N_AGENTS/2))

    return partypolar, prefpolar, binarypolar, kemenypolar