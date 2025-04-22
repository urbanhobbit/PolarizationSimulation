import numpy as np
import math

def deliberation_step_matched(positions, profiles, delta_matrix, t, party_positions,
                             opinion_space_size=10, mu_a=0.05, mu_r=1.4, discount_coeff=0.99, interaction_rate=1.0):
    N = len(positions)
    new_positions = positions.copy()
    new_profiles = profiles.copy()
    current_discount = discount_coeff ** (t - 1)

    for listener in range(N):
        speaker = np.random.randint(N)
        if listener == speaker:
            continue

        delta = delta_matrix[profiles[listener], profiles[speaker]]
        p_listen = current_discount * delta

        if np.random.rand() < p_listen:
            attraction = mu_a * (positions[speaker] - positions[listener])
            theta = 2 * np.pi * np.random.rand()
            noise = mu_r * np.array([np.cos(theta), np.sin(theta)])
            new_positions[listener] += attraction + noise
            new_positions[listener] = np.clip(new_positions[listener], -opinion_space_size/2, opinion_space_size/2)

            dist_to_parties = np.sqrt(np.sum((new_positions[listener] - party_positions) ** 2, axis=1))
            pref_ranking = np.argsort(dist_to_parties)
            pref_index = 0  # 0-based indexing
            nparty = len(party_positions)
            for a in range(nparty - 1):
                pref_index += np.sum(pref_ranking[a:nparty] < pref_ranking[a]) * math.factorial(nparty - a - 1)
            new_profiles[listener] = pref_index

    return new_positions, new_profiles