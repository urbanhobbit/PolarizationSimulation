import numpy as np

def deliberation_step_matched(positions, profiles, delta_matrix, t,
                             opinion_space_size=10, mu_a=0.05, mu_r=1.4, discount_coeff=0.99, interaction_rate=1.0):
    N = len(positions)
    new_positions = positions.copy()
    current_discount = discount_coeff ** (t - 1)

    for listener in range(N):
        if np.random.rand() > interaction_rate:
            continue

        speaker = np.random.randint(N)
        if listener == speaker:
            continue

        delta = delta_matrix[profiles[listener], profiles[speaker]]
        p_listen = current_discount * delta

        if np.random.rand() < p_listen:
            attraction = mu_a * (positions[speaker] - positions[listener])
            # Rastgele tepki vektörü: Birim çember üzerinde rastgele açı
            theta = 2 * np.pi * np.random.rand()  # Rastgele açı (0 to 2π)
            noise = mu_r * np.array([np.cos(theta), np.sin(theta)])  # Birim çember vektörü
            move = attraction + noise
            new_positions[listener] += move

            new_positions[listener] = np.clip(new_positions[listener], -opinion_space_size/2, opinion_space_size/2)

    return new_positions