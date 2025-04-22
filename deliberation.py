def deliberation_step_matched(positions, profiles, delta_matrix, t, party_positions, opinion_space_size, mu_a, mu_r, discount_coeff, interaction_rate):
    current_discount = discount_coeff ** t
    N = len(positions)
    delta_matrix_shape = delta_matrix.shape

    for _ in range(int(interaction_rate * N)):
        listener = np.random.randint(0, N)
        speaker = np.random.randint(0, N)
        if listener == speaker:
            continue

        listener_idx = profiles[listener]
        speaker_idx = profiles[speaker]

        # Safety check to prevent IndexError
        if not (0 <= listener_idx < delta_matrix_shape[0] and 0 <= speaker_idx < delta_matrix_shape[1]):
            raise ValueError(f"Profile indices out of bounds: listener_idx={listener_idx}, speaker_idx={speaker_idx}, delta_matrix_shape={delta_matrix_shape}")

        delta = delta_matrix[listener_idx, speaker_idx]
        p_listen = current_discount * delta

        if np.random.rand() < p_listen:
            # Update positions (implementation not shown, assuming it exists)
            pass  # Replace with your existing logic

    return positions, profiles
