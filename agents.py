import numpy as np
from math import factorial

class Agents:
    def __init__(self, n, size):
        self.n_agents = n
        self.positions = np.random.uniform(-size/2, size/2, (n, 2))
        self.preferences = None
        self.pref_indices = None  # Tercih indekslerini saklamak için

    def update_preferences(self, party_positions):
        distances = np.linalg.norm(self.positions[:, None, :] - party_positions[None, :, :], axis=2)
        # Tam tercih sıralaması
        self.preferences = np.argsort(distances, axis=1)
        # Tercih indekslerini MATLAB tarzında hesapla
        n_parties = len(party_positions)
        self.pref_indices = np.ones(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            for a in range(n_parties - 1):
                self.pref_indices[i] += np.sum(self.preferences[i, a:n_parties] < self.preferences[i, a]) * factorial(n_parties - a - 1)
        return self.preferences