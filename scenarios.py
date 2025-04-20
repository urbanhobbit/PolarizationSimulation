import numpy as np

SCENARIOS = {
    "Scenario 1 - Ideal World": {
        "party_positions": [
            [0, 4],
            [-3.46, -2],
            [3.46, -2],
        ],
        "delta_matrix": np.ones((6, 6))
    },

    "Scenario 2 - Echo Chambers": {
        "party_positions": [
      	    [0, 4],
            [-3.46, -2],
            [3.46, -2],
         ],
        "delta_matrix": np.array([
            [1, 1, 0.1, 0.1, 0.1, 0.1],
            [1, 1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 1, 1, 0.1, 0.1],
            [0.1, 0.1, 1, 1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 1, 1],
            [0.1, 0.1, 0.1, 0.1, 1, 1]
        ])
    },

    "Scenario 3 - One Group Isolated": {
        "party_positions": [
       	    [0, 4],
            [-3.46, -2],
            [3.46, -2],
        ],
        "delta_matrix": np.array([
            [1, 1, 1, 1, 0.1, 0.1],
            [1, 1, 1, 1, 0.1, 0.1],
            [1, 1, 1, 1, 0.1, 0.1],
            [1, 1, 1, 1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 1, 1],
            [0.1, 0.1, 0.1, 0.1, 1, 1]
        ])
    },

    "Scenario 4 - Polarized Echo Chambers": {
        "party_positions": [
            [0, 4.6188],
            [-4, -2.3094],
            [4, -2.3094]
        ],
        "delta_matrix": np.array([
            [1, 1, 0.1, 0.1, 0.1, 0.1],
            [1, 1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 1, 1, 0.1, 0.1],
            [0.1, 0.1, 1, 1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 1, 1],
            [0.1, 0.1, 0.1, 0.1, 1, 1]
        ])
    },

    "Scenario 5 - Radicals Not Isolated": {
        "party_positions": [
            [0, 3.45],
            [-2, -1.15],
            [2, -1.15]
        ],
        "delta_matrix": np.ones((6, 6))
    },

    "Scenario 6 - Radicals Isolated": {
        "party_positions": [
            [0, 3.45],
            [-2, -1.15],
            [2, -1.15]
        ],
        "delta_matrix": np.array([
            [1, 1, 0.1, 0.1, 0.1, 0.1],
            [1, 1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 1, 1, 1, 1],
            [0.1, 0.1, 1, 1, 1, 1],
            [0.1, 0.1, 1, 1, 1, 1],
            [0.1, 0.1, 1, 1, 1, 1]
        ])
    },

    "Scenario 7 - 12 Angry Men": {
        "party_positions": [
            [0, 3.45],
            [-2, -1.15],
            [2, -1.15]
        ],
        "delta_matrix": np.array([
            [1, 1, 0.1, 0.1, 0.1, 0.1],
            [1, 1, 0.1, 0.1, 0.1, 0.1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]
        ])
    },

    "Scenario 8 - Polarized No Echo Chambers": {
        "party_positions": [
            [0, 4.6188],
            [-4, -2.3094],
            [4, -2.3094]
        ],
        "delta_matrix": np.ones((6, 6))
    }
}

