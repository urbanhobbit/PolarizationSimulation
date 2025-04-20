# config.py

# Simulation Parameters
N_AGENTS = 1000              # Number of agents in the simulation
T = 1000                     # Total number of interactions (iterations)
N_PARTIES = 3                # Number of political parties
OPINION_SPACE_SIZE = 10.0    # Size of the opinion space (range for agents' positions)
INTERACTION_RATE = 1.0      # Probability of interaction between agents
MU_ATTRACTION = 0.05        # Attraction coefficient
MU_REACTION = 1.4         # Reaction noise coefficient (for random movement)
DISCOUNT_COEFF = 0.99       # Discount coefficient for interactions over time
