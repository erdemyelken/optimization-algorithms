import numpy as np

# Grey Wolf Optimizer (GWO) function
def gwo_search(fun, lb, ub, dim, SearchAgents_no, Max_iter):

    # Initialization
    Alpha_pos = np.zeros(dim)
    Beta_pos = np.zeros(dim)
    Delta_pos = np.zeros(dim)

    Alpha_score = float("inf")
    Beta_score = float("inf")
    Delta_score = float("inf")

    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.uniform(lb[i], ub[i], SearchAgents_no)

    # Main loop
    for l in range(0, Max_iter):

        for i in range(0, SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness = fun(Positions[i, :])

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Delta_score = Beta_score
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()

            if (fitness > Alpha_score) and (fitness < Beta_score):
                Delta_score = Beta_score
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()

            if (fitness > Alpha_score) and (fitness > Beta_score) and (fitness < Delta_score):
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()

        # Update the position of search agents including omegas
        a = 2 - l * ((2) / Max_iter)  # Eq. (3.3) in the paper
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):

                r1 = np.random.random()  # r1 is a random number in [0,1]
                r2 = np.random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a  # Eq. (3.4) in the paper
                C1 = 2 * r2  # Eq. (3.5) in the paper

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])  # Eq. (3.6) in the paper
                X1 = Alpha_pos[j] - A1 * D_alpha  # Eq. (3.7) in the paper

                r1 = np.random.random()
                r2 = np.random.random()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta # Eq. (3.8) in the paper
                
                r1 = np.random.random()
                r2 = np.random.random()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta  # Eq. (3.9) in the paper

                Positions[i, j] = (X1 + X2 + X3) / 3  # Eq. (3.10) in the paper

        # Output
        best_pos = Alpha_pos
        best_score = Alpha_score

    return best_pos, best_score
