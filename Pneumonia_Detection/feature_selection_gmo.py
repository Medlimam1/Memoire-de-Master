import numpy as np
from utils import evaluate_features_with_knn

class GMO:
    def __init__(self, X, y, num_agents=50, max_iter=50, alpha=0.9, beta=0.1):
        self.X = X
        self.y = y
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.dim = X.shape[1]

    def fitness(self, binary_vector):
        if np.sum(binary_vector) == 0:
            return 1
        selected_indices = np.where(binary_vector == 1)[0]
        acc, _, _, _, _ = evaluate_features_with_knn(self.X, self.y, selected_indices)
        selection_ratio = len(selected_indices) / self.dim
        return self.alpha * (1 - acc) + self.beta * selection_ratio

    def optimize(self):
        agents = np.random.randint(0, 2, (self.num_agents, self.dim))
        fitness_values = np.array([self.fitness(agent) for agent in agents])

        best_idx = np.argmin(fitness_values)
        best_agent = agents[best_idx].copy()

        for _ in range(self.max_iter):
            mean_agent = np.mean(agents, axis=0)
            for i in range(self.num_agents):
                r = np.random.rand(self.dim)
                agents[i] = np.where(r < mean_agent, 1, 0)

            fitness_values = np.array([self.fitness(agent) for agent in agents])
            best_idx = np.argmin(fitness_values)
            best_agent = agents[best_idx].copy()

        selected_indices = np.where(best_agent == 1)[0]
        acc, f1, auc, _, _ = evaluate_features_with_knn(self.X, self.y, selected_indices)

        return selected_indices, acc, f1, auc
