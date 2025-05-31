import numpy as np
from utils import evaluate_features_with_knn

# إعداد PSO
class Particle:
    def __init__(self, dim):
        self.position = np.random.randint(0, 2, dim)
        self.velocity = np.random.rand(dim)
        self.best_position = self.position.copy()
        self.best_score = float('inf')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pso_feature_selection(X, y, num_particles=50, max_iter=50, w=0.7, c1=1.5, c2=1.5):
    dim = X.shape[1]
    swarm = [Particle(dim) for _ in range(num_particles)]
    global_best_position = np.random.randint(0, 2, dim)
    global_best_score = float('inf')

    for iter in range(max_iter):
        for particle in swarm:
            binary_position = np.round(sigmoid(particle.position))
            if np.sum(binary_position) == 0:
                continue
            selected_indices = np.where(binary_position == 1)[0]
            acc, _, _, _, _ = evaluate_features_with_knn(X, y, selected_indices)
            score = 1 - acc  # نريد تعظيم الدقة

            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = binary_position.copy()

            if score < global_best_score:
                global_best_score = score
                global_best_position = binary_position.copy()

        for particle in swarm:
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            cognitive = c1 * r1 * (particle.best_position - particle.position)
            social = c2 * r2 * (global_best_position - particle.position)
            particle.velocity = w * particle.velocity + cognitive + social
            particle.position += particle.velocity

    final_selected = np.where(global_best_position == 1)[0]
    acc, f1, auc, _, _ = evaluate_features_with_knn(X, y, final_selected)

    return final_selected, acc, f1, auc
