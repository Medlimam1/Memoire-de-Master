import numpy as np
from utils import evaluate_features_with_knn
from geneticalgorithm import geneticalgorithm as ga

# إعداد دالة الهدف الخاصة بـ GA
class GAWrapper:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def fitness_function(self, binary_vector):
        binary_vector = np.round(binary_vector).astype(int)
        if np.sum(binary_vector) == 0:
            return 1  # أسوأ حالة إذا لم يتم اختيار أي سمة

        selected_indices = np.where(binary_vector == 1)[0]
        acc, _, _, _, _ = evaluate_features_with_knn(self.X, self.y, selected_indices)
        return 1 - acc  # لأن GA تبحث عن الحد الأدنى


def ga_feature_selection(X, y):
    dim = X.shape[1]
    ga_obj = GAWrapper(X, y)

    varbound = np.array([[0, 1]] * dim)
    algorithm_param = {'max_num_iteration': 50, 'population_size': 50, 'mutation_probability': 0.1,
                       'elit_ratio': 0.01, 'crossover_probability': 0.5, 'parents_portion': 0.3,
                       'crossover_type': 'uniform', 'max_iteration_without_improv': None}

    model = ga(function=ga_obj.fitness_function,
               dimension=dim,
               variable_type='bool',
               variable_boundaries=varbound,
               algorithm_parameters=algorithm_param)

    model.run()

    best_vector = np.round(model.output_dict['variable']).astype(int)
    selected_indices = np.where(best_vector == 1)[0]

    acc, f1, auc, _, _ = evaluate_features_with_knn(X, y, selected_indices)

    return selected_indices, acc, f1, auc
