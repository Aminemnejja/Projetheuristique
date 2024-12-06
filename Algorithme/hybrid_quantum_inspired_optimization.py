import numpy as np
import random


# Modifiez la fonction pour garantir que les résultats ont la bonne forme.
def hybrid_quantum_inspired_optimization(func, D, Tmax, population_size, step, alpha=0.1, pm_init=0.1):
    # Initialisation
    q_population = np.full((population_size, D), 0.5)
    best_solution = None
    best_score = -float('inf')
    memory = []  # Mémoire des meilleures solutions
    results = []  # Résultats sauvegardés à chaque intervalle `step`

    for t in range(Tmax):
        # Étape 1 : Collapse - Transformation des q-bits en solutions binaires
        binary_population = (np.random.rand(population_size, D) < q_population).astype(int)

        # Étape 2 : Évaluation des solutions
        scores = np.array([func(ind) for ind in binary_population])

        # Mise à jour de la mémoire des meilleures solutions
        for i, score in enumerate(scores):
            # Comparer les scores en utilisant np.max(score) pour garantir qu'on utilise un scalaire
            if len(memory) < population_size or np.max(score) > np.max([mem[1] for mem in memory]):
                memory.append((binary_population[i], score))
                if len(memory) > population_size:
                    # Trouver et retirer la solution avec le score le plus bas
                    memory.sort(key=lambda x: np.max(x[1]))  # Trier en fonction des scores
                    memory.pop(0)

        # Mise à jour de la meilleure solution globale
        max_index = np.argmax(scores)
        if scores[max_index] > best_score:
            best_score = scores[max_index]
            best_solution = binary_population[max_index]

        # Étape 3 : Mise à jour des q-bits
        for i in range(population_size):
            for d in range(D):
                if binary_population[i][d] == 1:
                    q_population[i][d] += alpha * (1 - q_population[i][d])
                else:
                    q_population[i][d] -= alpha * q_population[i][d]

        # Étape 4 : Mutation adaptative
        pm = pm_init / (1 + t)  # Probabilité de mutation diminue avec le temps
        for i in range(population_size):
            for d in range(D):
                if np.random.random() < pm:
                    q_population[i][d] = np.clip(q_population[i][d] + np.random.uniform(-0.1, 0.1), 0, 1)

        # Étape 5 : Recherche locale (optionnelle, pour les meilleures solutions)
        for i in range(population_size):
            candidate = binary_population[i].copy()
            indices = random.sample(range(D), 2)  # Inverser deux bits aléatoires
            candidate[indices[0]] = 1 - candidate[indices[0]]
            candidate[indices[1]] = 1 - candidate[indices[1]]
            candidate_score = func(candidate)
            if candidate_score > scores[i]:
                binary_population[i] = candidate
                scores[i] = candidate_score

        # Enregistrer les résultats à intervalles réguliers
        if (t + 1) % step == 0:
            results.append(best_score)

    # Assurez-vous que tous les résultats sont scalaires ou ont une forme compatible
    results = np.array(results)
    return results, best_solution, best_score