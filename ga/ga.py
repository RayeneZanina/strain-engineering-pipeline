import numpy as np
import cobra
import tqdm

class GeneticAlgorithm:
    def __init__(self,model, target_id, biomass_id, population_size=100, mutation_rate=0.01, crossover_rate=0.5, elitism_rate=0.1, max_generations=200):
        self.base_model = model.copy()
        self.model = self.base_model.copy()
        self.target_rxn = self.model.reactions.get_by_id(target_id)
        self.biomass_rxn = self.model.reactions.get_by_id(biomass_id)
        self.num_genes = len(self.model.genes)
        baseline_solution = self.model.optimize()
        self.biomass_baseline = baseline_solution.fluxes[self.biomass_rxn.id]
        self.target_baseline = baseline_solution.fluxes[self.target_rxn.id]
        self.base_fluxes = baseline_solution.fluxes.values

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.max_generations = max_generations
        self.population = np.random.randint(
            0,
            2,
            size=(self.population_size, self.num_genes)
        )

        self.cache = {}

    def evaluate_fitness(self, individual):
        if tuple(individual) in self.cache:
            return self.cache[tuple(individual)]
        model = self.base_model.copy()
        for i, val in enumerate(individual):
            if val == 1:
                model.genes[i].knock_out()
        solution = model.optimize()
        if solution.status != 'optimal':
                return -100
            
        biomass_flux, target_flux = solution.fluxes[self.biomass_rxn.id], solution.fluxes[self.target_rxn.id]
        reward =  5 * (target_flux - self.target_baseline) + (biomass_flux - self.biomass_baseline) - 0.1 * np.sum(individual)
        if biomass_flux < 0.05:
                reward -= 5
        self.cache[tuple(individual)] = reward
        return reward
            
    def evaluate_population(self):
        fitness_scores = np.array([self.evaluate_fitness(individual) for individual in self.population])
        return fitness_scores
    
    def select_elites(self, fitness_scores):
        num_elites = int(self.elitism_rate * self.population_size)
        elite_indices = np.argsort(fitness_scores)[-num_elites:]
        elites = self.population[elite_indices]
        return elites
    
    def crossover(self, parent1, parent2):
        crossover_mask = np.random.rand(self.num_genes) < self.crossover_rate
        child = np.where(crossover_mask, parent1, parent2)
        return child
    
    def mutate(self, individual):
        mutation_mask = np.random.rand(self.num_genes) < self.mutation_rate
        individual[mutation_mask] = 1 - individual[mutation_mask] # flip
        return individual
    
    def next_gen(self, elites):
        new_pop = [elite.copy() for elite in elites]
        while len(new_pop) < self.population_size:
            parent1 = elites[np.random.randint(len(elites))]
            parent2 = elites[np.random.randint(len(elites))]
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_pop.append(child)
        self.population = np.array(new_pop)

    def run(self):
        best_individual = None
        best_fitness = -np.inf

        for generation in tqdm.tqdm(range(self.max_generations)):
            fitness_scores = self.evaluate_population()
            elites = self.select_elites(fitness_scores)

            if np.max(fitness_scores) > best_fitness:
                best_fitness = np.max(fitness_scores)
                best_individual = self.population[np.argmax(fitness_scores)].copy()
            
            self.next_gen(elites)
        
        best_knockouts = [self.model.genes[i].id for i in range(self.num_genes) if best_individual[i] == 1]

        return best_knockouts, best_fitness
