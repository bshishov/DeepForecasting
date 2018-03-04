import numpy as np
from collections import deque


class GeneticAlgorithm(object):
    def __init__(self, space: dict, fitness_fn: callable, hall_of_fame: int = 10):
        self.params_choices = list(space.values())
        self.params_names = list(space.keys())
        self.fitness_fn = fitness_fn
        self.cache = {}
        self.hall_of_fame = deque(maxlen=hall_of_fame)
        self.fitness_hits = 0

    def crossover(self, genome_a: tuple, genome_b: tuple) -> tuple:
        assert len(genome_a) == len(genome_b) and len(genome_a) > 1
        crossover_point = np.random.choice(range(1, len(genome_a)))
        return genome_a[:crossover_point] + genome_b[crossover_point:]

    def mutate(self, genome) -> tuple:
        point = np.random.choice(len(genome))
        choices = list(range(len(self.params_choices[point])))
        choices.remove(genome[point])
        rnd_val = np.random.choice(choices)
        return genome[:point] + (rnd_val,) + genome[point + 1:]

    def sample(self) -> tuple:
        return tuple(np.random.choice(len(choices)) for choices in self.params_choices)

    def get_kwargs(self, genome: tuple) -> dict:
        return {name: self.params_choices[i][genome[i]] for i, name in enumerate(self.params_names)}

    def fitness(self, genome: tuple, verbose: bool = True) -> float:
        if genome in self.cache:
            return self.cache[genome]

        self.fitness_hits += 1
        fitness = self.fitness_fn(**self.get_kwargs(genome))
        self.cache[genome] = fitness

        if len(self.hall_of_fame) > 0:
            _, best_f = self.hall_of_fame[-1]
            # Keep track of best individual
            if fitness > best_f:
                best_kwargs = self.get_kwargs(genome)
                if verbose:
                    print('New best:\n\tArgs: {0}\n\tF: {1:.3f}'.format(best_kwargs, fitness))
                self.hall_of_fame.append((best_kwargs, fitness))
        else:
            self.hall_of_fame.append((self.get_kwargs(genome), fitness))

        return fitness

    def run(self, n_iters=100, population_size=100, mutation_chance=0.5, verbose: bool = True):
        population = [self.sample() for _ in range(population_size)]
        fitnesses = [self.fitness(genome, verbose=verbose) for genome in population]

        for i in range(n_iters):
            f_exp = np.exp(fitnesses)
            p = f_exp / np.sum(f_exp)

            new_population = []
            new_fitnesses = []
            for i in range(population_size):
                # Pair selection
                pair = np.random.choice(range(len(population)), 2, p=p, replace=False)
                parent_a = population[pair[0]]
                parent_b = population[pair[1]]

                # Crossover
                new_genome = self.crossover(parent_a, parent_b)

                # Mutation
                if np.random.rand() < mutation_chance:
                    new_genome = self.mutate(new_genome)

                new_population.append(new_genome)
                new_fitnesses.append(self.fitness(new_genome, verbose=verbose))

            population = new_population
            fitnesses = new_fitnesses

        if len(self.hall_of_fame) > 0:
            return self.hall_of_fame[-1]

        return None
