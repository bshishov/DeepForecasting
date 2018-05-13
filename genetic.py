import numpy as np
import enum
from collections import deque


class SelectionType(enum.Enum):
    PROPORTIONAL = 0
    EXPONENTIAL = 1


class GeneticAlgorithm(object):
    def __init__(self, space: dict, fitness_fn: callable, hall_of_fame: int = 10):
        self.params_names = sorted(space.keys())
        self.params_choices = [space[key] for key in self.params_names]

        print('Parameters: {0}\n{1}'.format(len(self.params_names), self.params_names))

        combinations = 1
        for choices in self.params_choices:
            combinations *= len(choices)

        print('Total combinations: {0}'.format(combinations))

        self.fitness_fn = fitness_fn
        self.cache = {}
        self.hall_of_fame = deque(maxlen=hall_of_fame)
        self.fitness_hits = 0
        self.cache_hits = 0

    def crossover(self, genome_a: tuple, genome_b: tuple) -> tuple:
        """
        Generates new genome by performing simple crossover over two genomes
        Crossover point is a uniformly selected index of gene in range [1, len(genome) - 1]
        to make sure that at leas 1 gene from parent's genome will be selected

        :param genome_a: First parent's genome
        :param genome_b: Second parent's genome
        :return: New (child) genome
        """
        assert len(genome_a) == len(genome_b) and len(genome_a) > 1
        crossover_point = np.random.choice(range(1, len(genome_a)))
        return genome_a[:crossover_point] + genome_b[crossover_point:]

    def mutate(self, genome) -> tuple:
        """
        Mutates on randomly selected gene of a genome with a new value from alphabet
        Old gene value is excluded from choices

        :param genome: Source genome
        :return: New (mutated) genome
        """
        point = np.random.choice(len(genome))
        choices = list(range(len(self.params_choices[point])))
        choices.remove(genome[point])
        rnd_val = np.random.choice(choices)
        return genome[:point] + (rnd_val,) + genome[point + 1:]

    def sample(self) -> tuple:
        """
        Generates an uniformly samples genome

        :return: New (random) genome
        """
        return tuple(np.random.choice(len(choices)) for choices in self.params_choices)

    def get_kwargs(self, genome: tuple) -> dict:
        return {name: self.params_choices[i][genome[i]] for i, name in enumerate(self.params_names)}

    def fitness(self, genome: tuple, verbose: bool = True) -> float:
        if genome in self.cache:
            self.cache_hits += 1
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
            genome_kwargs = self.get_kwargs(genome)
            self.hall_of_fame.append((genome_kwargs, fitness))
            if verbose:
                print('New best:\n\tArgs: {0}\n\tF: {1:.3f}'.format(genome_kwargs, fitness))

        return fitness

    def run(self,
            n_iters=100,
            population_size=100,
            mutation_chance=0.5,
            verbose: bool = True,
            selection: SelectionType = SelectionType.EXPONENTIAL):
        population = [self.sample() for _ in range(population_size)]
        fitnesses = np.array([self.fitness(genome, verbose=verbose) for genome in population], dtype=np.float32)

        for iteration in range(n_iters):
            if selection == SelectionType.EXPONENTIAL:
                # Exponential selection
                f_exp = np.exp(fitnesses)
                p = f_exp / np.sum(f_exp)
            elif selection == SelectionType.PROPORTIONAL:
                # Proportional selection
                # Rescale so fitnesses will be in range [0, 1],
                # And add small bias to be able to select even the worst genome
                f_rescaled = (fitnesses - np.min(fitnesses)) / (np.max(fitnesses) - np.min(fitnesses)) + 0.1

                # Probabilities of being selected (proportional)
                p = f_rescaled / np.sum(f_rescaled)

            new_population = []
            new_fitnesses = []
            for _ in range(population_size):
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


def _test_ga():
    def _test_fitness(x, y):
        return -(1 + np.sin(x) * np.cos(y))

    ga = GeneticAlgorithm({
        'x': np.linspace(-3.5, 3.5, 1000),
        'y': np.linspace(-3.5, 3.5, 1000),
    }, _test_fitness)
    ga.run(n_iters=1000, population_size=10, verbose=True, selection=SelectionType.PROPORTIONAL)
    print('Fitness hits: {0}'.format(ga.fitness_hits))
    print('Cache hits: {0}'.format(ga.cache_hits))


if __name__ == '__main__':
    _test_ga()