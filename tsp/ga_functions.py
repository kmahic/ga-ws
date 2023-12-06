import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(42)

# ----------------- #
# - RECOMBINATION - #
# ----------------- #
def cycle_crossover(parent1: list, parent2: list) -> list:
    "Crossover by Cycle Crossover"

    # initialize children, where child1 = parent2 and vica verca
    child1 = parent2.copy()
    child2 = parent1.copy()

    # sample random starting index
    startIndex = random.randint(0, len(parent1) - 1)
    index = startIndex

    while True:
        # set the child value at index to equal the value of the parent at index
        child1[index] = parent1[index]
        child2[index] = parent2[index]

        # calculate next index by finding index in first vector that has value at index in second vector
        nextIndex = parent1.index(parent2[index])
        index = nextIndex

        # if nextIndex is where we started, cycle is finished
        if index == startIndex:
            break

    return [child1, child2]


def ordered_crossover(parent1: list, parent2: list) -> list:
    """
    Randomly select a subset of the first parent string and then fill the remainder of the route
    with the genes from the second parent in the order in which they appear,
    without duplicating any genes in the selected subset from the first parent
    """

    # randomly select subset of first parent by sampling start and end indexes
    startidx = random.randint(0, len(parent1) - 1)
    endidx = startidx
    while endidx == startidx:
        endidx = random.randint(0, len(parent1) - 1)
    startidx, endidx = sorted([startidx, endidx])

    # extract subsets
    parent1_subset = parent1[startidx : endidx + 1]
    parent2_subset = parent2[startidx : endidx + 1]

    # calculate remaining items in lists
    remaining_parent1 = [item for item in parent1 if item not in parent2_subset]
    remaining_parent2 = [item for item in parent2 if item not in parent1_subset]

    # first child should have subset from parent1, remainder from parent2
    child1 = (
        remaining_parent2[:startidx] + parent1_subset + remaining_parent2[startidx:]
    )

    # second child should have subset from parent2, remainder from parent1
    child2 = (
        remaining_parent1[:startidx] + parent2_subset + remaining_parent1[startidx:]
    )

    return [child1, child2]


def n_point_crossover(n: int, parent1: list, parent2: list) -> list:
    "Performs N-Point Crossover on two parents."
    # TODO
    return []


def uniform_crossover(parent1: list, parent2: list) -> list:
    "Each gene (bit) is selected randomly from one of the corresponding genes of the parent chromosomes."
    # TODO
    return []


# ----------------- #
# --- SELECTION --- #
# ----------------- #
def tournament_selection(
    population: list, select: int, tournament_size: int, tournament_winners: int
) -> list:
    "Deterministically select from population using the tournament technique"

    def tournament(population):
        # sample k contestors to compete in the tournament
        contestors = random.choices(population, k=tournament_size)

        # sort the contesting solutions according to fitness
        contestors = sorted(contestors, key=lambda x: x.fitness)

        # return the best solutions
        return contestors[:tournament_winners]

    winners = []
    while len(winners) + tournament_winners <= select:
        winners += tournament(population)

    assert len(winners) == select
    return winners


def rank_based_selection(population: list, select: int) -> list:
    "Probabilistically select individuals based on relative rank"
    sorted_pop = sorted(population, key=lambda x: x.fitness)
    n = len(population)

    def get_rank_relative_probability(n, rank):
        nneg = 0.5
        nnpos = 2 - nneg
        return (1 / n) * (nneg + (nnpos - nneg) * ((rank - 1) / (n - 1)))

    rank_probabilities = [
        get_rank_relative_probability(n, rank + 1) for rank in range(n)
    ]

    def rank_solution(solution):
        rank = sorted_pop.index(solution)
        rank_prob = rank_probabilities[rank]
        return rank, rank_prob

    selected = []
    while len(selected) < select:
        # sample random probability
        p = random.random()

        # select random solution from population
        sample = random.choice(population)
        _, rank_prob = rank_solution(sample)

        # select if sampled probability is lower or eq to solutions rank prob
        if p <= rank_prob:
            selected.append(sample)

    return selected


def fitness_proportionate_selection(population: list, select: int) -> list:
    """
    Probabilistically selects individuals from the population based on relative fitness
    Probability function is P(x) = 1 - x/max
    """
    # TODO
    return []


# ------------------ #
# ---- MUTATION ---- #
# ------------------ #
def swap_mutation(indexes: list, mutation_probability: float) -> list:
    """
    Mutation by probabilistically swapping two indexes in a vector with each other.
    Goes through each entry in vector and decides if it should swap or not.
    """
    mutated_indexes = indexes.copy()
    N = len(indexes)

    for from_index in range(N):
        should_mutate = random.random() < mutation_probability
        if should_mutate:
            to_index = random.randint(0, N - 1)
            switch_to = mutated_indexes[to_index]
            switch_from = mutated_indexes[from_index]

            mutated_indexes[from_index] = switch_to
            mutated_indexes[to_index] = switch_from

    return mutated_indexes


def inversion_mutation(tour: list) -> list:
    """
    Mutation by sampling two indexes and reversing the order of the elements in between them.
    """
    # TODO
    return None


# ------------------ #
# ---- SURVIVAL ---- #
# ------------------ #
def elitist_survival(
    population: list, children: list, keep_elite_fraction: float
) -> list:
    "Replacement by elitism. Keeps n fittest individuals from original population."
    keep_elite = sorted(population, key=lambda x: x.fitness)[
        : int(keep_elite_fraction * len(population))
    ]
    best_children = sorted(children, key=lambda x: x.fitness)
    survivors = keep_elite + best_children[: (len(children) - len(keep_elite))]

    return survivors


def generational_survival(children: list) -> list:
    "Let new generation completely replace the old one"
    return children
