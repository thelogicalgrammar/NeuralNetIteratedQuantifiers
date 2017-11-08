import pandas as pd
import numpy as np
import random as rnd
import itertools as iter
import matplotlib.pyplot as plt
import scipy.stats

def random_quantifiers(number_agents, list_inputs):
    """
    returns a pd dataframe with meaning rows and a column for each agent
    columns are series of binaries describing which inputs are compatible with agent's quantifier
    """
    random_functions = np.random.randint(2, size=(list_inputs.size, number_agents))
    languages = pd.DataFrame(random_functions, index=list_inputs)
    return languages


def create_languages_dataframe(agents, possible_inputs):
    """
    Given a list of agents, this function creates a DataFrame with agents as columns, inputs as indices and binary
    values for whether the quantifier applies or not to the input in the agent's language
    """
    languages_dataframe = pd.DataFrame(index=possible_inputs)
    for n, agent in enumerate(agents):
        languages_dataframe[n] = agent.produce(possible_inputs)
    return languages_dataframe


def generate_list_inputs(l):
    # l is the length of the bit strings
    # returns a list of all lists of bits of l length
    indices_generator = iter.chain.from_iterable(iter.combinations(range(l), r) for r in range(len(range(l))+1))
    return pd.Series([[1 if n in indices_list else 0 for n in range(l)] for indices_list in indices_generator])


def generate_random_input(list_inputs):
    return rnd.choice(list_inputs)


def equal_languages(actual, wanted, tolerance=0.05):
    # every agent disagrees w/ actual lang < 5% of time
    # TODO: documentation
    assert actual.shape == wanted.shape
    difference = actual.subtract(wanted)
    means = difference.mean()
    means = means.apply(lambda x: abs(x) < tolerance)
    return means.all()


def check_quantity(agent):
    # goes through all the meanings and checks whether (or how much) the learned quantifier satisfies quantity
    pass


def language_distributions(population):
    langs = population.languages.as_matrix()
    # ignore first, always 0.5 for some reason
    for poss_input in range(1, len(langs)):
        kde = scipy.stats.gaussian_kde(langs[poss_input, :])
        x = np.linspace(0, 1, 200)
        plt.plot(x, kde(x))
    plt.show()
