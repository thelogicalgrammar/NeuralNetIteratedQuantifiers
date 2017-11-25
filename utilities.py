import pandas as pd
import numpy as np
import random as rnd
import itertools as iter


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
    return pd.Series([tuple([1 if n in indices_list else 0 for n in range(l)]) for indices_list in indices_generator])


def generate_random_input(list_inputs):
    return rnd.choice(list_inputs)


def check_quantity(languages):
    """
    This function is a bloody mess. I should be put in prison.

    Input: languages as a DataFrame
    Return: value quantifying quantity

    For each number of 1s, calculates the sum for all # of 1s:
        # of inputs with that number of 1s - 1 *
        proportion of the inputs with that # of 1s that are true

    I changed the indices of the language frames to tuples to allow for indexing
    """
    rounded = np.round(languages)
    quantity = pd.Series(0, index=rounded.columns)
    for n_ones, inputs_with_n_ones in rounded.groupby(np.count_nonzero, axis=0):
        proportions = inputs_with_n_ones.apply(lambda x: np.sum(x)/x.size, axis=0)
        to_append = proportions * n_ones
        quantity += to_append
    return quantity
