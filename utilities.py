import numpy as np
import random as rnd
import itertools as iter


def generate_list_inputs(l):
    # l is the length of the bit strings
    # returns a list of all lists of bits of l length
    # indices generator is a list of lists of indices
    indices_generator = iter.chain.from_iterable(iter.combinations(range(l), r) for r in range(len(range(l))+1))
    return np.array([[1 if n in indices_list else 0 for n in range(l)] for indices_list in indices_generator])


def random_quantifiers(number_agents, inputs):
    """
    returns an array, each column is a boolean vector describing which inputs are compatible with agent's quantifier
    the rows correspond to the elements of list_inputs
    """
    return np.random.randint(2, size=(inputs.shape[0], number_agents))


def create_languages_array(agents, possible_inputs, map=False):
    """
    agents as columns, inputs rows
    whether the quantifier applies or not to the input in the agent's language
    """
    languages = np.empty(shape=(possible_inputs.shape[0], len(agents)))
    if map:
        for n, agent in enumerate(agents):
            languages[:, n] = agent.map(possible_inputs).T
    else:
        for n, agent in enumerate(agents):
            # assumes that agent.produce is vectorized
            languages[:, n] = agent.produce(possible_inputs).T
    return languages


def generate_random_input(list_inputs):
    # note that numpy's own np.random.choice assumes 1d array
    # while rnd.random correctly picks a single row
    return rnd.choice(list_inputs)


def check_quantity(list_inputs, map_lang):
    # TODO: consider vectorizing across first axis (i.e. generation) of the 3-d results array
    # TODO: this function is written pretty badly

    """
    Calculates quantity as 1 - H(quantifier is true at the model | model size)
    """
    # prob_num is the array with the unconditional probability of each # of 1s in a random model
    count_ones = np.count_nonzero(list_inputs, axis=1)
    num_arrays_of_length = np.unique(count_ones, return_counts=True)[1]
    prob_num = num_arrays_of_length / list_inputs.shape[0]
    # 2d array with shape (quantifier true values, model size) that is true if the quantifier is
    # true at that model, at the column corresponding to that model size
    temp = np.zeros(shape=(map_lang.shape[0], list_inputs.shape[1]+1))
    # there must be a better way of doing this but I can't think of it atm
    for i in np.arange(0, len(map_lang)):
        temp[i, count_ones[i]] = map_lang[i]

    num_true_by_size = np.sum(temp, axis=0)
    prob_true_by_size = num_true_by_size / num_arrays_of_length
    prob_false_by_size = 1 - prob_true_by_size
    log1 = np.log2(prob_true_by_size)
    log1[log1 == -np.inf] = 0
    entropy1 = prob_true_by_size * log1
    log2 = np.log2(prob_false_by_size)
    log2[log2 == -np.inf] = 0
    entropy2 = prob_false_by_size * log2
    cond_entropy = -np.sum(prob_num * (entropy1 + entropy2))

    # since the maximum entropy of a bernoulli variable is 1 bit, cond_entropy <= 1
    # make it into a distance rather than a similarity.
    # If quantity is 1, it means that the quantifier is completely monotonic
    quantity = 1 - cond_entropy
    return quantity


if __name__ == "__main__":
    import population as pop
    max_model_size = 10
    agent = pop.Agent(max_model_size)
    inputs = generate_list_inputs(max_model_size)
    languages = create_languages_array([agent], inputs, map=True)
    print(check_quantity(inputs, languages))
