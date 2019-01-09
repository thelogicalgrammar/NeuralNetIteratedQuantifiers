from functools import lru_cache
import numpy as np
from utilities import generate_list_models, create_languages_array
import population as pop
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn
import random as rnd


def L1_dist(array1, array2):
    """
    Finds the proportion of models (rows) about which two sets of agents (columns) disagree.
    :param array1: Array of shape (models, agents)
    :param array2: Array of shape (models, agents)
    :return: Total proportion of disagreement across all agents
    """
    assert(array1.shape == array2.shape)
    return np.sum(np.absolute(array1 - array2)) / array1.size


def check_agents_similarity(agent1, agent2, all_models, mapping=False):
    """
    returns the proportion of the models about which the agents disagree if map==True
    or the average difference between their confidence level about each model if map==False
    """
    judgments = create_languages_array([agent1, agent2], all_models,
                                       mapping)
    # proportion of models where the judgments of the two agents are different
    prop_different = L1_dist(judgments[:, 0], judgments[:, 1])
    return prop_different


def check_agent_quantifier_similarity(agent, quantifier, all_models):
    """
    Returns the proportion of models where the agent differs from the quantifier
    """
    judgments = agent.map(all_models)
    prop_different = L1_dist(quantifier, judgments)
    return prop_different


def agent_quantifier_test(max_model_size=None, quant=None, train_split=0.75):
    """
    Plots the difference between a random agent and a quantifier (random if not passed) as the agent observed
    data produced from the quantifier
    """
    if not max_model_size:
        max_model_size = 10
    all_models = generate_list_models(max_model_size)
    quantifier = quant or np.random.randint(0, 2, size=(len(all_models), 1))
    all_models, quantifier = shuffle_learning_model(all_models, quantifier)
    train_models, test_models = train_test_split(all_models, train_split)
    train_quant, test_quant = train_test_split(quantifier, train_split)

    agent = pop.NetworkAgent(max_model_size)
    train_distances, test_distances = [], []
    for i in range(1000):
        random_indices = np.random.randint(0, len(train_models),
                                           int(0.7*len(train_models)))
        models = train_models[random_indices]
        production = train_quant[random_indices]
        # if i == 0:
        #     seaborn.distplot(agent.produce(all_models), label='initial')
        agent.learn(models, production)
        # if i == 0:
        #     seaborn.distplot(agent.produce(all_models), label='after one')
        #     plt.legend()
        #     plt.show()
        train_distances.append(check_agent_quantifier_similarity(
            agent, train_quant, train_models))
        test_distances.append(check_agent_quantifier_similarity(
            agent, test_quant, test_models))
    plt.scatter(range(len(train_distances)), train_distances)
    plt.scatter(range(len(test_distances)), test_distances)
    plt.show()
    return train_distances, test_distances


def agent_agent_test():
    """
    Shows how the similarity between two agents evolves as the second agent
    sees more and more of the first agent's output
    """
    max_model_size = 10
    all_models = generate_list_models(max_model_size)
    agent1, agent2 = pop.NetworkAgent(max_model_size), pop.NetworkAgent(max_model_size)
    distances = []
    for i in range(1000):
        random_indices = np.random.randint(0, len(all_models),
                                           int(0.7*len(all_models)))
        # models are randomly picked rows of all_models
        models = all_models[random_indices]
        production = agent1.map(models)
        if i == 0:
            seaborn.distplot(agent1.produce(all_models), label='initial')
            plt.show()
        agent2.learn(models, production)
        distances.append(check_agents_similarity(agent1, agent2, all_models))
    plt.scatter(range(len(distances)), distances)
    plt.show()


def produce_random_quants(max_model_size, all_models, n_quants=1, qtype="random"):
    """
    Produces a random quantifier with a given length and optional type.
    Possible types: "random", "mon", "network", "uniform"
    # TODO: implement more quantifier types
    """

    if n_quants > 1:
        return np.column_stack(tuple(produce_random_quants(max_model_size, all_models) for _ in range(n_quants)))

    if n_quants == 1:

        if qtype == "random":
            return np.random.randint(2, size=(len(all_models), 1))

        if qtype == "mon":
            # create random monotone quantifier
            bound_position = np.random.randint(max_model_size)
            direction = np.random.randint(2)
            sizes = np.sum(all_models, axis=1)
            return np.where(
                ((direction == 1) & (sizes >= bound_position)) | ((direction == 0) & (sizes <= bound_position)), 1, 0).reshape(-1, 1)

        elif qtype == "conv":
            # create random convex (possible monotone) quantifier
            bounds_position = np.sort(np.random.choice(max_model_size, size=2, replace=False))
            direction = np.random.randint(2)
            counts = np.sum(all_models, axis=1)
            quant = (counts <= bounds_position[0]) | (counts >= bounds_position[1]) == direction
            return quant.reshape(-1, 1).astype(np.int)

        elif qtype == "network":
            return pop.NetworkAgent(max_model_size).map(all_models).astype(np.int)

        elif qtype == "uniform":
            return pop.UniformRandomAgent(max_model_size).map(all_models).astype(np.int)

        else:
            raise ValueError("Value of quantifier type not recognized. Either mon, conv, or random")


def test_monotonicity_preference():
    """
    Check whether the agents are faster to learn monotone than non-monotone quantifiers
    """
    max_model_size = 7
    all_models = generate_list_models(max_model_size)
    mon_dist = []
    non_mon_dist = []
    for i in range(100):
        mon_quant = produce_random_quants(max_model_size, all_models, qtype="mon")
        non_mon_quant = produce_random_quants(max_model_size, all_models)
        mon_dist.append(agent_quantifier_test(max_model_size, mon_quant))
        non_mon_dist.append(agent_quantifier_test(max_model_size, non_mon_quant))
        print(i)
    plt.plot(np.mean(mon_dist, axis=0), label="Mon")
    plt.plot(np.mean(non_mon_dist, axis=0), label="Non mon")
    plt.legend()
    plt.show()


def check_probability_matching_few_models():
    """
    train neural nets on conflicting model with hand selected models to check if they do probability matching
    """

    repetitions_per_model = 10000
    prob_models = [0.1, 0.9]
    models = [[0, 1, 1], [1, 1, 0]]

    for model, p_model in zip(models, prob_models):
        model = np.tile(model, (repetitions_per_model, 1))
        judgs = np.random.binomial(n=1, p=p_model, size=(repetitions_per_model, 1))
        agent = pop.NetworkAgent(3)
        agent.learn(model, judgs)
    print(np.column_stack((prob_models, agent.produce(models))))


def check_probability_matching_other_agent(real_teacher, uncertainty=1.):
    """
    Pretty much like agent_agent_test but with sampling instead of map and the possibility of simulating the teacher

    If it is done with a real teacher, the average distance between their production probs diminishes because
    the learner gets more and more towards 0.5, which is where the average random teacher starts from.

    So to see whether the learner is actually doing probability matching (rather than always getting maximally uncertain
    when it receives contradictory models) use a simulated teacher, preferably with a low level of uncertainty
    (i.e. a confident teacher).
    """
    max_model_size = 7
    all_models = generate_list_models(max_model_size)
    agent2 = pop.NetworkAgent(max_model_size)
    if real_teacher:
        agent1 = pop.NetworkAgent(max_model_size)
    else:
         agent1 = pop.ConfidenceTeacher(max_model_size, uncertainty)

    distances = []
    for i in range(3000):
        random_indices = np.random.randint(0, len(all_models),
                                           int(0.9*len(all_models))
                                           )
        # models are randomly picked rows of all_models
        models = all_models[random_indices]
        production = agent1.sample(models)
        agent2.learn(models, production)
        distances.append(check_agents_similarity(agent1, agent2, all_models))

    plt.scatter(range(len(distances)), distances)
    plt.show()


def shuffle_learning_model(models, parent_bools, restrict=1.):
    """
    Gets data that an agent learns from and shuffles and restricts it.
    Note that the returned data is simply a restricted and shuffled version of a quantifier, without repeated models.
    :param models: array containing all the models, i.e. all truth value combinations over objects.
    Shape is (# models, # objects)
    :param parent_bools: a full truth value distribution over those models, i.e. a quantifier
    :param restrict: which proportion of the shuffled quantifier to return
    :return:
    """
    learning_subset_indices = np.random.randint(len(models), size=int(len(models) * restrict))
    models = models[learning_subset_indices, :]
    truth_values = parent_bools[learning_subset_indices, :]
    return models, truth_values


def train_test_split(arr, split):
    max_idx = int(len(arr)*split)
    return arr[:max_idx], arr[max_idx:]


def test_order_importance():
    """
    Is there more variability across the quantifiers guessed from the same underlying
    true quantifier, if the order of the observations is shuffled across learners?
    In other words, does the order matter for learning?
    Check whether agents learn a quantifier from the same observations more consistently (even if wrongly)
    if those observations are always in the same order rather than shifted order
    """
    max_model_size = 7
    all_models = generate_list_models(max_model_size)
    n_tests = 1000

    # check for different quantifiers
    for i in range(1):
        quantifier = produce_random_quants(max_model_size, all_models)
        models, truth_values = shuffle_learning_model(all_models, quantifier, restrict=0.7)

        # unshuffled condition
        learners = [pop.NetworkAgent(max_model_size) for _ in range(n_tests)]
        map(lambda agent: agent.learn(models, truth_values, shuffle_by_epoch=False), learners)
        # unshuffled_test is the array of the languages learned from the quantifier without shuffling the model
        unshuffled_test = create_languages_array(learners, all_models)

        # shuffled condition
        learners = [pop.NetworkAgent(max_model_size) for _ in range(n_tests)]
        map(lambda agent: agent.learn(shuffle_learning_model(models, truth_values)), learners)
        # shuffled_test is the array of the languages learned from the quantifier when shuffling the model
        shuffled_test = create_languages_array(learners, all_models)

        # calculate the standard deviation in what the agents learned for every model
        unshuffled_std = np.std(unshuffled_test, axis=1)
        shuffled_std = np.std(shuffled_test, axis=1)

        # calculate the differences in standard deviations for the shuffled and unshuffled group
        # if shuffling has an effect, the differences should be positive
        differences_std = shuffled_std - unshuffled_std

        plt.hist(differences_std, bins=100)
        plt.show()


def measure_upward_monotonicity(all_models, quantifier):
    """

    :param all_models:
    :param quantifier:
    :return:
    """
    if np.all(quantifier) or not np.any(quantifier):
        return 1
    props = []
    #only consider those models for which the quantifier is true (non zero returns indices)
    for i in np.nonzero(quantifier.flatten() == 1)[0]:
        model = all_models[i, :]
        tiled_model = np.tile(model, (len(all_models), 1))
        extends = np.all(tiled_model * all_models == tiled_model, axis=1).flatten()
        #proportion of true extensions of that model for the quantifier
        props.append(np.sum(quantifier[extends])/np.sum(extends))
    return np.mean(props)


@lru_cache(maxsize=None)
def monotonicity_memoized(models_string, quantifier_string):
    """models_string = models.tostring() for 2d int array models;
    quantifier_string = quantifier.tostring() for 1d int array """
    quantifier = np.frombuffer(quantifier_string, dtype=int)
    models = np.frombuffer(models_string, dtype=int).reshape((len(quantifier), -1))
    return measure_monotonicity(models, quantifier)


@lru_cache(maxsize=None)
def quantity_memoized(models_string, quantifier_string):
    """models_string = models.tostring() for 2d int array models;
    quantifier_string = quantifier.tostring() for 1d int array """
    quantifier = np.frombuffer(quantifier_string, dtype=int)
    models = np.frombuffer(models_string, dtype=int).reshape((len(quantifier), -1))
    return check_quantity(models, quantifier)


def measure_monotonicity(all_models, quantifier, type="extensions"):
    if type == "extensions":
        return np.max(
            [measure_upward_monotonicity(all_models, quantifier),
             measure_upward_monotonicity(all_models, 1 - quantifier),
             # downward monotonicity
             measure_upward_monotonicity(1 - all_models, 1 - quantifier),
             measure_upward_monotonicity(1 - all_models, quantifier)])
    elif type == "step":
        pass


def measure_negation_monotonicity(all_models, quantifier):
    return measure_upward_monotonicity(all_models, 1-quantifier)


def quantifiers_in_order_of_monotonicity(l):
    """
    Prints all quantifiers on models of length l in order of monotonicity.
    :param l: Max model size
    :return: None
    """
    models = generate_list_models(l)
    quantifiers = generate_list_models(len(models)).astype(int)
    mon_values = np.empty(shape=(len(quantifiers), 1))
    for i in range(len(quantifiers)):
        mon_values[i] = measure_monotonicity(models, quantifiers[i])
    order_indices = np.argsort(mon_values, axis=0)
    with np.printoptions(threshold=np.inf):
        pprint([(quantifier, mon_value)
                for quantifier, mon_value
                in zip(quantifiers[order_indices].tolist(), mon_values[order_indices].tolist())])


def chance_property_distribution(l, property, agents):
    """
    Plots the distribution of a property in the given set of agents
    :param l: max model length
    :param property: property as a function that take input (models, quantifiers)
    :param agents: list of agents
    :return: None
    """
    models = generate_list_models(l)
    # random_quants = np.random.randint(2, size=(sample_size, len(models)))
    random_quants = [agent.map(models) for agent in agents]
    properties = [property(models, random_quant) for random_quant in random_quants]
    seaborn.distplot(properties)
    plt.show()


def find_proportions_of_quantifiers(quantifiers):
    """
    :param quantifiers: A frame of quantifiers with shape (# models, # quantifiers)
    :return: Two arrays; the first is a version of quantifiers without repetitions, the second is counts
    """
    unique_quantifiers, quantifiers_count = np.unique(quantifiers, return_counts=True, axis=1)
    return unique_quantifiers, quantifiers_count


def counts_without_sort_and_unique(quantifiers):
    """
    :param quantifiers: Array of quantifiers, with shape (# models, # quantifiers)
    :return: An array that for each quantifier says the number of repetitions of that quantifier
    """
    quantifiers, counts = np.unique(quantifiers)
    unsorted_counts = np.zeros(shape=(quantifiers.shape[1], 1))
    for quant_index in range(quantifiers.shape[1]):
        indices_identical = np.where(quantifiers[:, quant_index] == quantifiers)
        unsorted_counts[indices_identical] += 1


def detect_region_of_motion(random_quantifiers, generations):
    """
    TODO: finish this function
    Finds the quantifiers overrepresented in generations given their proportions in a random set of agents
    :param random_quantifiers: array of random quantifiers with shape (# models, # quantifiers)
    :param generations: a 3d array with shape (# generations, # models, # agents)
    :return: two arrays. The first array is the array of quantifiers in generations that are overrepresented given the
    random distribution. The second is a row vector with a measure of the unexpectedness of the quants in the first array.
    """
    # transforms the generation into quantifiers (i.e. 0/1)
    observed_generations = np.around(generations).astype(np.int)
    splitted_generations = [i[0] for i in np.split(observed_generations, np.arange(1, len(observed_generations)))]
    observed_quantifiers = np.column_stack(splitted_generations)

    # get the unique quantifiers and the counts of the quantifiers
    unique_random_quantifiers, counts_random_quantifiers = find_proportions_of_quantifiers(random_quantifiers)
    unique_observed_quantifiers, counts_observed_quantifiers = find_proportions_of_quantifiers(observed_quantifiers)

    # create dictionaries with string versions of quantifiers as keys and their count as value
    observed_counts_dict = {str(quant): count for quant, count in zip(
        unique_observed_quantifiers.T.tolist(), counts_observed_quantifiers.tolist()
    )}
    random_counts_dict = {str(quant): count for quant, count in zip(
        unique_random_quantifiers.T.tolist(), counts_random_quantifiers.tolist()
    )}

    # does add-0.5 Laplace smoothing
    observed_keys_not_in_random_dict = {key: 0. for key in observed_counts_dict.keys() - random_counts_dict.keys()}
    random_counts_dict.update(observed_keys_not_in_random_dict)
    random_counts_dict = {key: value+1 for key, value in random_counts_dict.items()}

    # lognormalizes the smoothed dict of random quantifiers and finds the surprisal every quantifier
    lognormalization_constant_random = np.log2(np.sum(list(random_counts_dict.values())))
    random_surprisals = {
        quant: np.log2(value) - lognormalization_constant_random for quant, value in random_counts_dict.items()}

    # # lognormalizes the dict of observed quantifiers and calculates the surprisal in the observed list
    # lognormalization_constant_observed = np.log2(np.sum(list(observed_counts_dict.values())))
    # observed_surprisals = {
    #     quant: np.log2(value) - lognormalization_constant_observed for quant, value in observed_counts_dict.items()}

    # calculates the difference in surprisals
    # differential_surprisals_observed = {quant: observed_surprisals[quant] - random_surprisals[quant] for quant in observed_surprisals.keys()}

    # calculate the surprisal of observing the observed quantifiers the number of times they were observed
    # given the distribution from the random quantifiers
    conditional_surprisal = {quant: -random_surprisals[quant] * observed_counts_dict[quant] for quant in observed_counts_dict.keys()}
    # pprint(conditional_surprisal)
    return conditional_surprisal


def inter_generational_movement_speed(generations, parents):
    """
    Finds the speed at which languages as changing as the generations go by
    In an ideal case, it should start fast and then get slow as the simulation finds a spot it likes
    in the language space.
    TODO: Test this function
    :param all_models: all models as rows
    :param generations: a 3d array with shape (generations, models, agents)
    :param parents: a dataframe with shape (len(generations)-1), generations.shape[2]) that gives for each agent
    the index of its parent in the previous generation
    :return: the movement speed for each successive generation.
    """
    changes = [0, ]
    for gen_index in range(1, len(generations)):
        children = generations[gen_index]
        parent_qs = generations[gen_index-1, :, parents[gen_index]].T
        changes.append(L1_dist(parent_qs, children))
    return np.array(changes)


def check_quantity(list_models, map_lang):
    # TODO: vectorize across first axis (i.e. generation) of the 3-d results array
    """
    Calculates quantity as 1 - H(quantifier is true at the model | model size)
    """
    # prob_num is the array with the unconditional probability of each # of 1s in a random model
    count_ones = np.count_nonzero(list_models, axis=1)
    num_arrays_of_length = np.unique(count_ones, return_counts=True)[1]
    prob_num = num_arrays_of_length / sum(num_arrays_of_length)

    model_sizes = np.sum(list_models, axis=1)
    true_model_sizes = model_sizes[np.nonzero(map_lang)]
    num_true_by_size = np.bincount(true_model_sizes,
                                   minlength=max(model_sizes)+1)

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


def check_quantifier_ultrafilter(all_models, quantifier):
    """
    Check whether a given quantifier is an ultrafilter
    An ultrafilter is a quantifier whose truth value simply depends on one of the objects in the model.
    :param all_models: An array with shape (# models, # objects)
    :param quantifier: A column vector with size # models
    :return: Whether quantifier is an ultrafilter or not
    """
    map_quant = np.around(quantifier).astype(int).reshape((-1,1))
    tiled_quant = np.tile(map_quant, reps=(1, all_models.shape[1]))
    identity = tiled_quant == all_models
    columns_identical = np.all(identity, axis=0)
    # relies on the fact that an ultrafilter can only depend on a single object, not more than one
    # and therefore the output of nonzero will be unique if there is one at all
    return np.nonzero(columns_identical)[0][0] if np.any(columns_identical) else -1


if __name__ == '__main__':
    # a = produce_random_quants(10, generate_list_models(10), 1000000, qtype="network")
    models = generate_list_models(3)
    # quantifier = produce_random_quants(3, models)
    quantifier = [
         [0],
         [0],
         [1],
         [0],
         [1],
         [0],
         [0],
         [1]]
    print(quantifier)
    print(models)
    print(check_quantifier_ultrafilter(models, quantifier))
    # np.save("/exports/eddie/scratch/s1569804/random_network_quantifiers", a)
