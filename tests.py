import numpy as np
from utilities import generate_list_inputs, create_languages_array
import population as pop
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn
import random as rnd


def L1_dist(column1, column2):
    return np.sum(np.absolute(column1 - column2)) / len(column1)


def check_agents_similarity(agent1, agent2, possible_inputs, mapping=False):
    """
    returns the proportion of the inputs about which the agents disagree if map==True
    or the average difference between their confidence level about each input if map==False
    """
    judgments = create_languages_array([agent1, agent2], possible_inputs,
                                       mapping)
    # proportion of models where the judgments of the two agents are different
    prop_different = L1_dist(judgments[:, 0], judgments[:, 1])
    return prop_different


def check_agent_quantifier_similarity(agent, quantifier, possible_inputs):
    judgments = agent.map(possible_inputs)
    prop_different = L1_dist(quantifier, judgments)
    return prop_different


def agent_quantifier_test(input_length=None, quant=None, train_split=0.75):
    if not input_length:
        input_length = 10
    possible_inputs = generate_list_inputs(input_length)
    quantifier = quant or np.random.randint(0, 2, size=(len(possible_inputs), 1))
    possible_inputs, quantifier = shuffle_learning_input(
        possible_inputs, quantifier)
    train_inputs, test_inputs = train_test_split(possible_inputs, train_split)
    train_quant, test_quant = train_test_split(quantifier, train_split)

    agent = pop.NetworkAgent(input_length)
    train_distances, test_distances = [], []
    for i in range(1000):
        random_indices = np.random.randint(0, len(train_inputs),
                                           int(0.7*len(train_inputs)))
        inputs = train_inputs[random_indices]
        production = train_quant[random_indices]
        # if i == 0:
        #     seaborn.distplot(agent.produce(possible_inputs), label='initial')
        agent.learn(inputs, production)
        # if i == 0:
        #     seaborn.distplot(agent.produce(possible_inputs), label='after one')
        #     plt.legend()
        #     plt.show()
        train_distances.append(check_agent_quantifier_similarity(
            agent, train_quant, train_inputs))
        test_distances.append(check_agent_quantifier_similarity(
            agent, test_quant, test_inputs))
    plt.scatter(range(len(train_distances)), train_distances)
    plt.scatter(range(len(test_distances)), test_distances)
    plt.show()
    return train_distances, test_distances


def agent_agent_test():
    """
    Shows how the similarity between two agents evolves as the second agent
    sees more and more of the first agent's output
    """
    input_length = 10
    possible_inputs = generate_list_inputs(input_length)
    agent1, agent2 = pop.NetworkAgent(input_length), pop.NetworkAgent(input_length)
    distances = []
    for i in range(1000):
        random_indices = np.random.randint(0, possible_inputs.shape[0],
                                           int(0.7*len(possible_inputs)))
        # inputs are randomly picked rows of possible_inputs
        inputs = possible_inputs[random_indices]
        production = agent1.map(inputs)
        if i == 0:
            seaborn.distplot(agent1.produce(possible_inputs), label='initial')
            plt.show()
        agent2.learn(inputs, production)
        distances.append(check_agents_similarity(agent1, agent2, possible_inputs))
    plt.scatter(range(len(distances)), distances)
    plt.show()


def random_quant(input_length, possible_inputs, qtype="random"):

    if qtype=="random":
        return np.random.randint(2, size=(possible_inputs.shape[0], 1))

    if qtype=="mon":
        # create random monotone quantifier
        bound_position = np.random.randint(input_length)
        direction = np.random.randint(2)
        sizes = np.sum(possible_inputs, axis=1)
        return np.where(
            ((direction == 1) & (sizes >= bound_position)) | ((direction == 0) & (sizes <= bound_position)), 1, 0).reshape(-1, 1)

    elif qtype=="conv":
        # create random convex (possible monotone) quantifier
        bounds_position = np.sort(np.random.choice(input_length, size=2, replace=False))
        direction = np.random.randint(2)
        counts = np.sum(possible_inputs, axis=1)
        quant = (counts <= bounds_position[0]) | (counts >= bounds_position[1]) == direction
        return quant.reshape(-1, 1).astype(np.int)

    else:
        # TODO: implement more quantifier types
        raise ValueError("Value of quantifier type not recognized. Either mon, conv, or random")


def test_monotonicity_preference():
    """
    Check whether the agents are faster to learn monotone than non-monotone quantifiers
    """
    input_length = 7
    possible_inputs = generate_list_inputs(input_length)
    mon_dist = []
    non_mon_dist = []
    for i in range(100):
        mon_quant = random_quant(input_length, possible_inputs, type="mon")
        non_mon_quant = random_quant(input_length, possible_inputs)
        mon_dist.append(agent_quantifier_test(input_length, mon_quant))
        non_mon_dist.append(agent_quantifier_test(input_length, non_mon_quant))
        print(i)
    plt.plot(np.mean(mon_dist, axis=0), label="Mon")
    plt.plot(np.mean(non_mon_dist, axis=0), label="Non mon")
    plt.legend()
    plt.show()


def check_probability_matching_few_models():
    # do neural nets do probability matching?
    # train them on conflicting input with hand selected models
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
    when it receives contradictory inputs) use a simulated teacher, preferably with a low level of uncertainty
    (i.e. a confident teacher).
    """
    input_length = 7
    possible_inputs = generate_list_inputs(input_length)
    agent2 = pop.NetworkAgent(input_length)
    if real_teacher:
        agent1 = pop.NetworkAgent(input_length)
    else:
         agent1 = pop.SimulatedTeacher(input_length, uncertainty)

    np.set_printoptions(suppress=True)
    print(create_languages_array([agent1, agent2], possible_inputs, map=False))
    distances = []
    for i in range(3000):
        random_indices = np.random.randint(0, possible_inputs.shape[0],
                                           int(0.9*len(possible_inputs))
                                           )
        # inputs are randomly picked rows of possible_inputs
        inputs = possible_inputs[random_indices]
        production = agent1.sample(inputs)
        agent2.learn(inputs, production)
        distances.append(check_agents_similarity(agent1, agent2, possible_inputs))
        print(create_languages_array([agent1, agent2], possible_inputs, map=False))
    np.set_printoptions(suppress=False)
    plt.scatter(range(len(distances)), distances)
    plt.show()


def shuffle_learning_input(inputs, parent_bools, restrict=1.):
    learning_subset_indices = np.random.randint(inputs.shape[0], size=int(inputs.shape[0] * restrict))
    models = inputs[learning_subset_indices, :]
    truth_values = parent_bools[learning_subset_indices, :]
    return models, truth_values


def train_test_split(arr, split):
    max_idx = int(len(arr)*split)
    return arr[:max_idx], arr[max_idx:]


def test_order_importance():
    """
    Is there more variability across the guessed quantifiers if the order is shuffled every time?
    In other words, does the order matter for learning?
    Check whether agents learn a quantifier from the same observations more consistently (even if wrongly)
    if those observations are always in the same order rather than shifted order
    """
    input_length = 7
    possible_inputs = generate_list_inputs(input_length)
    n_tests = 1000

    # check for different quantifiers
    for i in range(1):
        quantifier = random_quant(input_length, possible_inputs)
        models, truth_values = shuffle_learning_input(possible_inputs, quantifier, restrict=0.7)

        # unshuffled condition
        learners = [pop.NetworkAgent(input_length) for _ in range(n_tests)]
        map(lambda agent: agent.learn(models, truth_values, shuffle_by_epoch=False), learners)
        # unshuffled_test is the array of the languages learned from the quantifier without shuffling the input
        unshuffled_test = create_languages_array(learners, possible_inputs)

        # shuffled condition
        learners = [pop.NetworkAgent(input_length) for _ in range(n_tests)]
        map(lambda agent: agent.learn(shuffle_learning_input(models, truth_values)), learners)
        # shuffled_test is the array of the languages learned from the quantifier when shuffling the input
        shuffled_test = create_languages_array(learners, possible_inputs)

        # calculate the standard deviation in what the agents learned for every model
        unshuffled_std = np.std(unshuffled_test, axis=1)
        shuffled_std = np.std(shuffled_test, axis=1)

        print(unshuffled_std)
        print(shuffled_std)

        # calculate the differences in standard deviations for the shuffled and unshuffled group
        # if shuffling has an effect, the differences should be positive
        differences_std = shuffled_std - unshuffled_std

        plt.hist(differences_std, bins=100)
        plt.show()


def measure_upward_monotonicity(possible_inputs, quantifier):
    if np.all(quantifier) or not np.any(quantifier):
        return 1
    props = []
    #only consider those models for which the quantifier is true (non zero returns indices)
    for i in np.nonzero(quantifier.flatten()==1)[0]:
        model = possible_inputs[i, :]
        tiled_model = np.tile(model, (possible_inputs.shape[0], 1))
        extends = np.all(tiled_model*possible_inputs == tiled_model, axis=1).flatten()
        #proportion of true extensions of that model for the quantifier
        props.append(np.sum(quantifier[extends])/np.sum(extends))
    return np.mean(props)


def measure_monotonicity(possible_inputs,quantifier):
    return np.max([measure_upward_monotonicity(possible_inputs, quantifier),
        measure_upward_monotonicity(possible_inputs, np.logical_not(quantifier))])


if __name__ == '__main__':
    # check_probability_matching_other_agent(real_teacher=False, uncertainty=1.)
    # check_probability_matching_other_agent(real_teacher=False, uncertainty=0.5)
    agent_quantifier_test()
    #test_monotonicity_preference()
    #test_order_importance()
