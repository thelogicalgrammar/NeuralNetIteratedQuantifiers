import numpy as np
from utilities import generate_list_inputs, create_languages_array
import population as pop
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn
import random as rnd


class SimulatedTeacher:
    def __init__(self, input_length, uncertainty):
        """
        input length is the max model length
        0 < uncertainty
        For values < than 1, the agent prefers either true or false
        For values > than 1, the agent prefers to be neutral (i.e. around 0.5)
        """
        self.possible_inputs = generate_list_inputs(input_length)
        self.confidence = np.random.beta(uncertainty, uncertainty, size=(self.possible_inputs.shape[0], 1))

    def produce(self, agent_input):
        # returns the confidence for each model (row) of agent_input
        indices = np.apply_along_axis(
            lambda row: np.argwhere(np.all(row == self.possible_inputs, axis=1)),
            axis=1,
            arr=agent_input
        ).flatten()
        return self.confidence[indices]

    def map(self, agent_input):
        """
        Returns 0 or 1, by 'argmaxing' the probabilities, i.e. returning
        whichever one had higher probability.
        """
        return np.around(self.produce(agent_input))

    def sample(self, agent_input):
        probabilities = self.produce(agent_input)
        uniforms = np.random.rand(len(probabilities), 1)
        # choices: (N, 1) shape of booleans
        choices = uniforms < probabilities
        # if want 1/0, return choices.astype(int)
        return choices.astype(int)


def check_agents_similarity(agent1, agent2, possible_inputs, map=False):
    """
    returns the proportion of the inputs about which the agents disagree if map==True
    or the average difference between their confidence level about each input if map==False
    """
    judgments = create_languages_array([agent1, agent2], possible_inputs, map)
    # proportion of models where the judgments of the two agents are different
    prop_different = np.sum(np.abs(judgments[:, 0] - judgments[:, 1])) / possible_inputs.shape[0]
    return prop_different


def check_agent_quantifier_similarity(agent, quantifier, possible_inputs):
    judgments = agent.map(possible_inputs)
    prop_different = np.sum(np.abs(quantifier - judgments)) / possible_inputs.shape[0]
    return prop_different


def agent_quantifier_test(input_length=None, quant=False):
    if not input_length:
        input_length = 10
    possible_inputs = generate_list_inputs(input_length)
    if type(quant) == np.ndarray:
        quantifier = np.random.randint(0, 2, size=(len(possible_inputs), 1))
    else:
        quantifier = quant
    agent = pop.Agent(input_length)
    distances = []
    for i in range(1000):
        random_indices = np.random.randint(0, possible_inputs.shape[0],
                                           int(0.7*len(possible_inputs)))
        inputs = possible_inputs[random_indices]
        production = quantifier[random_indices]
        # if i == 0:
        #     seaborn.distplot(agent.produce(possible_inputs), label='initial')
        agent.learn(inputs, production)
        # if i == 0:
        #     seaborn.distplot(agent.produce(possible_inputs), label='after one')
        #     plt.legend()
        #     plt.show()
        distances.append(check_agent_quantifier_similarity(agent, quantifier, possible_inputs))
    return distances


def agent_agent_test():
    """
    Shows how the similarity between two agents evolves as the second agent
    sees more and more of the first agent's output
    """
    input_length = 10
    possible_inputs = generate_list_inputs(input_length)
    agent1, agent2 = pop.Agent(input_length), pop.Agent(input_length)
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


def random_mon_quant(input_length, possible_inputs):
    # create random monotone quantifier
    bound_position = np.random.randint(input_length)
    direction = np.random.randint(2)
    sizes = np.sum(possible_inputs, axis=1)
    return np.where(
        ((direction == 1) & (sizes >= bound_position)) | ((direction == 0) & (sizes <= bound_position)), 1, 0).reshape(-1, 1)


def random_conv_quant(input_length, possible_inputs):
    # create random convex (possible monotone) quantifier
    bounds_position = np.sort(np.random.choice(input_length, size=2, replace=False))
    direction = np.random.randint(2)
    counts = np.sum(possible_inputs, axis=1)
    quant = (counts <= bounds_position[0]) | (counts >= bounds_position[1]) == direction
    return quant.reshape(-1, 1).astype(np.int)


def test_monotonicity_preference():
    """
    Check whether the agents are faster to learn monotone than non-monotone quantifiers
    """
    input_length = 10
    possible_inputs = generate_list_inputs(input_length)
    mon_dist = []
    non_mon_dist = []
    for i in range(100):
        mon_quant = random_mon_quant(input_length, possible_inputs)
        non_mon_quant = random_conv_quant(input_length, possible_inputs)
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
        agent = pop.Agent(3)
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
    agent2 = pop.Agent(input_length)
    if real_teacher:
        agent1 = pop.Agent(input_length)
    else:
         agent1 = SimulatedTeacher(input_length, uncertainty)

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


if __name__ == '__main__':
    # check_probability_matching_other_agent(real_teacher=False, uncertainty=1.)
    # check_probability_matching_other_agent(real_teacher=False, uncertainty=0.5)

    test_monotonicity_preference()