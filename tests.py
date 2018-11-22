import numpy as np
from utilities import generate_list_inputs, create_languages_array
import population as pop
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn


def check_agents_similarity(agent1, agent2, possible_inputs):
    """
    returns the proportion of the inputs about which the agents disagree
    """
    judgments = create_languages_array([agent1, agent2], possible_inputs,
                                       map=True)
    # proportion of models where the judgments of the two agents are different
    prop_different = np.sum(np.abs(judgments[:, 0] - judgments[:, 1])) / possible_inputs.shape[0]
    return prop_different


def check_agent_quantifier_similarity(agent, quantifier, possible_inputs):
    judgments = agent.map(possible_inputs)
    prop_different = np.sum(np.abs(quantifier - judgments)) / possible_inputs.shape[0]
    return prop_different


def agent_quantifier_test():
    input_length = 10
    possible_inputs = generate_list_inputs(input_length)
    quantifier = np.random.randint(0, 2, size=(len(possible_inputs), 1))
    agent = pop.Agent(input_length)
    distances = []
    for i in range(1000):
        random_indices = np.random.randint(0, possible_inputs.shape[0],
                                           int(0.7*len(possible_inputs)))
        inputs = possible_inputs[random_indices]
        production = quantifier[random_indices]
        if i == 0:
            seaborn.distplot(agent.produce(possible_inputs), label='initial')
        agent.learn(inputs, production)
        if i == 0:
            seaborn.distplot(agent.produce(possible_inputs), label='after one')
            plt.legend()
            plt.show()
        distances.append(check_agent_quantifier_similarity(agent, quantifier, possible_inputs))
    plt.scatter(range(len(distances)), distances)
    plt.show()


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


if __name__ == '__main__':
    agent_quantifier_test()
    agent_agent_test()
