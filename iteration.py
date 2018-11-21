import utilities as util
import population as pop
import numpy as np


def iterate(n_generations, n_agents, bottleneck, length_inputs):
    # generate all the binary strings of the given length
    # possible_inputs is a 2d array, where each row is a model
    possible_inputs = util.generate_list_inputs(length_inputs)
    # create first generation
    parent_generation = pop.Population(n_agents, length_inputs)
    # data is a 3-d numpy array with shape (# gen, # possible inputs, # agents)
    data = np.empty(shape=(n_generations+1, possible_inputs.shape[0], n_agents))

    for n in range(n_generations):
        # the new generation is created
        child_generation = pop.Population(n_agents, length_inputs)
        # the new generation learns from the old one
        child_generation.learn_from_population(parent_generation, bottleneck)
        # stores some data to be analyzed later!
        data[n] = util.create_languages_array(parent_generation.agents, possible_inputs)
        # the new generation becomes the old generation, ready to train the next generation
        parent_generation = child_generation
        print("Done generation {} out of {} \n\n".format(n, n_generations))

    # stores the data from the last trained generation
    data[n_generations] = util.create_languages_array(parent_generation.agents, possible_inputs)
    return data


if __name__ == "__main__":
    input_values = {
        "n_generations": 5,
        "n_agents": 1,
        "bottleneck": 100,
        "length_inputs": 3
    }

    data = iterate(**input_values)
    print(data)
