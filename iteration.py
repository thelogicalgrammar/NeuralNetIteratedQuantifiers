import utilities as util
import population as pop
import pandas as pd
import numpy as np


def iterate(n_generations, n_agents, bottleneck, length_inputs):

    # generate all the binary strings of the given length
    possible_inputs = util.generate_list_inputs(length_inputs)

    # first generation is created
    parent_generation = pop.Population(n_agents, possible_inputs)

    # determines the list of initial random languages which is needed for training
    parent_generation.languages = util.create_languages_dataframe(
        parent_generation.agents, parent_generation.possible_inputs)

    multi = pd.MultiIndex.from_product([np.arange(n_generations+1), np.arange(n_agents)], names=["Gen", "Agents"])
    data = pd.DataFrame(index=possible_inputs, columns=multi)

    for n in range(n_generations):

        # the new generation is created
        child_generation = pop.Population(n_agents, possible_inputs)

        # the new generation learns from the old one
        child_generation.learn_from_population(parent_generation, bottleneck)

        # stores some data to be analyzed later!
        data[n] = parent_generation.information()

        # the new generation becomes the old generation, ready to train the next generation
        parent_generation = child_generation

        print("Done generation {} out of {} \n\n".format(n, n_generations))

    # stores the data from the last trained generation
    data[n_generations] = parent_generation.information()

    return data

if __name__ == "__main__":
    input_values = {
        "n_generations": 5,
        "n_agents": 2,
        "bottleneck": 100,
        "length_inputs": 5
    }

    data = iterate(**input_values)
    for generation, languages in data.groupby(level=0, axis=1):
        languages.columns = languages.columns.droplevel(0)
        print(util.check_quantity(languages))
