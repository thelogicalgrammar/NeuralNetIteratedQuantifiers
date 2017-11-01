import utilities as util
import population as pop

n_generations = 100
n_agents = 10
bottleneck = 10
length_inputs = 7

# generate all the binary strings of the given length
possible_inputs = util.generate_list_inputs(length_inputs)

# initial_languages is a dataframe with possible inputs as row and agents as columns
# a cell says whether the agent (column) applies the quantifier to the input (row)
initial_languages = util.random_quantifiers(n_agents, possible_inputs)

# first generation is created
parent_generation = pop.Population(n_agents, possible_inputs)

# determines the list of initial unintended random languages which is useful for initial training
parent_generation.languages = util.create_languages_dataframe(
    parent_generation.agents, parent_generation.possible_inputs)

# we make really sure that the agents in the first generation have the intended random languages
parent_generation.really_really_learn(initial_languages)

data = []
for n in range(n_generations):

    # the new generation is created
    child_generation = pop.Population(n_agents, possible_inputs)

    # the new generation learns from the old one
    child_generation.learn_from_population(parent_generation, bottleneck)

    # stores some data to be analyzed later!
    data.append(parent_generation.information())

    # the new generation becomes the old generation, ready to train the next generation
    parent_generation = child_generation

    print("Done generation {} out of {}".format(n, n_generations))

# stores the data from the last trained generation
data.append(parent_generation.information())
