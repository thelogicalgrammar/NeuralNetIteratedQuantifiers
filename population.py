import random as rnd
import pandas as pd
import utilities as util


class Agent:
    def __init__(self):
        pass

    def learn(self, data):
        """
        CONTROLLED BY NEURAL NET
        Data is a Series with inputs as indices and 0/1 as values
        (basically it's the whole training data from the parent)
        """
        pass       # just to check whether the rest of the simulation runs

    def produce(self, agent_input):
        """
        CONTROLLED BY NEURAL NET
        Returns 0 or 1 for whether the agent's quantifier is compatible with agent_input (which is an array of bits)
        """
        return rnd.choice([0,1])     # just to check whether the rest of the simulation runs


class Population:
    def __init__(self, size, possible_inputs):
        # all binary strings of the wanted lengths
        self.possible_inputs = possible_inputs
        # list of agent objects (I am not sure what you need to initialize your neural net)
        self.agents = [Agent() for _ in range(size)]

    def learn_from_population(self, parent_pop, bottleneck_size):
        """
        parent_languages is the DataFrame with all the languages for all inputs from previous population
        Each child in self.agents is selected in turns. A random parent from old pop is selected.
        A Series "data" is created by sampling with replacement from the parent's language
        The child is trained
        """
        parent_languages = parent_pop.languages
        for child in self.agents:
            parent_lang = parent_languages[rnd.choice(parent_languages.columns)]
            data = parent_lang.sample(bottleneck_size, replace=True)
            child.learn(data)
        self.languages = util.create_languages_dataframe(self.agents, self.possible_inputs)

    def really_really_learn(self, wanted_languages, check_every=10):
        """
        This function makes absolutely sure that the agents really learn the intended languages
        It teaches each agent the language that they are supposed to learn and checks whether they have learned it
        every check_every teachings
        It stops training when there is no difference between the language they are supposed to learn and the
        language they actually learned
        """
        # Maybe I should implement some tolerance for error?
        # this bit can be made more efficient by focusing on training only the agents that haven't learned yet
        while True:
            for n in range(check_every):
                # go through a whole training session (all agents with all inputs)
                [agent.learn(wanted_languages[agent_n]) for agent_n, agent in enumerate(self.agents)]
            actual_languages = util.create_languages_dataframe(self.agents, self.possible_inputs)
            if actual_languages.equals(wanted_languages):
                self.languages = util.create_languages_dataframe(self.agents, self.possible_inputs)
                return

    def information(self):
        return self.languages
