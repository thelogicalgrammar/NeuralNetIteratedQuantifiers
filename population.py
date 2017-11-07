import random as rnd
import pandas as pd
import numpy as np
import utilities as util
import keras


class Agent:
    def __init__(self, input_length):

        # TODO: parameterize model structure
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(8, input_dim=input_length, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
        self._model = model

    def learn(self, data, batch_size=12, epochs=1, validation_split=0.25):
        """
        CONTROLLED BY NEURAL NET
        Data is a Series with inputs as indices and 0/1 as values
        (basically it's the whole training data from the parent)
        """
        x = np.array(data.index.tolist())
        y = data.values
        self._model.fit(x, y, batch_size=batch_size, epochs=epochs,
                validation_split=validation_split)

    def produce(self, agent_input):
        """
        CONTROLLED BY NEURAL NET
        Returns 0 or 1 for whether the agent's quantifier is compatible with agent_input (which is an array of bits)
        """
        x = np.array(agent_input.tolist())
        predictions = self._model.predict(x)
        # TODO: argmax here, or only later if needed?
        return predictions


class Population:
    def __init__(self, size, possible_inputs):
        # all binary strings of the wanted lengths
        self.possible_inputs = possible_inputs
        # list of agent objects (I am not sure what you need to initialize your neural net)
        input_length = len(possible_inputs[0])
        self.agents = [Agent(input_length) for _ in range(size)]

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
                [agent.learn(wanted_languages[agent_n], epochs=10,
                    validation_split=0.0) for agent_n, agent in enumerate(self.agents)]
            actual_languages = util.create_languages_dataframe(self.agents, self.possible_inputs)
            # TODO: fix this equality check
            if util.equal_languages(actual_languages, wanted_languages):
                self.languages = util.create_languages_dataframe(self.agents, self.possible_inputs)
                return

    def information(self):
        return self.languages
