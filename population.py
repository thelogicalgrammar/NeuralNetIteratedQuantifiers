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
        Returns probability assigned to 1 for list of inputs (array of lists of bits)
        """
        x = np.array(agent_input.tolist())
        return self._model.predict(x)

    def map(self, agent_input):
        """
        Returns 0 or 1, by 'argmaxing' the probabilities, i.e. returning
        whichever one had higher probability.
        """
        return np.around(self.produce(agent_input))

    def sample(self, agent_input):
        """
        Returns 0 or 1 for some inputs, by sampling from the network's output
        probability.
        """
        probabilities = self.produce(agent_input)
        uniforms = np.random.rand(len(probabilities), 1)
        # choices: (N, 1) shape of booleans
        choices = uniforms < probabilities
        # if want 1/0, return choices.astype(int)
        return choices.astype(int)


class Population:
    def __init__(self, size, possible_inputs):
        # all binary strings of the wanted lengths
        self.possible_inputs = possible_inputs
        # list of agent objects
        input_length = len(possible_inputs[0])
        self.agents = [Agent(input_length) for _ in range(size)]

    def learn_from_population(self, parent_pop, bottleneck_size):
        """
        parent_languages is the DataFrame with all the languages for all inputs from previous population
        Each child in self.agents is selected in turns. A random parent from old pop is selected.
        A Series "data" is created by sampling with replacement from the parent's language
        The child is trained
        """
        for child in self.agents:
            parent = rnd.choice(parent_pop.agents)
            data_for_parent = self.possible_inputs.sample(bottleneck_size, replace=True)  # sample func from pd.Series
            data_for_child = pd.Series(parent.map(data_for_parent).flatten(), data_for_parent)
            child.learn(data_for_child)
        self.languages = util.create_languages_dataframe(self.agents, self.possible_inputs)

    def information(self):
        return self.languages
