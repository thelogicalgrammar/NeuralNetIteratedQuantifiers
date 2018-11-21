import random as rnd
import numpy as np
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

    def learn(self, inputs, parent_bools, batch_size=12, epochs=1, validation_split=0.25):
        """
        CONTROLLED BY NEURAL NET
        inputs is an array with shape (# inputs, size model)
        parent_bools is the parent's judgment for each of the inputs
        """
        x = inputs
        y = parent_bools
        self._model.fit(x, y, batch_size=batch_size, epochs=epochs,
                validation_split=validation_split)

    def produce(self, agent_input):
        """
        CONTROLLED BY NEURAL NET
        Returns probability assigned to 1 for list of inputs (array of lists of bits)
        """
        x = agent_input
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
    def __init__(self, size, input_length):
        self.input_length = input_length
        # list of agent objects
        self.agents = [Agent(input_length) for _ in range(size)]

    def learn_from_population(self, parent_pop, bottleneck_size):
        """
        Each child in self.agents is selected in turn. A random parent from old pop is selected with replacement.
        inputs is created as a random array of booleans (there can be repeated rows, I don't know if this is fine)
        The child is trained on the production of the parent for the vectors in inputs
        """
        for child in self.agents:
            parent = rnd.choice(parent_pop.agents)
            inputs = np.random.randint(0, 2, size=(bottleneck_size, self.input_length))
            # make the parent produce the data for the sampled inputs
            parent_bools = parent.map(inputs)
            child.learn(inputs, parent_bools)
