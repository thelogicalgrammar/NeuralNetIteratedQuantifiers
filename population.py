import random as rnd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_length):
        super(MLP, self).__init__()
        # TODO: parameterize this structure
        self.fc1 = nn.Linear(input_length, 16)
        self.fc2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # output is [batch_size, 1]
        return torch.sigmoid(self.output(x))


class Agent:
    def __init__(self, input_length):
        self.model = MLP(input_length)

    def learn(self, inputs, parent_bools, batch_size=32, epochs=4):
        # TODO: play with options here?
        optim = torch.optim.Adam(self.model.parameters())
        for epoch in range(epochs):
            # re-order the data each epoch
            permutation = np.random.permutation(len(inputs))
            # -- [bottleneck_size, input-length]
            x = inputs[permutation]
            y = parent_bools[permutation]
            num_batches = int(len(x) / batch_size)
            for batch in range(num_batches):
                optim.zero_grad()
                # get net predictions
                batch_x = x[batch*batch_size:(batch+1)*batch_size]
                predictions = self.model(batch_x)
                batch_y = y[batch*batch_size:(batch+1)*batch_size]
                # loss
                loss = F.binary_cross_entropy(predictions,
                                              torch.Tensor(batch_y))
                # back-propagate the loss
                loss.backward()
                optim.step()

    def produce(self, agent_input):
        return self.model(agent_input).detach().numpy()

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
