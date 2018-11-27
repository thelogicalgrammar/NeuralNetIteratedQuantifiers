import random as rnd
import numpy as np
from utilities import generate_list_models
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, model_length):
        super(MLP, self).__init__()
        # TODO: parameterize this structure
        self.fc1 = nn.Linear(model_length, 16)
        self.fc2 = nn.Linear(16, 16)
        self.bn = nn.BatchNorm1d(16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # output is [batch_size, 1]
        x = self.bn(x)
        return torch.sigmoid(self.output(x))


class Agent:

    def produce(self, agent_model):
        pass

    def map(self, agent_model):
        """
        Returns 0 or 1, by 'argmaxing' the probabilities, i.e. returning
        whichever one had higher probability.
        """
        return np.around(self.produce(agent_model))

    def sample(self, agent_model):
        """
        Returns 0 or 1 for some models, by sampling from the network's output
        probability.
        """

        probabilities = self.produce(agent_model)
        uniforms = np.random.rand(len(probabilities), 1)
        # choices: (N, 1) shape of booleans
        choices = uniforms < probabilities
        # if want 1/0, return choices.astype(int)
        return choices.astype(int)


class FixedAgent(Agent):

    def __init__(self, model_length):
        self.possible_models = generate_list_models(model_length)
        self.confidence = None

    def produce(self, agent_model):
        # returns the confidence for each model (row) of agent_model
        indices = np.apply_along_axis(
            lambda row: np.argwhere(np.all(row == self.possible_models, axis=1)),
            axis=1,
            arr=agent_model
        ).flatten()
        return self.confidence[indices]


class ConfidenceTeacher(FixedAgent):
    def __init__(self, model_length, uncertainty):
        """
        model length is the max model length
        0 < uncertainty
        For values < than 1, the agent prefers either true or false
        For values > than 1, the agent prefers to be neutral (i.e. around 0.5)
        """
        super(ConfidenceTeacher, self).__init__(model_length)
        self.confidence = np.random.beta(uncertainty, uncertainty,
                                         size=(len(self.possible_models), 1))


class UniformRandomAgent(FixedAgent):

    def __init__(self, model_length):
        super(UniformRandomAgent, self).__init__(model_length)
        self.confidence = np.random.uniform(
            size=(len(self.possible_models), 1))


class NetworkAgent(Agent):
    def __init__(self, model_length):
        self.model = MLP(model_length)

    def learn(self, models, parent_bools, batch_size=32, num_epochs=3, shuffle_by_epoch=True):
        # TODO: play with options here?
        optim = torch.optim.Adam(self.model.parameters())
        for epoch in range(num_epochs):
            # re-order the data each epoch
            permutation = np.random.permutation(len(models)) if shuffle_by_epoch else np.arange(len(models))
            # -- [bottleneck_size, model-length]
            x = models[permutation]
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

    def produce(self, agent_model):
        return self.model(agent_model).detach().numpy()


class Population:
    def __init__(self, size, model_length):
        self.model_length = model_length
        # list of agent objects
        self.agents = [NetworkAgent(model_length) for _ in range(size)]

    def learn_from_population(self, parent_pop, bottleneck_size, num_epochs=1):
        """
        Each child in self.agents is selected in turn. A random parent from old pop is selected with replacement.
        models is created as a random array of booleans (there can be repeated rows, I don't know if this is fine)
        The child is trained on the production of the parent for the vectors in models
        """
        for child in self.agents:
            parent = rnd.choice(parent_pop.agents)
            models = np.random.randint(0, 2, size=(bottleneck_size, self.model_length))
            # make the parent produce the data for the sampled models
            parent_bools = parent.map(models)
            child.learn(models, parent_bools, num_epochs)
