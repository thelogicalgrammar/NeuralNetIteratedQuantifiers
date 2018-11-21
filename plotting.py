import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import iteration


def language_distributions(population):
    langs = population.languages.as_matrix()
    # ignore first, always 0.5 for some reason
    for poss_input in range(1, len(langs)):
        kde = scipy.stats.gaussian_kde(langs[poss_input, :])
        x = np.linspace(0, 1, 200)
        plt.plot(x, kde(x))
    plt.xlabel("Generations")
    plt.ylabel("")
    plt.show()


def violin_plots_confidence(n_generations, n_agents, bottleneck, length_inputs):
    # FIX: it works for any number of agents and generations
    # but doesn't show the difference between generations and agents with colours

    data = iteration.iterate(n_generations, n_agents, bottleneck, length_inputs)

    plt.violinplot(data.as_matrix())

    plt.title("Bottleneck: {}".format(bottleneck))
    plt.xlabel("Generations")
    plt.ylabel("Confidence for all inputs")
    plt.show()

if __name__ == "__main__":

    input_values = {
        "n_generations": 50,
        "n_agents": 1,
        "bottleneck": 4000,
        "length_inputs": 5
    }

    violin_plots_confidence(**input_values)
