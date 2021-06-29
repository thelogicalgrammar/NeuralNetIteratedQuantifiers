import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import scipy.stats
import numpy as np
import seaborn as sns
import pandas as pd
import itertools as it
import matplotlib
from glob import glob
from os import path

import sys
sys.path.insert(0, '../')
import tests
import population as pop
import analysis
import utilities
import iteration
from utilities import generate_list_models
from population import NetworkAgent


def print_quantifier(quantifier, models, indices_model=None):
    for i, model in enumerate(models):
        if quantifier[i-1] != quantifier[i]:
            print("\n")
        if indices_model is not None:
            model = model[indices_model]
        print(model, int(quantifier[i]))


def load_quants_trials(fname='Archive_unshuffled_sgd', bottleneck=200, num_epochs=4):
    quants = []
    for quant_path in glob(f"../../{fname}/bottleneck-{bottleneck}+*num_epochs-{num_epochs}*/quantifiers.npy"):
        quants.append(np.load(quant_path))
    return np.array(quants)


def get_proportions_of_property(df, property_name, criterion):
    df_property = df.melt(
        id_vars=['num_epochs', 'bottleneck', 'num_trial', 'generation'], 
        value_vars=[v for v in df.columns if property_name in v],
        value_name=property_name
    )
    
    groupby_df = df_property.groupby(['num_epochs', 'bottleneck'])
    return groupby_df[property_name].apply(
        lambda x: criterion(x).sum()/x.size)


def distance_to_ultrafilter(quant, models):
    return np.logical_not(np.column_stack((models, 1-models)) == quant.reshape(-1,1)).sum(axis=0).min()


def add_property(fpath, property_func=distance_to_ultrafilter, propname='ultradist'):
    models = generate_list_models(10)
    for f_name in glob(fpath):

        trial_info = trial_info_from_fname(f_name)
        csv_path = f_name[:-3] + 'csv'

        df = pd.read_csv(csv_path)
    #     print(df)
        frame = pd.DataFrame()
        data = np.load(f_name)
        for generation in range(len(data)):
            gen_data = data[generation]
            gen_row = dict()
            for agt in range(data.shape[-1]):
                gen_quant = np.around(gen_data[:,agt])
    #             print(distance_to_ultrafilter(gen_quant))
                gen_row[propname+'_'+str(agt)] = property_func(gen_quant, models)
            frame = frame.append(gen_row, ignore_index=True)
        new_df = pd.concat([df,frame], axis=1)
        new_df.to_csv(csv_path)


def get_summaries(fn_pattern):
    _data_list = []
    for f_name in glob(fn_pattern):
        trial_info = trial_info_from_fname(f_name)
        parameters, values = zip(*trial_info.items())
        df = pd.read_csv(f_name)
        df[[*parameters]] = pd.DataFrame([[*values]], index=df.index)
        df["path_name"] = f_name
        _data_list.append(df)
    data = pd.concat(_data_list)
    return data


def plot_property(df, ax=None, property_name=None, **kwargs):
    df_property = df.filter(like=property_name, axis=1)
    generations = df_property.index
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(generations, df_property.mean(axis=1), s=0.1, **kwargs)
    return ax


def trial_info_from_fname(fname):
    """
    I rewrote so that it works in windows. Do change it if there are problems!
    directory rather than filename contains the trial parameters
    """
    # trial_root = path.split(fname)[1].split(".")[0]
    trial_root = path.basename(path.dirname(fname))
    kvs = trial_root.split('+')
    trial_info = dict([kv.split('-') for kv in kvs])
    return trial_info


def plot_property_against_property(df, fig, ax, x_property, y_property,
        c="generation", plot_most=True, plot_non_most=True):
    """
    """
    
    data_x = df.filter(like=str(x_property), axis=1).values.flatten()
    data_y = df.filter(like=str(y_property), axis=1).values.flatten()  
    
    # is 'mostlike' is defined consider it, otherwise just include 
    # nothing in 'most'
    most = df.filter(like="mostlike").values.flatten().astype(bool)
    if len(most) == 0:
        most = np.zeros_like(data_x).astype(bool)
    
    if c=="most_like":
                
        if plot_most:
            scatter_most = ax.scatter(
                data_x[most], data_y[most],
                s=0.05, c="blue")
        
        if plot_non_most:
            scatter_other = ax.scatter(
                data_x[np.logical_not(most)], data_y[np.logical_not(most)],
                s=0.05, c="orange")
        
        if plot_most and plot_non_most:
            fig.legend((scatter_most, scatter_other), ("most like", "not most like"))
            
        return
            
    elif c=="generation":
        colors = np.repeat(df.index.values.astype(float), 10) 
    
    else:
        colors = df.filter(like=c, axis=1).values.flatten()    
        
    indices_plot = np.zeros_like(data_x).astype(bool)
    if plot_most:
        indices_plot[most] = True
    if plot_non_most:
        nonmost = np.logical_not(most)
        indices_plot[nonmost] = True

    scatter = ax.scatter(
        data_x[indices_plot],
        data_y[indices_plot],
        s=0.05,
        c=colors[indices_plot],
        cmap='viridis'
    )

    ax_divider = make_axes_locatable(ax)

    ax_right = ax_divider.append_axes(
        'right',
        size='20%',
        pad=0
    )
    
    ax_top = ax_divider.append_axes(
        'top',
        size='20%',
        pad=0
    )
    sns.kdeplot(
        y=data_y[indices_plot],
        ax=ax_right,
        fill=True,
        # color='blue'
    )
    sns.kdeplot(
        x=data_x[indices_plot],
        ax=ax_top,
        fill=True,
        # color='blue'
    )
    for a in [ax_right, ax_top]:
        a.set_xlabel('')
        a.set_ylabel('')
        a.set_xticks([])
        a.set_yticks([])
    return scatter


def parameters_from_name(name):
    return dict([kv.split("-") for kv in path.splitext(path.basename(name))[0].split("+")])


def wrapper_property_against_property(idx, df, fig, ax, axes, x_property, y_property, cbarlabel=None, **kwargs):
    """

    """
    # sct is None if c != "num_trials"
    sct = plot_property_against_property(
        df, fig, ax, x_property, y_property, **kwargs)
    
    if kwargs["c"] != "most_like" and idx==0:
        cbar = plt.colorbar(
            sct,
            ax=axes,
            fraction=0.1,
            aspect=30,
#             pad=0.5,
            shrink=0.7
        )
        label = kwargs['c'] if cbarlabel is None else cbarlabel
#         cbar.ax.tick_params(
#             labelsize='10'
#         )
        cbar.set_label(
            label, 
#             fontsize=10
        )
    
    return (lambda num_epochs: f"Num epochs: {num_epochs}\n{y_property}",
            lambda bottleneck: f"{x_property}\nBtlnk: {bottleneck}")
            
            
def wrapper_single_property(idx, df, fig, ax, axes, **kwargs):
    """
    kwargs need: property_name
    """
    num_trials = np.array([int(trial_info_from_fname(name)["num_trial"]) 
                           for name in df["path_name"].values])

    plot_property(df, ax, c=num_trials, alpha=0.5, cmap="Dark2", **kwargs)
    
    xlabelfunc = lambda num_epochs: (
        'Num epochs: ' + num_epochs +
        '\n' + kwargs['property_name'].capitalize()
    )
    ylabelfunc = lambda bottleneck: "Generation"
    return xlabelfunc, ylabelfunc


def plot_property_across_parameters(df,plot_continuation,xlabel=None,**kwargs):
    """
    """
    sns.set_style('dark')
    # parameters of interests: bottleneck, num_epochs
    df_separated = {
        x: pd.DataFrame(y) 
        for x, y 
        in df.groupby(by=["bottleneck", "num_epochs"])
    }
    
    fig, axes = plt.subplots(
        2, 4, 
        # sharex=True,
        # sharey=True,
        figsize=(7,5)
    )

    bottleneck_dict = {"200": 0, "512": 1, "715": 2, "1024": 3}
    num_epochs_dict = {"4": 0, "8": 1}

    for idx, ((bottleneck, num_epochs), sub_df) in enumerate(
            df_separated.items()):

        xax_index = num_epochs_dict[num_epochs]
        yax_index = bottleneck_dict[bottleneck]
        ax = axes[xax_index,yax_index]

        ylabelfunc, xlabelfunc = plot_continuation(
            idx, 
            sub_df, 
            fig, 
            ax, 
            axes, 
            **kwargs
        )
        
        if num_epochs_dict[num_epochs] == 0:
            # ax.set_title(f"Bttlnk: {bottleneck}", fontsize=12)
            ax.set_xticks([])
        elif num_epochs_dict[num_epochs] == 1:
            if xlabel is None:
                ax.set_xlabel(
                    xlabelfunc(bottleneck), 
#                     fontsize=10
                )
            else:
                ax.set_xlabel(xlabel)

        if bottleneck_dict[bottleneck] == 0:
            ax.set_ylabel(
                ylabelfunc(num_epochs), 
#                 fontsize=10
            )
        else:
            ax.set_yticks([])
        
    ax.set_ylim(0,1)
    fig.subplots_adjust(
#         left=0.2,
        right=1-0.2,
#         bottom=0.2
    )
    
    return fig, axes


def find_selected_quants(bottleneck, num_epochs, burn_in=0):
    fname_pattern = (
        "C:/Users/s1569804/Desktop/Archive_shuffled_adam/" +
        f"bottleneck-{bottleneck}+max_model_size-10+n_agents-10+n_generations-300+num_epochs-{num_epochs}+" +
        "num_trial-*+shuffle_input-True/quantifiers.npy"
    )
    
    quants_list = []
    for fname in glob(fname_pattern):
        quants_list.append(np.load(fname)[burn_in:])
    print(quants_list[0].shape)
    quants = np.concatenate(quants_list)
    rounded_quants = np.round(quants)

    # get the monotonicity and quantity values for the quantifiers I loaded
    # (I am assuming that I got the quantifiers for all num_trials)

    trial_root = path.split(path.split(fname)[-2])[1]
    kvs = trial_root.split('+')
    trial_info = dict([kv.split('-') for kv in kvs])

    qry = ' and '.join(["{} == '{}'".format(k,v) for k,v in trial_info.items() if k != "num_trial"])

    sub_df = df.query(qry)

    mon_values = sub_df.filter(like="monotonicity").values
    quant_values = sub_df.filter(like="quantity").values

    # create a mask for the ones where the two extremes are different (the non-degenerates basically)
    # has true or false, same size as rounded_quants
    most_like = (rounded_quants[:,0,:] != rounded_quants[:,-1,:])

    ### Get the confidence levels by number of 1s!

    # put the dimension that indexes the models first
    # and then get only the ones that are most_like
    selected_quants = np.einsum("ijk->jik", quants)[:,most_like]
    
    # associates for each element of selected_quants the number of 1s for the model corresponding to that element
    num_ones = np.tile(np.sum(models, axis=1, keepdims=True), reps=(selected_quants.shape[1]))
    
    return selected_quants, num_ones
    
    
def plot_confidence_levels(bottleneck, num_epochs, ax):
    
    selected_quants, num_ones = find_selected_quants(bottleneck, num_epochs)
    
    # I'm not interested in the number of 1s, but rather than difference between 
    # the size of the model and the "average size", i.e. 5 1s. 
    num_ones = np.abs(5 - num_ones)
    
    # flatten both of them because the specific model doesn't matter anymore
    selected_quants = selected_quants.flatten()
    num_ones = num_ones.flatten()

    ######### plotting

    cmap = plt.cm.spring
    color_list = cmap(np.linspace(1, 0, 5))
    # loop over differences between 5 and model size
    for i in range(5):
        # where the model 
        argwhere = np.argwhere(num_ones == i)
        confidences = selected_quants[argwhere]
        hist = ax.hist(confidences, bins=100, alpha=0.4, color=color_list[i], density=True)


def get_mean_confidence_by_size(quant, models):
    sizes = models.sum(axis=1)
    indices = np.eye(models.shape[1]+1)[sizes]
    quant_by_size = indices * quant.reshape(-1,1)
    mean_confidence_by_size = quant_by_size.sum(axis=0) / indices.sum(axis=0)
    return mean_confidence_by_size


def find_most_like(quant, models):
    """
    Returns whether a quantifier is most-like. I.e. the expected probability of a model's truth 
    changes monotonically as a function of number of 1s in the model
    """
    mean_confidence_by_size = get_mean_confidence_by_size(quant, models)
    diff_mean_conf_by_size = np.diff(mean_confidence_by_size)
    # check if always increasing or always decreasing
    return np.all(diff_mean_conf_by_size > 0) or np.all(diff_mean_conf_by_size < 0)


def measure_degeneracy(quant, models):
    """
    Returns whether a quantifier is degenerate-like. Simply check in how many places 
    the quantifier is difference from a degenerate quantifier
    """
    quant_map = np.round(quant)
    prop_1s = quant_map.sum() / len(quant_map)
    return max(prop_1s, 1-prop_1s)


################# from file on cogsci paper


def compare_mon_three(models):
    """
    """
    # compare the monotonicity measure for three natural language quantifiers
    
    all_quant = np.all(models, axis=1)

    card = np.sum(models, axis=1)
    less_than_2_more_than_5 = (card <= 2) | (card >= 5)

    even_number = card % 2 == 0

    with np.printoptions(threshold=np.inf):
        print(all_quant)
        print(less_than_2_more_than_5)
        print(even_number)

    print("All: ", tests.measure_monotonicity(models, all_quant))
    print("Less than 2 or more than 5: ", tests.measure_monotonicity(models, less_than_2_more_than_5))
    print("Even: ", tests.measure_monotonicity(models, even_number))


def calculate_df(df=None, fn_pattern=None, property_name='monotonicity'):
    """
    """
    # mean of monotonicity measure by number of epochs and bottleneck size

    assert (df is not None) ^ (fn_pattern is not None), 'Specify one of df or fn_pattern'

    if df is None:
        df = analysis.get_summaries(fn_pattern)

    print(df.columns)
    df = df.filter(regex=f'bottleneck|num_epochs|num_trial|{property_name}|generation').astype({
        "bottleneck": int,
        "num_epochs": int,
        "generation": int
    })

    # unpivots the property columns
    df = pd.melt(
        df,
        id_vars=["num_epochs", "num_trial", "generation", "bottleneck"],
        value_vars=[property_name+'_'+str(i) for i in range(10)],
        value_name=property_name
    )
    return df


def simple_single_plot(df, models):
    """
    """
    # simple single plot (but visually messy)
    
    # rename columns for figure legend
    df.rename(columns={
        "bottleneck": "Bottleneck",
        "num_epochs": "Epochs"
    }, inplace=True)

    ax = sns.lineplot(
        x='generation',
        y='monotonicity',
        hue='Bottleneck',
        style='Epochs',
        data=df,
        # specify as many colors as there are elements of the hue, otherwise it add uninstantiated values in the legend
        # see: https://stackoverflow.com/questions/51525284/the-hue-parameter-in-seaborn-relplot-skips-an-integer-when-given-numerical-d
        palette=["m", "g", "y", "r"]
    )
    
    plt.ylim(0, 1)
    plt.xlabel("Generation")
    plt.ylabel("Monotonicity")
    
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    ax.get_figure().savefig('./evolution_monotonicity_by_training_size.pdf')


def plot_random_quants():
    """
    """
    # Create the random quantifiers matrices for plotting
    # same plot but num of epochs in different plots
    n_quant = 300

    # create a bunch of random network quantifiers
    net_quants = tests.produce_random_quants(10, models, n_quants=n_quant, qtype="network")

    # create a bunch of truly random quantifiers
    rand_quants = tests.produce_random_quants(10, models, n_quants=n_quant, qtype="random")

    # calculate monotonicity of both
    rand_mons = np.apply_along_axis(
        lambda quantifier: tests.measure_monotonicity(
            models,
            quantifier
        ),
        0,
        rand_quants
    )

    net_mons = np.apply_along_axis(
        lambda quantifier: tests.measure_monotonicity(
            models,
            quantifier
        ),
        0,
        net_quants
    )

    sns.kdeplot(rand_mons, clip=(0.0, 1.0), label="Monotonicity for random quantifiers")
    sns.kdeplot(net_mons, clip=(0.0, 1.0), label="Monotonicity for random networks")
    plt.legend()
    #plt.show()
    plt.savefig("random_monotonicities.png",
                bbox_inches='tight',
                transparent=True,
                size=(8, 6), dpi=300)


def plot_by_number_epochs(df, filename='mon_by_trainingsize.png',
        property_name='monotonicity'):
    """
    This produces the plot for the paper
    """
    sns.set(font_scale=1.2)
    g = sns.FacetGrid(
        data=df, 
        col='num_epochs', 
        hue='bottleneck', 
        palette="Blues"
    )
    g = (
        g.map(sns.lineplot, 'generation', property_name)
         .add_legend()
         .set_axis_labels("Generation", property_name.capitalize())
    )
    
    axes = g.axes.flatten()
    axes[0].set_title("Epochs = 4")
    axes[1].set_title("Epochs = 8")
    
    g.savefig(f'./{filename}', dpi=300)


###################### original file


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
    """
    """
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
