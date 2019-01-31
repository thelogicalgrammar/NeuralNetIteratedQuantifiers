import argparse
import numpy as np
import pandas as pd
import glob
from plotnine import *
import tests
import utilities
from os import path


def trial_info_from_fname(fname):
    # I rewrote so that it works in windows. Do change it if there are problems!
    # directory rather than filename contains the trial parameters
    trial_root = path.split(path.split(path.normpath(fname))[0])[1]
    kvs = trial_root.split('+')
    trial_info = dict([kv.split('-') for kv in kvs])
    return trial_info


def gather_columns(data, prefix):
    return pd.DataFrame(
        pd.concat([data[col] for col in data.columns
                   if col.startswith(prefix)]),
        columns=[prefix])


def get_summaries(fn_pattern):
    _data_list = []
    for f_name in glob.glob(fn_pattern):
        trial_info = trial_info_from_fname(f_name)
        parameters, values = zip(*trial_info.items())
        df = pd.read_csv(f_name)
        df[[*parameters]] = pd.DataFrame([[*values]], index=df.index)
        df["path_name"] = f_name
        _data_list.append(df)
    data = pd.concat(_data_list)
    return data


def summarize_summaries(fn_pattern):
    """
    This function goes through the .csv files containing the results of each trial
    and prints various general info about all the dataframes
    :param fn_pattern: Get a pattern for the .csv files containing the summary of the trials
    :return: None
    """
    # allows to print full paths
    pd.options.display.max_colwidth = 1000

    data = get_summaries(fn_pattern)

    monotone_non_ultrafilters_indices = check_all_monotonicity_ultrafilter(data)

    n_mon = np.sum(data.filter(like="monotonicity_") == 1.).sum()
    print("\nNumber of monotone: \n", n_mon)
    print("Number of monotone non-ultrafilters: \n", len(monotone_non_ultrafilters_indices))
    print("Proportion of non-ultrafilter out of monotone: \n", len(monotone_non_ultrafilters_indices) / n_mon)

    full_output = True

    # this information takes too much space if there are a lot of non-ultrafilter monotone quantifiers
    if full_output:
        print("Information about fully monotone quantifiers that are not ultrafilters:")
        print("Mon non ultrafitlters indices\n", monotone_non_ultrafilters_indices)
        # # for printing full generation
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(data.iloc[list(zip(monotone_non_ultrafilters_indices))[0]])

    # get the actual non-degenerate non-ultrafilter but monotone quantifiers
    for row in monotone_non_ultrafilters_indices:
        # the regex matches any column name containing the agent index, as well as path_name and generation
        agent_data = data.iloc[row[0]].filter(regex=str(row[1])+"|path_name|generation")
        agent_data["index"] = row[1]
        # print what is needed to get the quantifier
        print("np.around(np.load(\"{}\")[{}, :, {}])".format(
            agent_data["path_name"], int(agent_data["generation"]), agent_data["index"]
        ).replace("\\", "/").replace(".csv", ".npy")
        )

    print("\nProportions of ultrafilters based on each object")
    objects, counts = np.unique(data.filter(like="ultrafilter_").values, return_counts=True)
    proportions = counts / np.sum(counts)
    print(pd.Series(proportions, index=objects))
    # print("Parameters that developed any perfectly monotone quantifiers")
    # print("(Generation is first generation where monotonicity appears)")
    # indices = np.any(data.filter(like="monotonicity_").values == 1., axis=1)
    # print(data.loc[indices, ["generation", "path_name"]].drop_duplicates(subset="path_name").sort_values("path_name"))

    degenerate_df = data.filter(like="degenerate_").values == 1.
    indices = np.any(degenerate_df, axis=1)
    # print("\nParameters that developed any degenerate quantifier")
    # print("(Generation is first generation where degeneracy appears)")
    # print(data.loc[indices, ["generation", "path_name"]].drop_duplicates(subset="path_name").sort_values("path_name"))
    print("Proportion of monotone: ", degenerate_df.sum() / degenerate_df.size)

    print("\nAverage movement speed by bottleneck size/number of epochs")
    # first groupby by both bottleneck and # epochs, then take the mean of the movement speed column
    groupby_training_size = data[["num_epochs", "bottleneck", "inter_generational_movement_speed"]].groupby(["num_epochs", "bottleneck"])
    print("How training size affects movement speed: ")
    print(groupby_training_size.agg({
        "inter_generational_movement_speed": ["mean", "std"]
    }))

    # TODO: add to this function


def check_all_monotonicity_ultrafilter(analysis):
    """
    Checks that all evolved quantifiers that are perfectly monotone are ultrafilters
    :param analysis: a dataframe with the analysis
    :return: Indices with inconsistencies (possibly empty)
    """
    # relies on the fact that the columns for monotonicity and ultrafilter check are in the same order
    monotonicity = analysis.filter(like="monotonicity_") == 1.
    ultrafilter = analysis.filter(like="ultrafilter_") > -1
    # check that the perfectly monotone quantifiers are the ultrafilters
    identical_matrix = monotonicity.values == ultrafilter.values
    # find the indices where the monotone quantifiers are not ultrafilters (i.e. where two matrices are different)
    indices = np.array(np.argwhere(np.logical_not(identical_matrix)))
    return indices


def analyze_trials(file_pattern, first_n=10, last_n=50):
    # TODO: get trial_info from dir names here as well?
    data = pd.DataFrame()
    fnames = glob.glob(file_pattern)
    for idx in range(len(fnames)):
        trial = pd.read_csv(fnames[idx])
        trial['num_trial'] = idx
        data = data.append(trial, ignore_index=True)
    data['num_trial'] = data['num_trial'].astype('category')

    data['mean_monotonicity'] = data[
        [col for col in data.columns
         if col.startswith('monotonicity')]].mean(axis=1)

    print(ggplot(data)
          + geom_line(aes(x='generation', y='mean_monotonicity',
                          group='num_trial', colour='num_trial'))
          + xlim((0, 100)))

    first_data = data[data['generation'] < first_n]
    last_data = data[data['generation'] > data['generation'].max() - last_n]

    first_monotonicities = gather_columns(first_data, 'monotonicity')
    first_monotonicities['time'] = 'first_' + str(first_n)

    last_monotonicities = gather_columns(last_data, 'monotonicity')
    last_monotonicities['time'] = 'last_' + str(last_n)

    monotonicities = pd.concat([first_monotonicities, last_monotonicities],
                               ignore_index=True)

    print(ggplot(monotonicities) +
          geom_density(aes(x='monotonicity', colour='time', fill='time'),
                       alpha=0.2))

    first_quantities = gather_columns(first_data, 'quantity')
    first_quantities['time'] = 'first_' + str(first_n)

    last_quantities = gather_columns(last_data, 'quantity')
    last_quantities['time'] = 'last_' + str(last_n)

    quantities = pd.concat([first_quantities, last_quantities],
                               ignore_index=True)

    print(ggplot(quantities) +
          geom_density(aes(x='quantity', colour='time')))


def summarize_trial(trial_info, data, parents):
    """Converts the output of one trial of iteration into a Pandas DataFrame,
    recording various metrics at each generation.

    Args:
        trial_info: dict, containing the args passed to iteration.iterate
        data: 3-D numpy array.  Dim 0: generations, Dim 1: model, Dim 2: agents
        parents: 2-D numpy array.  Dim 0: generations, Dim 1: idx of parent

    Returns:
        a pd.DataFrame
    """
    frame = pd.DataFrame()
    models = utilities.generate_list_models(int(trial_info['max_model_size']))
    for generation in range(len(data)):
        gen_data = data[generation, :, :]
        gen_row = {'generation': generation,
                   'num_trial': int(trial_info['num_trial'])}
        for agt in range(data.shape[-1]):
            # TODO: parameterize the per-agent methods to measure?
            # TODO: vectorize monotonicity etc so they apply to entire
            # generation, instead of this loop?
            gen_agt_map = np.around(gen_data[:, agt])
            gen_row['monotonicity_' + str(agt)] = tests.measure_monotonicity(
                models, gen_agt_map)
            gen_row['quantity_' + str(agt)] = tests.check_quantity(
                models, gen_agt_map)
            gen_row['ultrafilter_' + str(agt)] = tests.check_quantifier_ultrafilter(
                models, gen_agt_map
            )
            gen_row['degenerate_' + str(agt)] = np.all(gen_agt_map) or not np.any(gen_agt_map)
        frame = frame.append(gen_row, ignore_index=True)
    frame['inter_generational_movement_speed'] = (
        tests.inter_generational_movement_speed(data, parents))
    return frame


def batch_convert_to_csv(fn_pattern):
    """Converts a batch of .npy files, containing the output of one trial, to
    .csv files, with the summary of the trial from generations_to_table
    recorded.

    The new files will have the same base as the old files, but a new
    extension, namely .csv.

    Args:
        fn_patten: a pattern for matching a bunch of filenames
    """
    for fname in glob.glob(fn_pattern):
        data = np.load(fname)
        # NB: parents and trial_info assumes our naming convention from iteration.py,
        # so is not generic
        # in particular, the files are named path/to/trial_info_dir/quantifiers.ext and
        # path/to/trial_info_dir/parents.ext
        parents = np.load(fname.replace('quantifiers', 'parents')).astype(int)
        trial_info = trial_info_from_fname(fname)
        table = summarize_trial(trial_info, data, parents)
        old_ext_len = len(fname.split('.')[-1])
        table.to_csv(fname[:-(old_ext_len+1)] + '.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                        choices=['convert', 'analyze', 'summarize'],
                        default='convert')
    parser.add_argument('--file_pattern', type=str, default='./*/*/quantifiers.npy')
    args = parser.parse_args()

    if args.mode == 'convert':
        batch_convert_to_csv(args.file_pattern)
    elif args.mode == 'analyze':
        # TODO: args to analyze_trials?
        analyze_trials(args.file_pattern)
    elif args.mode == 'summarize':
        summarize_summaries(args.file_pattern)

