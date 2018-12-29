import argparse
import numpy as np
import pandas as pd
import glob
from plotnine import *
import tests
import utilities


def gather_columns(data, prefix):
    return pd.DataFrame(
        pd.concat([data[col] for col in data.columns
                   if col.startswith(prefix)]),
        columns=[prefix])


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
        trial_root = fname.split('/')[-2]  # -1 is filename, -2 is directory we want
        kvs = trial_root.split('+')
        trial_info = dict([kv.split('-') for kv in kvs])
        table = summarize_trial(trial_info, data, parents)
        old_ext_len = len(fname.split('.')[-1])
        table.to_csv(fname[:-(old_ext_len+1)] + '.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['convert', 'analyze'],
                        default='convert')
    # parser.add_argument('--file_pattern', type=str, default='*/quantifiers.npy')
    parser.add_argument('--file_pattern', type=str, default="./quantifiers.npy")
    args = parser.parse_args()

    if args.mode == 'convert':
        batch_convert_to_csv(args.file_pattern)
    elif args.mode == 'analyze':
        # TODO: args to analyze_trials?
        analyze_trials(args.file_pattern)
