import argparse
import numpy as np
import pandas as pd
import glob
import tests
import utilities


def summarize_trial(trial_info, data):
    """Converts the output of one trial of iteration into a Pandas DataFrame,
    recording various metrics at each generation.

    Args:
        trial_info: dict, containing the args passed to iteration.iterate
        data: 3-D numpy array.  Dim 0: generations, Dim 1: model, Dim 2: agents

    Returns:
        a pd.DataFrame
    """
    frame = pd.DataFrame()
    models = utilities.generate_list_models(int(trial_info['length_inputs']))
    for generation in range(len(data)):
        gen_data = data[generation, :, :]
        gen_row = {}
        for agt in range(data.shape[-1]):
            # TODO: parameterize the per-agent methods to measure?
            # TODO: vectorize monotonicity etc so they apply to entire
            # generation, instead of this loop?
            gen_agt_map = np.around(gen_data[:, agt])
            gen_row['monotonicity_' + str(agt)] = tests.measure_monotonicity(
                models, gen_agt_map)
            gen_row['quantity_' + str(agt)] = tests.check_quantity(
                models, gen_agt_map)
        frame = frame.append(gen_row, ignore_index=True)
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
        # NB: this trial_info assumes our naming convention from iteration.py,
        # so is not generic
        kvs = fname[:-4].split('+')
        trial_info = dict([kv.split('-') for kv in kvs])
        table = summarize_trial(trial_info, data)
        old_ext_len = len(fname.split('.')[-1])
        table.to_csv(fname[:-(old_ext_len+1)] + '.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['convert', 'analyze'],
                        default='convert')
    parser.add_argument('--file_pattern', type=str, default='*.npy')
    args = parser.parse_args()

    if args.mode == 'convert':
        batch_convert_to_csv(args.file_pattern)
