# The emergence of monotone quantifiers via iterated learning

This project explores the evolution of the universal of monotonicity in the semantics of quantifiers with the help of a computational model. 
It combines the iterated learning paradigm with neural networks as a model of learning. 
This repository contains everything that is needed to run the model as well as tools to analyse the results.

## Getting Started

### Running the model 

The core function for running the model is `iterate` in *iteration.py*. 
The `iterate` function accepts the following parameters (See paper for more details on the meaning of each parameter):

- `num_trial`: The index of the model run when running the model multiple times.
- `bottleneck`: The number of observations used to train each neural network learner.
- `save_path`: Where to save the model results.
- `n_generations`: For how many generations of iterated learning to run.
- `n_agents`: How many agents in each generation.
- `max_model_size`: The number of elements in the restrictor set.
- `num_epochs`: How many epochs to train the neural network learners for.
- `shuffle_input`: Whether to shuffle the input of the neural networks.

Example of a run of the model:

```
python ../iteration.py --num_trial 1 --bottleneck 512 --save_path path/to/folder/ --n_generations 300 --n_agents 10 --max_model_size 10 --num_epochs 4 --shuffle_input True
```

`iterate` stores the results in an .npy file containing an array with shape (# generations, # objects in domain, # agents).

### Analyzing the results

The core function for analyzing the results of a single model is `summarize_trial` in *analysis.py*. 
See function docstring for more details on the parameters.

Example of analysis:
```python
data = np.load('path/to/results.npy')
# NB: parents and trial_info assumes that the files are named path/to/trial_info_dir/quantifiers.ext and path/to/trial_info_dir/parents.ext
parents = np.load('path/to/parents.npy').astype(int)
trial_info = trial_info_from_fname('path/to/results.npy')
table = summarize_trial(trial_info, data, parents)
```

When running the model repeatedly or with multiple values, it might become impractical to analyze each result file individually by hand.
`batch_convert_to_csv` saves the summaries of each result file matching a file pattern.

Example of analysis of multiple files:
```
python analysis.py --file_pattern pattern/to/results*.npy
```

Finally, when running the model multiple times for each of multiple combinations of parameter values, 
it is convenient to summarize the summaries of each combination of parameters that were obtained with `summarize_trial` or `bath_convert_to_csv`.
`summarize_summaries` helps by grouping summary files by combinations of parameter values and printing summaries for each.

Example of analysis of multiple runs of multiple combinations of parameter values:
```
python analysis.py --mode summarize --file_pattern pattern/matching/summaries*.csv
```

### Plotting the results

*CogSci_paper.py* contains snippets of code needed to reproduce the plots shown in the CogSci paper.
Other plotting functions can be found in plotting.py.

### Data

The original data plotted in the CogSci paper is available at https://osf.io/ume39/?view_only=5af8aa196e484001ae1758cc5bcff5a4.

## Authors

* **Shane Steinert-Threlkeld** - *Co-first author*
* **Fausto Carcassi** - *Co-first author*
* **Jakub Szymanik** - *Second author*
