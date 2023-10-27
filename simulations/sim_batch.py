""" Script to run bandit simulations on chimeric protein engineering datasets.
"""

import load_data
import pickle
from sklearn.gaussian_process import kernels
import trial_types
import tools


dataset = 'P450reduced'
DATA, ENCODING = load_data.get_dataset(dataset)

def series(num_trials, strategy, max_seqs, **strat_params):
    """ Runs *num_trials* trials with procedure *form*.
    """

    batch = strat_params['batch']
    trial = trial_types.get_trial_func_batch(strategy, **strat_params)
    css = []
    print(strategy, flush=True)
    for i in range(num_trials):
        print(i, flush=True)
        chosen_seqs = trial(DATA, max_seqs_explored=max_seqs)

        css.append(chosen_seqs)

    pickle.dump(css, open(f'20221130_{dataset}_run/{strategy}_batch{batch}.p', 'wb'))



num = 10 # number of replicate runs
linear_kernel = kernels.DotProduct(1) + kernels.WhiteKernel(1)

for bs in range(1,21):
    series(num, 'random_selection_batch',1000, batch=bs)
    series(num, 'ucb_batch', 1000, kernel=linear_kernel,batch=bs)
    series(num, 'pp_ucb_batch', 1000, kernel=linear_kernel,batch=bs)
    series(num, 'expected_ucb_batch', 1000, kernel=linear_kernel,batch=bs)

