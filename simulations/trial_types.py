""" Holds functions that implement various bandit solution strategies.

Call get_trial_func() with the desired strategy name and parameters to get a
function that can be called with datapoints as the only argument.
general_trial implements the process of selecting a sequence on the given round
using a abstract stratgy over multiple iterations.

All strategies take as arguments a List of unchosen DPs, a List of chosen_DPs,
and any parameters needed to perform selection. All strategies return the
chosen sequence and a List of sequences that were deemed valid to choose from.

Available strategies:
    "random_selection": Selects a random sequence.
    "ucb": Calculates the upper confidence bound of each unchosen sequence with
        the equation mean + 2*std where mean and std come from gaussian process
        regression on the active chosen sequences. Requires a sklearn kernel
        for the GPR.
    "ucb_inactive_0C": Same as "ucb" but inactive chosen sequences are
        considered to have a T50 of 0 for the GPR.
    "ucb_inactive_0C_mean_active": Same as "ucb_inactive_0C" but means are only
        calculated from GPR on the active sequences.
    "expected_ucb": Same as "ucb" but also does GPC on sequence functionality
        using the chosen seqs as training data, the lowest ucb score is
        subtracted from all ucb scores, and the predicted probabilities and
        ucb for each sequence are multiplied to calculate a new ucb.
    "pp_ucb": Same as "ucb" except a GPC is done to predict the functionality
        probability of each unchosen sequence, then ucb scores are only
        calculated for sequences that are predicted to be positive.
"""

from inspect import getfullargspec
from functools import partial
import random

import matplotlib.pyplot as plt
import numpy as np

import tools


def get_trial_func(strat_name, **strat_params):
    """ Factory function that returns the function corresponding to
    *func_name*.
    """
    try:
        strat_func = globals()[strat_name]
    except KeyError:
        raise ValueError(f'Trial type {strat_name} not implemented.') from None

    reduced_strat_func = partial(strat_func, **strat_params)
    rsf_params = getfullargspec(reduced_strat_func)[0]
    if rsf_params != ['hidden_seqs', 'chosen_seqs']:
        raise ValueError(f'{strat_name} params {rsf_params[2:]} must be'
                         'specified as keys.')

    trial_func = partial(general_trial, choice_strat=reduced_strat_func)
    return trial_func



def get_trial_func_batch(strat_name, **strat_params):
    """ Factory function that returns the function corresponding to
    *func_name*.
    """
    try:
        strat_func = globals()[strat_name]
    except KeyError:
        raise ValueError(f'Trial type {strat_name} not implemented.') from None

    reduced_strat_func = partial(strat_func, **strat_params)
    rsf_params = getfullargspec(reduced_strat_func)[0]
    if rsf_params != ['hidden_seqs', 'chosen_seqs']:
        raise ValueError(f'{strat_name} params {rsf_params[2:]} must be'
                         'specified as keys.')

    trial_func = partial(general_trial_batch, choice_strat=reduced_strat_func)
    return trial_func


def random_selection(hidden_seqs, chosen_seqs):
    chosen_seq_ind = random.randrange(len(hidden_seqs))
    possible_seqs = hidden_seqs
    return chosen_seq_ind, possible_seqs


# batch strategies include batch size as last parameter and output a list of sequences (rather than indices)
def random_selection_batch(hidden_seqs, chosen_seqs, batch):
    new_seqs = [random.choice(hidden_seqs) for i in range(batch)]
    return new_seqs


def ucb(hidden_seqs, chosen_seqs, kernel):
    thermo_seqs = [(cs.seq, cs.T50) for cs in chosen_seqs
                   if not np.isnan(cs.T50)]
    possible_seqs = hidden_seqs
    if not thermo_seqs:
        chosen_seq_ind = random.randrange(len(hidden_seqs))
    else:
        r_mean, r_std = tools.gp_reg(thermo_seqs, possible_seqs, kernel)
        for ps, u, d in zip(possible_seqs, r_mean, r_std):
            ps.ucb = u + 2*d
        chosen_seq = max(possible_seqs, key=lambda x: x.ucb)
        chosen_seq_ind = hidden_seqs.index(chosen_seq)
    return chosen_seq_ind, possible_seqs


def ucb_batch(hidden_seqs, chosen_seqs, kernel, batch):
    thermo_seqs = [(cs.seq, cs.T50) for cs in chosen_seqs if not np.isnan(cs.T50)]
    possible_seqs = hidden_seqs
    if not thermo_seqs:
        new_seqs = [random.choice(hidden_seqs) for i in range(batch)]

    else:
        new_seqs = []
        while len(new_seqs) < batch: 
            r_mean, r_std = tools.gp_reg(thermo_seqs, possible_seqs, kernel)
            for ps, u, d in zip(possible_seqs, r_mean, r_std):
                ps.ucb = u + 2*d
                ps.pred_T50 = u

            next_seq = max(possible_seqs, key=lambda x: x.ucb)
            new_seqs.append(next_seq)
            possible_seqs = [s for s in possible_seqs if s not in new_seqs]
            thermo_seqs.append((next_seq.seq,next_seq.pred_T50)) # add predicted T50 to data set

    return new_seqs # batch just returns a list of sequences 


def ucb_inactive_0C(hidden_seqs, chosen_seqs, kernel):
    thermo_seqs = [(cs.seq, cs.T50) if not np.isnan(cs.T50) else (cs.seq, 0)
                   for cs in chosen_seqs]
    possible_seqs = hidden_seqs
    if not thermo_seqs:
        chosen_seq_ind = random.randrange(len(hidden_seqs))
    else:
        r_mean, r_std = tools.gp_reg(thermo_seqs, possible_seqs, kernel)
        for ps, u, d in zip(possible_seqs, r_mean, r_std):
            ps.ucb = u + 2*d
        chosen_seq = max(possible_seqs, key=lambda x: x.ucb)
        chosen_seq_ind = hidden_seqs.index(chosen_seq)
    return chosen_seq_ind, possible_seqs


def ucb_inactive_0C_mean_active(hidden_seqs, chosen_seqs, kernel):
    thermo_seqs = [(cs.seq, cs.T50) for cs in chosen_seqs
                   if not np.isnan(cs.T50)]
    thermo_seqs_all = [(cs.seq, cs.T50) if not np.isnan(cs.T50)
                       else (cs.seq, 0) for cs in chosen_seqs]
    possible_seqs = hidden_seqs
    if not thermo_seqs:
        chosen_seq_ind = random.randrange(len(hidden_seqs))
    else:
        r_mean, r_std = tools.gp_reg(thermo_seqs, possible_seqs, kernel)
        r_mean_all, r_std_all = tools.gp_reg(thermo_seqs_all, possible_seqs,
                                             kernel)
        for ps, u, d in zip(possible_seqs, r_mean, r_std_all):
            ps.ucb = u + 2*d
        chosen_seq = max(possible_seqs, key=lambda x: x.ucb)
        chosen_seq_ind = hidden_seqs.index(chosen_seq)
    return chosen_seq_ind, possible_seqs


def expected_ucb(hidden_seqs, chosen_seqs, kernel):
    thermo_seqs = [(cs.seq, cs.T50) for cs in chosen_seqs if not np.isnan(cs.T50)]
    possible_seqs = hidden_seqs
    if not thermo_seqs:
        chosen_seq_ind = random.randrange(len(possible_seqs))
    else:
        func_seqs = [(cs.seq, 0) if np.isnan(cs.T50) else (cs.seq, 1)
                     for cs in chosen_seqs]
        if len(thermo_seqs) - len(func_seqs) == 0:
            for ps in possible_seqs:
                ps.prob = 1
        else:
            test_probs = tools.gp_class(func_seqs, possible_seqs, kernel)
            for ps, p in zip(possible_seqs, test_probs):
                ps.prob = p
        r_mean, r_std = tools.gp_reg(thermo_seqs, possible_seqs, kernel)
        for ps, u, d in zip(possible_seqs, r_mean, r_std):
            ps.ucb = u + 2*d
        min_ucb = min(possible_seqs, key=lambda x: x.ucb).ucb
        for ps in possible_seqs:
            ps.ucb = (ps.ucb - min_ucb) * ps.prob
        chosen_seq = max(possible_seqs, key=lambda x: x.ucb)
        chosen_seq_ind = possible_seqs.index(chosen_seq)
    return chosen_seq_ind, possible_seqs


def expected_ucb_batch(hidden_seqs, chosen_seqs, kernel,batch):
    thermo_seqs = [(cs.seq, cs.T50) for cs in chosen_seqs if not np.isnan(cs.T50)]
    possible_seqs = hidden_seqs
    if not thermo_seqs:
        new_seqs = [random.choice(hidden_seqs) for i in range(batch)]
    else:
        new_seqs = []
        while len(new_seqs) < batch: 
            func_seqs = [(cs.seq, 0) if np.isnan(cs.T50) else (cs.seq, 1) for cs in chosen_seqs]
            if len(thermo_seqs) - len(func_seqs) == 0:
                for ps in possible_seqs:
                    ps.prob = 1
            else:
                test_probs = tools.gp_class(func_seqs, possible_seqs, kernel)
                for ps, p in zip(possible_seqs, test_probs):
                    ps.prob = p
            r_mean, r_std = tools.gp_reg(thermo_seqs, possible_seqs, kernel)
            for ps, u, d in zip(possible_seqs, r_mean, r_std):
                ps.ucb = u + 2*d
                ps.pred_T50 = u

            min_ucb = min(possible_seqs, key=lambda x: x.ucb).ucb
            for ps in possible_seqs:
                ps.ucb = (ps.ucb - min_ucb) * ps.prob

            next_seq = max(possible_seqs, key=lambda x: x.ucb)
            new_seqs.append(next_seq)
            possible_seqs = [s for s in possible_seqs if s not in new_seqs]
            thermo_seqs.append((next_seq.seq,next_seq.pred_T50)) # add predicted T50 to data set

    return new_seqs # batch just returns a list of sequences 


def pp_ucb(hidden_seqs, chosen_seqs, kernel):
    thermo_seqs = [(cs.seq, cs.T50) for cs in chosen_seqs
                   if not np.isnan(cs.T50)]
    if not thermo_seqs:
        chosen_seq_ind = random.randrange(len(hidden_seqs))
        possible_seqs = hidden_seqs
    else:
        func_seqs = [(cs.seq, 0) if np.isnan(cs.T50) else (cs.seq, 1)
                     for cs in chosen_seqs]
        if len(thermo_seqs) - len(func_seqs) == 0:
            possible_seqs = hidden_seqs
        else:
            test_probs = tools.gp_class(func_seqs, hidden_seqs, kernel)
            for hs, p in zip(hidden_seqs, test_probs):
                hs.prob = p
            possible_seqs = [dp for dp in hidden_seqs if dp.prob >= 0.5]
            if not possible_seqs:
                possible_seqs = hidden_seqs
        r_mean, r_std = tools.gp_reg(thermo_seqs, possible_seqs, kernel)
        for ps, u, d in zip(possible_seqs, r_mean, r_std):
            ps.ucb = u + 2*d
        chosen_seq = max(possible_seqs, key=lambda x: x.ucb)
        chosen_seq_ind = hidden_seqs.index(chosen_seq)
    return chosen_seq_ind, possible_seqs


def pp_ucb_batch(hidden_seqs, chosen_seqs, kernel,batch):
    thermo_seqs = [(cs.seq, cs.T50) for cs in chosen_seqs if not np.isnan(cs.T50)]
    if not thermo_seqs:
        new_seqs = [random.choice(hidden_seqs) for i in range(batch)]
    else:
        new_seqs = []
        while len(new_seqs) < batch:
            func_seqs = [(cs.seq, 0) if np.isnan(cs.T50) else (cs.seq, 1) for cs in chosen_seqs]
            if len(thermo_seqs) - len(func_seqs) == 0:
                possible_seqs = hidden_seqs
            else:
                test_probs = tools.gp_class(func_seqs, hidden_seqs, kernel)
                for hs, p in zip(hidden_seqs, test_probs):
                    hs.prob = p
                possible_seqs = [dp for dp in hidden_seqs if dp.prob >= 0.5]
                if not possible_seqs:
                    possible_seqs = hidden_seqs
            r_mean, r_std = tools.gp_reg(thermo_seqs, possible_seqs, kernel)
            for ps, u, d in zip(possible_seqs, r_mean, r_std):
                ps.ucb = u + 2*d
                ps.pred_T50 = u

            next_seq = max(possible_seqs, key=lambda x: x.ucb)
            new_seqs.append(next_seq)
            hidden_seqs = [s for s in hidden_seqs if s not in new_seqs]
            thermo_seqs.append((next_seq.seq,next_seq.pred_T50)) # add predicted T50 to data set

    return new_seqs # batch just returns a list of sequences 


def ts_inactive_0C(hidden_seqs, chosen_seqs, kernel):
    thermo_seqs = [(cs.seq, cs.T50) if not np.isnan(cs.T50) else (cs.seq, 0)
                   for cs in chosen_seqs]
    possible_seqs = hidden_seqs
    if not thermo_seqs:
        chosen_seq_ind = random.randrange(len(hidden_seqs))
    else:
        r_mean, r_std = tools.gp_reg(thermo_seqs, possible_seqs, kernel)
        r_samples = np.random.normal(r_mean, r_std)
        print(r_samples)
        chosen_seq_ind = np.argmax(r_samples)
    return chosen_seq_ind, possible_seqs


def general_trial(datapoints, choice_strat, stop_at_max=True,max_seqs_explored=None, video=False, **video_kwargs):
    """Perform a learning trial.
    
    Args:
        datapoints: List of DPs in dataset.
        choice_strat: function of how to choose the next sequence to test given
            the chosen and unchosen sequences.
        stop_at max: If the selection process should stop when the maximum T50
            sequence is found.
        max_seqs_explored: Int of how many sequences to explore before
            stopping.
        video: If a trial learning video should be made.
        **video_kwards: video_frame() arguments. 

    Returns:
        List of DPs in the order they were chosen.
    """

    hidden_seqs = datapoints[:]
    chosen_seqs = []
    curr_max_t50 = float('NaN')
    actual_max_t50 = max([dp.T50 for dp in datapoints])

    if video:
        video_fig = plt.figure(figsize=(10, 7))
        plt.xlabel('sequences')
        plt.ylabel('T50')
        artists = []

    while hidden_seqs:
        if max_seqs_explored and len(chosen_seqs) > max_seqs_explored:
            break
        if stop_at_max and curr_max_t50 == actual_max_t50:
            break

        chosen_seq_ind, possible_seqs = choice_strat(tuple(hidden_seqs),tuple(chosen_seqs))

        chosen_seq = hidden_seqs.pop(chosen_seq_ind)
        chosen_seq.explored = True
        chosen_seqs.append(chosen_seq)

        if video:
            art = tools.video_frame(datapoints, possible_seqs, chosen_seq,
                                    video_fig)
            artists.append(art)

        if not np.isnan(chosen_seq.T50):
            if chosen_seq.T50 > curr_max_t50 or np.isnan(curr_max_t50):
                curr_max_t50 = chosen_seq.T50

        print(len(chosen_seqs), curr_max_t50, chosen_seq.T50)

    if video:
        art = tools.video_frame(datapoints, possible_seqs, chosen_seq,
                                video_fig, final=True)
        artists.append(art)
        ani = tools.make_video(video_fig, artists, **video_kwargs)
        return chosen_seqs, ani

    return chosen_seqs


def general_trial_batch(datapoints, choice_strat, stop_at_max=False,max_seqs_explored=None, video=False, **video_kwargs):
    """Perform a learning trial.
    
    Args:
        datapoints: List of DPs in dataset.
        choice_strat: function of how to choose the next sequence to test given
            the chosen and unchosen sequences.
        stop_at max: If the selection process should stop when the maximum T50
            sequence is found.
        max_seqs_explored: Int of how many sequences to explore before
            stopping.
        video: If a trial learning video should be made.
        **video_kwards: video_frame() arguments. 

    Returns:
        List of DPs in the order they were chosen.
    """

    hidden_seqs = datapoints[:]
    chosen_seqs = []
    curr_max_t50 = float('NaN')
    actual_max_t50 = max([dp.T50 for dp in datapoints])

    while len(chosen_seqs) < max_seqs_explored:

        new_seqs = choice_strat(tuple(hidden_seqs),tuple(chosen_seqs))

        hidden_seqs = [s for s in hidden_seqs if s not in new_seqs]
        for seq in new_seqs:
            seq.explored = True 
        chosen_seqs.extend(new_seqs)

        curr_max_t50 = np.nanmax([s.T50 for s in chosen_seqs])

        if curr_max_t50==actual_max_t50:
            return chosen_seqs

    return chosen_seqs




class DP(object):
    """Represents each datapoint in the dataset.

    pred_T50, std, ucb, and prob may be meaningless depending on the context
    of the current trial iteration.

    Public attributes:
        seq: List of 1s and 0s denoting an amino acid sequnce. (read-only)
        T50: Float of the T50 associated with seq. Will be float('NaN') for
            inactive seqs. (read-only)
        index: datapoints List index of the DP. (read-only)
        explored: Whether the DP has been chosen in the current trial.
        pred_T50: Predicted T50 of the DP in the current trial iteration.
        std: Standard error of the T50 prediction.
        ucb: Upper confidence bound for the unchosen_T50 in the current trial
            iteration.
        prob: Functionality probability predicted for the DP in the current
            trial iteration.
    """
    def __init__(self, seq, T50, index):
        self._seq = seq
        self._T50 = T50
        self._dps_index = index
        self.explored = False
        self.pred_T50 = 50
        self.std = 15
        self.ucb = self.pred_T50 + 2*self.std
        self.prob = 1

    @property
    def seq(self):
        return self._seq

    @property
    def T50(self):
        return self._T50

    @property
    def dps_index(self):
        return self._dps_index
