""" Tools for protein engineering bandit simulations.


Available functions:
    video_frame: Make a single frame of a learning video for a trial.
    make_video: Finalizes video creation and shows it. (Must be called after
        all video_frame calls and before any other plotting function).
    progress_graph: Plots graph of a trial's chosen and maximum T50 over time.
    get_filled_T50s: Converts saved simultion data into uniform T50 data.
    itrs_to_temp_boxplot: Makes a boxplot of the number of iterations to reach
        a given temp.
    gp_class: Convenience wrapper for Gaussian Process Classification.
    gp_reg: Convenience for Gaussian Process Regression.
"""

from matplotlib import animation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, \
                                        GaussianProcessClassifier


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


def video_frame(datapoints, possible_seqs, chosen_seq, fig, final=False):
    """Make a frame of a trial learning video.

    Trial learning videos show the real T50s of each datapoint, the previously
    selected sequences, and the UCB scores of the possible sequences.
    video_frame() should be called on every iteration of a trial. Set final to
    true on the final frame and then call make_video(). No plotting should be
    done between the first video_frame() call and make_video().

    Args:
        datapoints: List of DP objects. The ordering should stay constant
            between calls.
        possible_seqs: List of seqs for which ucb should be plotted.
        chosen_seq: DP object called on current round.
        fig: plt Figure object where video is plotted. This must be the same
            object on every video_frame call.
        final: Whether the current call is the final video_frame() call. Adds
            a legend to the video.

    Returns:
        List of Lists of artists, which are the plt objects drawn. Needed for
            make_video() call.
    """
    plt.ylim(20, max([ps.ucb for ps in possible_seqs]) + 3)
    plt.ylim(20, 90)
    explored = [(dp.dps_index, dp.T50) for dp in datapoints if dp.explored]
    explored_pos = [x for x in explored if not np.isnan(x[1])]
    explored_neg = [(i, 21) for i, T50 in explored if np.isnan(T50)]
    unexplored = [(dp.dps_index, dp.T50) for dp in datapoints
                  if not dp.explored]
    unexplored_pos = [x for x in unexplored if not np.isnan(x[1])]
    unexplored_neg = [(i, 21) for i, T50 in unexplored if np.isnan(T50)]
    ucb = [(dp.dps_index, dp.ucb) for dp in possible_seqs]
    cs = (chosen_seq.dps_index, chosen_seq.ucb)
    l0, = plt.plot(*zip(*unexplored_pos), 'ro')
    l1, = plt.plot(*zip(*unexplored_neg), 'rx')
    l2, = plt.plot(*zip(*ucb), 'go')
    l3, = plt.plot(*zip(*explored_pos), 'bo')
    l4, = plt.plot(*zip(*explored_neg), 'bx')
    l5, = plt.plot(*cs, 'yo')
    if explored_pos:
        max_seq = max([x[1] for x in explored_pos])
        l6 = plt.axhline(max_seq, 0, len(datapoints))
        artists = [l0, l1, l2, l3, l4, l5, l6]
    else:
        artists = [l0, l1, l2, l3, l4, l5]
    if final:
        l0.set_label('unexplored positive')
        l1.set_label('unexplored negative')
        l2.set_label('predicted positive ucb')
        l3.set_label('explored positive')
        l4.set_label('explored negative')
        l5.set_label('current selected ucb')
        plt.legend()
    return artists


def make_video(fig, artists, show_video=False, save_video_fn=None,
               **ani_kwargs):
    """Makes the trial learning video.

    Args:
        fig: plt Figure object used for video. Must be same fig passed into
            video_frame() calls.
        artists: List of Lists of plt artists. Returned by video_frame().
        show_video: Whether plt.show() is called to display video.
        save_video_fn: Filename of saved video. Video will not be saved if
            None.
        **ani_kwargs: List of keywords to pass into ArtistAnimation()
            initialization. Look at ArtistAnimation documentation for a full
            list.
    """
    ani = animation.ArtistAnimation(fig, artists, **ani_kwargs)
    if save_video_fn:
        ani.save(save_video_fn)
    if show_video:
        plt.show()
    return ani


def progress_graph(chosen_seqs):
    """Make a graph that shows the progress of a trial over time.

    The progress graph is a plot of the selected T50 and maximum T50 at each
    iteration.

    Args:
        chosen_seqs: List of DP objects in order of selection by trial.
    """
    max_T50s = []
    curr_T50s = []
    for cs in chosen_seqs:
        if np.isnan(cs.T50):
            cs_T50 = 0
        else:
            cs_T50 = cs.T50
        curr_T50s.append(cs_T50)
        if not max_T50s or cs.T50 > max_T50s[-1]:
            max_T50s.append(cs_T50)
        else:
            max_T50s.append(max_T50s[-1])
    plt.figure()
    plt.plot(max_T50s, 'b-o')
    curr_pos = [(i, c) for i, c in enumerate(curr_T50s) if c]
    curr_neg = [(i, 35) for i, c in enumerate(curr_T50s) if not c]
    plt.plot(*zip(*curr_pos), 'ro')
    plt.plot(*zip(*curr_neg), 'rx')
    plt.show(block=True)


def get_filled_T50s(run_data, length):
    """Gets T50s for data generated by simulations.

    Args:
        run_data: List of Lists of Lists of DP objects representing chosen
            sequences of each trial (3rd list layer) of each strategy (2nd list
            layer) of a given run (outer list).
        length: Float specifying how far to extend all trial lengths with
            repetition of the last chosen sequence.

    Returns:
        List of Lists of Lists of floats that is the same as run_data but each
            trial length in a strategy is a uniform length and thermostability
            replaces the DP object.
    """
    run_T50s = []
    for strat_data in run_data:
        strat_T50s = []
        max_num_itrs = max(length, max([len(td) for td in strat_data]))
        for trial_data in strat_data:
            trial_T50s = [itr_cs.T50 for itr_cs in trial_data]
            for _ in range(max_num_itrs - len(trial_data)):
                trial_T50s.append(trial_T50s[-1])
            strat_T50s.append(trial_T50s)
        run_T50s.append(strat_T50s)
    return run_T50s


def _get_max_T50s(trial_T50s):
    """Helper function that gets the maximum T50 at each iteration of a trial.

    Args:
        trial_T50s: List of selected T50s at each iteration of a trial.

    Returns:
        Numpy array of maximum T50 selected at any round before or during the
        given iteration.
    """
    curr_max_t50 = float('NaN')
    max_T50s = []
    for curr_T50 in trial_T50s:
        if not np.isnan(curr_T50):
            if curr_T50 > curr_max_t50 or np.isnan(curr_max_t50):
                curr_max_t50 = curr_T50
        max_T50s.append(curr_max_t50)
    return np.array(max_T50s)


def learning_curve(run_T50s, strategies, save_fn=None, show=False, ylines=[], xlims=[]):
    """Plots curves of average maximum T50 data against number of iterations.

    Args:
        run_T50s: List of Lists (strategies) of Lists (trials) of floats which
            are the T50s of the sequences chosen at each iteration. Trial Lists
            must be a uniform length within a given strategy List.
        strategies: List of Strings for strategies names corresponding to the
            strategy Lists in run_T50s.
        save_fn: Filename (without extension) to save plot. Plot will be saved
            as both png and eps. None means saving will not occur.
        show: Whether plt.show() is called at the end of function.

    Returns:
        Figure that was created.
    """
    fig = plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('maximum T50')
    for strat_name, strat_T50s in zip(strategies, run_T50s):
        strat_max_T50s = [_get_max_T50s(trial_data) for trial_data 
                          in strat_T50s]
        untrimmed_step_data = np.array(strat_max_T50s).T
        step_data = tuple(x[~np.isnan(x)] for x in untrimmed_step_data)
        strat_means = [np.mean(x) for x in step_data]
        plt.plot(strat_means, label=strat_name)
    plt.legend()
    # for y in ylines:
    #    plt.plot(xlims,[y,y])

    plt.xlim(xlims)

    if save_fn:
        plt.savefig(save_fn + '.png')
        plt.savefig(save_fn + '.eps')
    if show:
        plt.show()
    return fig


def _get_itrs_to_temp(T50s, temp):
    """Helper function that returns the number of iterations needed to find a
    temperature above temp.

    Args:
        T50s: List of T50s selected at each iteration.
        temp: Float temperature threshold to search for.
    """
    for itr, curr_T50 in enumerate(T50s):
        if not np.isnan(curr_T50) and curr_T50 >= temp:
            return itr
    return len(T50s)


def itrs_to_temp_boxplot(run_T50s, strategies, temp_thresh, save_fn=None,
                          show=False):
    """Makes a boxplot of the number of iterations needed to reach the given
    temperature.

    Args:
        run_T50s: List of Lists (strategies) of Lists (trials) of floats which
            are the T50s of the sequences chosen at each iteration. Trial Lists
            must be a uniform length within a given strategy List.
        strategies: List of Strings for strategies names corresponding to the
            strategy Lists in run_T50s.
        temp_thresh: Temperature threshold for number of iterations to be
            calculated.
        save_fn: Filename (without extension) to save plot. Plot will be saved
            as both png and eps. None means saving will not occur.
        show: Whether plt.show() is called at the end of function.

    Returns:
        Figure that was created.
    """
    fig = plt.figure()
    plt.ylabel(f'trials needed to reach {temp_thresh}')
    itrs_to_thresh = []
    for strat_T50s in run_T50s:
        strat_itrs = []
        for trial_T50s in strat_T50s:
            trial_itrs = _get_itrs_to_temp(trial_T50s, temp_thresh)
            strat_itrs.append(trial_itrs)
        itrs_to_thresh.append(strat_itrs)
    plt.boxplot(itrs_to_thresh, labels=strategies)
    #plt.violinplot(itrs_to_thresh)
    plt.yscale('log')
    plt.ylim([0.8, 1000])

    
    if save_fn:
        plt.savefig(save_fn + '.png')
        plt.savefig(save_fn + '.eps')
    if show:
        plt.show()
    return fig


def itrs_to_temp_hist(run_T50s, strategies, temp_thresh, save_fn=None,show=False):
    """Makes a boxplot of the number of iterations needed to reach the given
    temperature.

    Args:
        run_T50s: List of Lists (strategies) of Lists (trials) of floats which
            are the T50s of the sequences chosen at each iteration. Trial Lists
            must be a uniform length within a given strategy List.
        strategies: List of Strings for strategies names corresponding to the
            strategy Lists in run_T50s.
        temp_thresh: Temperature threshold for number of iterations to be
            calculated.
        save_fn: Filename (without extension) to save plot. Plot will be saved
            as both png and eps. None means saving will not occur.
        show: Whether plt.show() is called at the end of function.

    Returns:
        Figure that was created.
    """
    
    fig = plt.figure()
    plt.ylabel(f'trials needed to reach {temp_thresh}')
    itrs_to_thresh = []
    for strat_T50s in run_T50s:
        strat_itrs = []
        for trial_T50s in strat_T50s:
            trial_itrs = _get_itrs_to_temp(trial_T50s, temp_thresh)
            strat_itrs.append(trial_itrs)
        itrs_to_thresh.append(strat_itrs)

    itrs_to_thresh = [[j+1 for j in i] for i in itrs_to_thresh] # min should be 1 not zero


    sns.kdeplot(itrs_to_thresh[0],log_scale=True, bw_adjust=1.5)    
    sns.kdeplot(itrs_to_thresh[1],log_scale=True, bw_adjust=1.5)    
    sns.kdeplot(itrs_to_thresh[2],log_scale=True, bw_adjust=1.5)    
    sns.kdeplot(itrs_to_thresh[3],log_scale=True, bw_adjust=1.5)    
    plt.legend(['random_selection', 'ucb', 'pp_ucb', 'expected_ucb'])

    plt.xlim([0.9, 1000])

    
    # med = [np.median(s) for s in itrs_to_thresh]

    # plt.plot([med[0],med[0]],[0,2])
    # plt.plot([med[1],med[1]],[0,2])
    # plt.plot([med[2],med[2]],[0,2])
    # plt.plot([med[3],med[3]],[0,2])

    if save_fn:
        plt.savefig(save_fn + '.png')
        plt.savefig(save_fn + '.eps')
    if show:
        plt.show()
    return fig



def gp_class(train_data, test_dps, kernel):
    """Wrapper for Gaussian Process Classification.

    Args:
        train_data: List of (binary_sequence, binary_functionality) for each
            sequence previously selected.
        test_dps: List of DP object to predict functionality on.
        kernel: sklearn.gaussian_process kernel to use in the GPC. See
            sklearn documentation for full list.

    Returns:
        List of functionality probabilities for each DP in test_dps.
    """

    if len(set([d[1] for d in train_data]))==2: # two classes
        gpc = GaussianProcessClassifier(kernel=kernel)
        gpc.fit(*zip(*train_data))
        test_x = [dp.seq for dp in test_dps]
        y_prob = gpc.predict_proba(test_x)
        prob_func = [p[1] for p in y_prob]
    
    else:
        prob_func = [1 for p in test_dps] # one class: just assign p=1 to all sequences 

    return prob_func


def gp_reg(train_data, test_dps, kernel):
    """Wrapper for Gaussian Process Regression.

    Args:
        train_data: List of (binary_sequence, binary_functionality) for each
            sequence previously selected.
        test_dps: List of DP object to predict thermostability on.
        kernel: sklearn.gaussian_process kernel to use in the GPC. See
            sklearn documentation for full list.

    Returns:
        y_mean: List of predicted T50s for each DP in test_dps.
        y_std: List of standard deviations of T50 prediction for each DP in
            test_DPS.
    """
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(*zip(*train_data))
    test_x = [dp.seq for dp in test_dps]
    y_mean, y_std = gpr.predict(test_x, return_std=True)
    return y_mean, y_std
