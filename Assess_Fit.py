import random
import ast
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import namedtuple
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from scipy.stats import pearsonr
from typing import Tuple, Dict, List, Optional

def _relative_sort(target_list: List, key_list: List) -> List:
    '''
    Sort target_list using the values of key_list to determine order
    
    :param target_list: The list to be sorted
    :param key_list: The list that controls sorting order
    :return: A list containing the values of target_list arranged based on 
    the values the correspond to in key_list
    '''

    zipped_pairs = zip(key_list, target_list)
    z = [x for _, x in sorted(zipped_pairs)]
    return z


def multipage(filename: str,
              figs: Optional[List[plt.Figure]]=None):
    '''
    Save a list of figures to a single pdf file.
    
    :param filename: The name of the output file
    :param figs: A list of pyplot figures. If None, instead save all figures
    currently open.
    '''
    
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def _parse_T50_data(filename: str,
                    use_all: bool = False,
                    observed_only: bool = False,
                    threshold: Optional[float] = None
                    
                    ) -> Tuple[Dict[str,
                                    Tuple[str, int]],
                               Dict[str,
                                    Tuple[str, float]],
                               Dict[str, str]]:
    '''
    Read T50 data from a file into a form usable in learning 
    :param filename: Name of file from which to read
    :param use_all: If True, include observed sequences in unobserved_seqs
    :param observed_only: If True, include only observed sequences in 
    unobserved_seqs
    :param threshold: If provided, unobserved_seqs will contain observed 
    sequences with T50s over the threshold. Use with observed_only to get 
    predictions for only those high-T50 sequences.
    :return: A tuple of three dicts: func_seqs, thermo_seqs, and 
    unexplored_seqs
    '''
    dataFrame = pd.read_csv(filename, dtype={"Seq_ID": str})
    dataFrame.set_index("Seq_ID", inplace=True)
    func_seqs: Dict[str, Tuple[str, int]] = {}
    thermo_seqs: Dict[str, Tuple[str, float]] = {}
    unexplored_seqs: Dict[str, str] = {}

    for seq_ID in dataFrame.index:
        row = dataFrame.loc[seq_ID]
        bin_seq = list([int(res) for res in row["Sequence"]])
        if str(row["T50"]) == "nan":
            if not observed_only:
                # Sequences not yet evaluated go in unexplored
                unexplored_seqs.update({seq_ID: bin_seq})
        elif row["T50"] == "dead":
            # Inactive sequences go in functional as a 0
            func_seqs.update({seq_ID: (bin_seq, 0)})
        elif row["T50"] == "retry":
            if not observed_only:
                # Confusing sequences also go in unexplored
                unexplored_seqs.update({seq_ID: bin_seq})
        else:
            # Live sequences go in functional as a 1 and in thermo_seqs as their T50s
            func_seqs.update({seq_ID: (bin_seq, 1)})
            thermo_seqs.update({seq_ID: (bin_seq, float(row["T50"]))})
            if observed_only:
                if threshold and float(row["T50"]) >= threshold:
                    unexplored_seqs.update({seq_ID: bin_seq})
                elif not threshold:
                    unexplored_seqs.update({seq_ID: bin_seq})
        if use_all:
            unexplored_seqs.update({seq_ID: bin_seq})

    return func_seqs, thermo_seqs, unexplored_seqs  # Return the three dictionaries


def _parse_partial_T50_data(filename: str,
                            scope: int,
                            use_all: bool = False,
                            observed_only: bool = False,
                            threshold: Optional[int] = None
                            ) -> Tuple[Dict[str,
                                            Tuple[str, int]],
                                       Dict[str,
                                            Tuple[str, float]],
                                       Dict[str, str]]:
    '''
    Recreate the state of learning at some past point
    :param filename: Name of file from which to read
    :param scope: Number of learning rounds to include
    :param use_all: If True, observed sequences are included in unobserved_seqs
    :param observed_only: If True, only observed sequences are included in
    unobserved_seqs
    :param threshold: If not None, unobserved_seqs only contains sequences with
    T50 above this value
    :return: Three dictionaries: func_seqs, thermo_seqs, and unexplored_seqs.
    func_seqs has sequence IDs as keys and tuples of format (one-hot encoded
    sequence, [1 for active or 0 for inactive]) as values.
    thermo_seqs has sequence IDs as keys and tuples of format (one-hot encoded
    sequence, T50) as values.
    unexplored_seqs has sequence IDs as keys and one-hot encoded sequences
    as values.
    '''

    cycle = int(filename.split('_')[2][0])
    slice_start = 3*cycle - 3
    slice_end = 3*cycle

    parents = ['1111', '2222', '3333', '4444', '5555', '6666']

    summary = pd.read_csv('Experiment_Summary.csv')
    sequence_lists = [ast.literal_eval(seqs) for seqs in summary['Sequences']]

    learn = [seq_list[slice_start:slice_end] for seq_list in sequence_lists]
    # Concatenate all sequences tested by the agent
    concat = [y for x in learn for y in x]
    tested_sequences = parents + concat[0:3 * scope]

    data = pd.read_csv(filename, index_col='Seq_ID')
    T50s = [data['T50'][seq] for seq in tested_sequences]

    T50s_focused = []
    observed_seqs = []
    for T50, seq in zip(T50s, tested_sequences):
        observed_seqs.append(seq)
        if T50 == 'retry':
            T50s_focused.append('r')
        elif T50 == 'dead':
            if observed_seqs.count(seq) < concat.count(seq):
                T50s_focused.append('r')
            else:
                T50s_focused.append('d')
        elif T50 == '':
            T50s_focused.append(None)
        else:
            if observed_seqs.count(seq) < concat.count(seq):
                T50s_focused.append('r')
            else:
                T50s_focused.append(float(T50))

    observations = {observed_seqs[i]: T50s_focused[i]
                    for i in range(len(observed_seqs))}

    func_seqs: Dict[str, Tuple[str, int]] = {}
    thermo_seqs: Dict[str, Tuple[str, float]] = {}
    unexplored_seqs: Dict[str, str] = {}

    for seq_ID in data.index:
        row = data.loc[seq_ID]
        bin_seq = list([int(res) for res in row["Sequence"]])

        if seq_ID not in observations:
            if not observed_only:
                # Sequences not yet evaluated go in unexplored
                unexplored_seqs.update({seq_ID: bin_seq})
        elif observations[seq_ID] == 'r':
            # Sequences not yet evaluated go in unexplored
            unexplored_seqs.update({seq_ID: bin_seq})
        elif observations[seq_ID] == 'd':
            # Inactive sequences go in functional as a 0
            func_seqs.update({seq_ID: (bin_seq, 0)})
        else:
            # Live sequences go in functional as a 1 and in thermo_seqs as their T50s
            func_seqs.update({seq_ID: (bin_seq, 1)})
            thermo_seqs.update({seq_ID: (bin_seq, observations[seq_ID])})
            if observed_only:
                if threshold and float(row["T50"]) >= threshold:
                    unexplored_seqs.update({seq_ID: bin_seq})
                elif not threshold:
                    unexplored_seqs.update({seq_ID: bin_seq})
        if use_all:
            unexplored_seqs.update({seq_ID: bin_seq})
    return func_seqs, thermo_seqs, unexplored_seqs  # Return the three dictionaries


def _choose_seq(func_seqs, thermo_seqs, unexplored_seqs, kernel):
    # No functional seqs, choose randomly
    if not thermo_seqs:
        chosen_seq_ID = random.choice(list(unexplored_seqs.keys()))
        return chosen_seq_ID, 55.0

    # If there are thermo_seqs, perform regression
    else:
        train_thermo_X, train_thermo_y = zip(*thermo_seqs.values())
        gpr = GPR(kernel=kernel)
        gpr.fit(train_thermo_X, train_thermo_y)

        if len(func_seqs) - len(thermo_seqs) > 0:
            gpc = GPC(kernel=kernel)
            train_func_X, train_func_y = zip(*func_seqs.values())
            gpc.fit(train_func_X, train_func_y)
            probs = {ID: gpc.predict_proba([seq])[0][1] for ID, seq
                     in unexplored_seqs.items()}
        else:
            probs = {ID: 1 for ID in unexplored_seqs}

        y_means = {}
        y_stds = {}
        for ID, seq in unexplored_seqs.items():
            y_mean, y_std = gpr.predict([seq], return_std=True)
            y_means.update({ID: y_mean})
            y_stds.update({ID: y_std})

        min_mean = min(y_means.values())
        y_means_zeroed = {ID: y_mean - min_mean for ID, y_mean
                          in y_means.items()}

        upper_bounds = {}
        for ID in y_means_zeroed:
            u = y_means_zeroed[ID]
            d = y_stds[ID]
            p = probs[ID]
            ub = (u + 2 * d) * p
            upper_bounds.update({ID: ub})

        # Choose seq with largest upper bound
        chosen_seq_ID = max(upper_bounds, key=lambda x: upper_bounds[x])
        pred_T50 = y_means[chosen_seq_ID]

        return chosen_seq_ID, pred_T50, y_means, y_stds, probs


def _get_overlaps(dataset_1, dataset_2, data_type):

    data_1 = []
    seq_IDs_1 = []
    data_2 = []
    seq_IDs_2 = []

    ucbs_1 = [mean + 2 * std - min(dataset_1.means)
              for mean, std in zip(dataset_1.means, dataset_1.stds)]
    ucbs_2 = [mean + 2 * std - min(dataset_2.means)
              for mean, std in zip(dataset_2.means, dataset_2.stds)]
    eucbs_1 = [ucb * prob for ucb, prob in zip(ucbs_1, dataset_1.probs)]
    eucbs_2 = [ucb * prob for ucb, prob in zip(ucbs_2, dataset_2.probs)]

    if data_type == 'eucb':

        for seq_ID, eucb in zip(dataset_1.key, eucbs_1):
            if seq_ID in dataset_2.key:
                data_1.append(eucb)
                seq_IDs_1.append(seq_ID)
        for seq_ID, eucb in zip(dataset_2.key, eucbs_2):
            if seq_ID in dataset_1.key:
                data_2.append(eucb)
                seq_IDs_2.append(seq_ID)
    elif data_type == 'mean':
        for seq_ID, mean in zip(dataset_1.key, dataset_1.means):
            if seq_ID in dataset_2.key:
                data_1.append(mean)
                seq_IDs_1.append(seq_ID)
        for seq_ID, mean in zip(dataset_2.key, dataset_2.means):
            if seq_ID in dataset_1.key:
                data_2.append(mean)
                seq_IDs_2.append(seq_ID)
    elif data_type == 'ucb':
        for seq_ID, ucb in zip(dataset_1.key, ucbs_1):
            if seq_ID in dataset_2.key:
                data_1.append(ucb)
                seq_IDs_1.append(seq_ID)
        for seq_ID, ucb in zip(dataset_2.key, ucbs_2):
            if seq_ID in dataset_1.key:
                data_2.append(ucb)
                seq_IDs_2.append(seq_ID)

    assert (seq_IDs_1 == seq_IDs_2)
    return (data_1, data_2)


learn_state = namedtuple(
    'learn_state', ['filename', 'means', 'stds', 'probs', 'key', 'next_seq'])


def learn(filenames: List[str], white_kernel: float = 1, dot_product: float = 1, scope: Optional[int] = None, predict_all: bool = False, predict_observed: bool = False, threshold=None):
    """
    Performs the experiment a pre-defined number of times with a pre-defined batch size
    :param iterations: Number of cycles of experimentation and learning to run
    :param filename: Name of file housing prior data and where new data will be saved
    :param test: boolean: True if testing, False if real.
    :param batchSize: Number of sequences to test concurrently
    """

    kernel = DotProduct(dot_product) + WhiteKernel(white_kernel)
    all_run_data = []  # Store all sequences to run at once here
    for filename in filenames:
        if scope is None:
            func_seqs, thermo_seqs, unexplored_seqs = _parse_T50_data(
                filename, use_all=predict_all, observed_only=predict_observed, threshold=threshold)
        else:
            func_seqs, thermo_seqs, unexplored_seqs = _parse_partial_T50_data(
                filename, scope, use_all=predict_all, observed_only=predict_observed, threshold=threshold)

        chosen_seq_ID, predT50, y_means, y_stds, probs = _choose_seq(func_seqs,
                                                                     thermo_seqs,
                                                                     unexplored_seqs,
                                                                     kernel)
        # print(chosen_seq_ID)
        # print(predT50)
        means = np.array([y_means[ID].item() for ID in y_means])
        stds = np.array([y_stds[ID].item() for ID in y_means])
        probs = np.array([probs[ID].item() for ID in y_means])
        key = list(y_means.keys())
        all_run_data.append(learn_state(
            filename, means, stds, probs, key, chosen_seq_ID))
    return all_run_data


def mean_std_plot(all_data):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    for i, data in enumerate(all_data):
        curr_axs = axs[(i) // 2, (i) % 2]
        curr_axs.scatter(data.means, data.stds)
        max_pair = [0, 0]
        for mean, std in zip(data.means, data.stds):
            if mean + 2 * std > max_pair[0] + 2 * max_pair[1]:
                max_pair = [mean, std]
        curr_axs.axline(max_pair, slope=-0.5)
        curr_axs.set_title(f'agent {i+1}')
        # curr_axs.text(max_pair[0] + 1, max_pair[1] + 0.2, data.next_seq)

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel('Predicted T50')
    plt.ylabel('Standard Deviation')
    plt.show()


def mean_prob_plot(all_data: List[learn_state],
                   all_data_2: List[learn_state]=None,
                   single_plot: bool=False,
                   save: bool=False,
                   sort_color: bool=False,
                   title: Optional[str]=None):
    '''
    Plot T50 predictions vs probability predictions for each agent in all_data
    
    :param all_data: A list of learn_state objects, one for each agent to plot
    :param all_data_2: If provided, plot these points over those in all_data
    :param single_plot: If True, plot all data on the same axes
    :param save: If True, save file as mean_prob_plot.eps
    :param sort_color: If True, plot points in order of increasing eUCB.
    Otherwise, plot in the order found in all_data.
    :param title: The title to apply to the created plot.
    :return: The pyplot figure generated.
    '''
    
    marker_size = 40

    if single_plot:
        fig, axs = plt.subplots(1, 1)
    else:
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    for i, data in enumerate(all_data):
        if single_plot:
            curr_axs = axs
        else:
            curr_axs = axs[(i) // 2, (i) % 2]
        x = data.means
        y = data.probs
        z = [(mean + 2 * std) for mean, std in zip(x, data.stds)]  # z is UCB
        z = [(ucb - min(z)) * prob for ucb, prob in zip(z, y)]  # z is eUCB
        if sort_color:
            x = _relative_sort(x, z)
            y = _relative_sort(y, z)
            z = sorted(z)
        curr_axs.set_ylim(0.1, 0.9)
        curr_axs.set_ylim(0, 1)
        curr_axs.set_xlim(10, 70)
        curr_axs.scatter(x, y, s=marker_size, c=z,
                         cmap='Wistia', vmin=0, vmax=30)
        curr_axs.set_title(f'agent {i+1}')

    if all_data_2:
        for i, data in enumerate(all_data_2):
            curr_axs = axs[(i) // 2, (i) % 2]
            curr_axs.scatter(data.means, data.probs, s=marker_size)
            curr_axs.set_title(f'agent {i+1}')

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel('Predicted T50')
    plt.ylabel('Probability Active')
    if title:
        plt.title(title)

    if save:
        plt.savefig('mean_prob_plot.eps', format='eps')

def mean_prob_plot_development(folder_name: str,
                               save_filename: str='all_mean_prob_plots.pdf'):
    
    figs = []
    for count in range(21):
        filename = f'{folder_name}/pred_round_{count}.pkl'
        all_data = pickle.load(open(filename, 'rb'))
        figs.append(mean_prob_plot(all_data, save=False, sort_color=True, title=f'round {count}'))
    multipage(f'{folder_name}/{save_filename}', figs)
        

def colorbar(cmap,
             y_min,
             y_max,
             save=False):
    fig = plt.figure()
    mappable = matplotlib.cm.ScalarMappable(cmap=cmap)
    mappable.set_clim(y_min, y_max)
    plt.colorbar(mappable)
    if save: 
        plt.savefig('colorbar.eps', format='eps')
    
    return fig


def eucb_plot(all_data, all_data_2=None, single_plot=False):
    if single_plot:
        fig, axs = plt.subplots(1, 1)
    else:
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    max_eucb = 0

    for i, data in enumerate(all_data):
        if single_plot:
            curr_axs = axs
        else:
            curr_axs = axs[(i) // 2, (i) % 2]
        x = [mean + 2 * std - min(data.means)
             for mean, std in zip(data.means, data.stds)]
        y = data.probs
        z = [x_val * y_val for x_val, y_val in zip(x, y)]
        scat = curr_axs.scatter(x, y, c=z, cmap='Wistia')
        curr_axs.set_title(f'agent {i+1}')
        if max(z) > max_eucb:
            max_eucb = max(z)
        fig.colorbar(scat, ax=curr_axs)

    if all_data_2:
        for i, data in enumerate(all_data_2):
            curr_axs = axs[(i) // 2, (i) % 2]
            x = [mean + 2 * std - min(data.means)
                 for mean, std in zip(data.means, data.stds)]
            y = data.probs
            z = [x_val * y_val for x_val, y_val in zip(x, y)]
            scat = curr_axs.scatter(x, y, c='black')
            curr_axs.set_title(f'agent {i+1}')
            if max(z) > max_eucb:
                max_eucb = max(z)
            # fig.colorbar(scat, ax=curr_axs)

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel('Minimum-Subtracted UCB')
    plt.ylabel('Probability Active')
    
    return fig


def max_means(all_data):
    output = []
    for data in all_data:
        best_key = ''
        best_mean = 0
        for key, mean in zip(data.key, data.means):
            if mean > best_mean:
                best_key = key
                best_mean = mean
        output.append([data.filename, best_key, best_mean])
    return output


def cross_agent_matrix(all_data, all_data_2=None, data_type: str = 'eucb', fg_xy: str = 'both', save=False):

    # mins = {'mean': 10, 'eucb': 0}
    # maxes = {'mean': 70, 'eucb': 35}
    mins = {'mean': 15, 'eucb': 0}
    maxes = {'mean': 65, 'eucb': 35}
    text_places = {'mean': 40, 'eucb': 17.5}

    marker_size = 2
    bg_color = 'gray'
    fg_color = 'red'
    line_color = 'black'

    side_length = len(all_data)

    fig, axs = plt.subplots(side_length, side_length, sharex=True, sharey=True)
    cmap = matplotlib.cm.get_cmap('inferno')

    corr_coefs = np.empty((side_length, side_length))
    learn_numbers = list(range(side_length))
    for i in learn_numbers:
        if i+1 < len(learn_numbers):
            for j in learn_numbers[i+1:]:

                data_i, data_j = _get_overlaps(
                    all_data[i], all_data[j], data_type=data_type)

                # Upper corner: heatmap
                coef, p_val = pearsonr(data_i, data_j)
                corr_coefs[i][j] = coef
                curr_axs = axs[i, j]
                rgba = cmap(coef)
                curr_axs.set_facecolor(rgba[0:3])
                curr_axs.text(text_places[data_type], text_places[data_type],
                              f'R={coef:.3f}', color='white', ha='center', va='center', fontsize=18)

                # Lower corner: scatters
                curr_axs = axs[j, i]
                curr_axs.set_xlim(mins[data_type], maxes[data_type])
                curr_axs.set_ylim(mins[data_type], maxes[data_type])

                curr_axs.scatter(data_j, data_i, s=marker_size,
                                 c=bg_color, alpha=0.5)

                if all_data_2:
                    if fg_xy == 'both' or fg_xy == 'y':
                        # Sequences observed by y-axis agent
                        data_i, data_j = _get_overlaps(
                            all_data_2[i], all_data[j], data_type=data_type)
                        curr_axs.scatter(
                            data_j, data_i, s=marker_size, c=fg_color)
                    if fg_xy == 'both' or fg_xy == 'x':
                        # Sequences observed by x-axis agent
                        data_i, data_j = _get_overlaps(
                            all_data[i], all_data_2[j], data_type=data_type)
                        curr_axs.scatter(
                            data_j, data_i, s=marker_size, c=fg_color)

                    curr_axs.set_xticks([])
                    curr_axs.set_yticks([])

                lin = np.linspace(mins[data_type], maxes[data_type])
                curr_axs.plot(lin, lin, color=line_color)

        curr_axs = axs[i, i]
        # curr_axs.hist(all_data[i].means)

    print(corr_coefs)

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    if save:
        plt.savefig('cross_learn_matrix_after_10_plus.eps', format='eps')
    plt.show()


def full_trace_traceback(filenames, data_type='eucb', predict_all: bool = False, observed_only: bool = False):
    summary = pd.read_csv('Experiment_Summary.csv')
    trace = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    for i in range(summary.shape[0] + 1):
        print(f'round {i}')
        all_data = learn(filenames, scope=i, predict_all=predict_all,
                         predict_observed=observed_only)
        for j, data in enumerate(all_data):
            columns = {key: prob for key, prob in zip(data.key, data.probs)}
            trace[j] = trace[j].append(columns, ignore_index=True)
    with open('probability_trace.pkl', 'wb') as pkl:
        pickle.dump(trace, pkl)
    print('done')


def plot_full_trace(trace, save=False):
    over_10 = [['6151', '6311'],
               ['6251', '6511'],
               ['6311', '6511'],
               ['6111', '6211', '6451', '6511', '6651']]
    for i, df in enumerate(trace):
        plt.figure(i, figsize=(15, 6))
        for head in df.head():
            plt.plot(df.index,
                     df[head],
                     color='gray',
                     alpha=0.1)
        for seq in over_10[i]:
            plt.plot(df.index,
                     df[seq],
                     color='green')
        plt.title(f'agent {i+1}')
        plt.xlabel('round')
        plt.xticks(list(range(0, 22, 2)))
        plt.ylabel('predicted T50 (C)')
        plt.ylim(15, 65)
        plt.xlim(0, 20)
        if save:
            plt.savefig(f'agent {i+1} over 10.eps', format='eps')
    plt.show()


def plot_full_prob_trace(trace, save=False):
    over_10 = [['6151', '6311'],
               ['6251', '6511'],
               ['6311', '6511'],
               ['6111', '6211', '6451', '6511', '6651']]
    for i, df in enumerate(trace):
        plt.figure(i, figsize=(15, 6))
        for head in df.head():
            plt.plot(df.index,
                     df[head],
                     color='gray',
                     alpha=0.1)
        for seq in over_10[i]:
            plt.plot(df.index,
                     df[seq],
                     color='green')
        plt.title(f'agent {i+1}')
        plt.xlabel('round')
        plt.xticks(list(range(0, 22, 2)))
        plt.ylabel('probability active')
        plt.ylim(0, 1)
        plt.xlim(0, 20)
        if save:
            plt.savefig(f'agent {i+1} probs over 10.eps', format='eps')
    plt.show()


def spread_trace(pred_trace, prob_trace, single_plot=False, save=False):

    if single_plot:
        fig, axs = plt.subplots(1, 1)
    else:
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    x_coords = []
    y_coords = []

    for i, pred_df, prob_df in zip(range(len(pred_trace)), pred_trace, prob_trace):

        x_coords.append([pred_df.T[j].std() for j in pred_df.index])
        y_coords.append([prob_df.T[j].std() for j in prob_df.index])

    for i in range(len(x_coords)):
        if single_plot:
            curr_axs = axs
        else:
            curr_axs = axs[(i) // 2, (i) % 2]

        curr_axs.plot(x_coords[i], y_coords[i])
        curr_axs.set_title(f'agent {i+1}')

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.ylabel('std of probability')
    plt.xlabel('std of T50')
    if save:
        plt.savefig('spread_trace.svg', format='svg')
    plt.show()


#===============================================================================
# def corr_coef_traceback(filenames, show_scatters=False, data_type='eucb', predict_all: bool = False, observed_only: bool = False):
#     summary = pd.read_csv('Experiment_Summary.csv')
#     trace = []
#     for i in range(summary.shape[0]+1):
#         print(f'round {i}')
#         all_data = learn(filenames, scope=i, predict_all=predict_all,
#                          predict_observed=observed_only)
#         corr_coefs = np.empty((4, 4))
#         for j in range(4):
#             for k in range(4):
#                 j_data, k_data = _get_overlaps(
#                     all_data[j], all_data[k], data_type=data_type)
#                 coef, p_val = pearsonr(j_data, k_data)
#                 corr_coefs[j][k] = coef
#                 if show_scatters:
#                     plt.figure(j * 10 + k)
#                     plt.scatter(j_data, k_data)
#                     linspace_min = min(min(j_data), min(k_data))
#                     linspace_max = max(max(j_data), max(k_data))
#                     linspace = np.linspace(linspace_min, linspace_max)
#                     plt.plot(linspace, linspace)
#                     if data_type == 'eucb':
#                         plt.title(f'Predicted eUCBs, round {i}')
#                     elif data_type == 'mean':
#                         plt.title(f'Predicted means, round {i}')
#                     elif data_type == 'ucb':
#                         plt.title(f'Predicted ucbs, round {i}')
#                     plt.xlabel(f'Learn {j+1}')
#                     plt.ylabel(f'Learn {k+1}')
#         if show_scatters:
#             plt.show()
#             plt.clf()
#         trace.append(corr_coefs)
#     print(trace)
#     for i, matrix in enumerate(trace):
#         plt.figure()
#         #sns.heatmap(matrix, annot=True, xticklabels=[
#         #            1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
#         plt.title(f'Coefficients round {i+1}')
#     plt.show()
#     return trace
#===============================================================================


def plot_matrix_trace(trace, x_dim=4, y_dim=4, save=False):
    fig_num = 0
    for j in range(x_dim):
        for k in range(y_dim):
            if j <= k:
                continue
            # plt.figure(fig_num)
            fig_num += 1
            pearson = [matrix[j][k] for matrix in trace]
            plt.plot(range(len(pearson)), pearson, label=f"agent_{j+1}x{k+1}")
            
            plt.ylim(-0.4, 1)
            plt.axhline(0, c='xkcd:gray', linestyle='--')
    plt.legend()
    plt.title(f'Pearson Correlations')
    if save:
        plt.savefig('Pearson_Traces_overlaid.eps', format='eps')
    plt.show()


def compare_at(filenames: List[str], scopes: List[int], data_type: str = 'eucb'):
    all_data = [learn([filename], scope=scope, predict_all=True,
                predict_observed=True)[0] for filename, scope in zip(filenames, scopes)]
    print(all_data)
    cross_agent_matrix(all_data, data_type=data_type)
    plt.show()


def all_learn_states(filenames: List[str]):
    for scope in range(21):
        all_data = learn(filenames, scope=scope,
                         predict_all=True, predict_observed=True)
        with open(f'learn_states/pred_round_{scope}.pkl', 'wb') as pkl:
            pickle.dump(all_data, pkl)


def pred_master_list():
    data_dicts = []
    for scope in range(21):
        with open(f'learn_states/pred_round_{scope}.pkl', 'rb') as pkl:
            all_data = pickle.load(pkl)
        for agent_number, learn_state in enumerate(all_data):
            df_line = {'agent': agent_number+1, 'rounds': scope}
            df_line.update({key: mean for key, mean in zip(
                learn_state.key, learn_state.means)})
            data_dicts.append(df_line)
    df = pd.DataFrame(data_dicts)
    df.to_csv('learn_states/all_mean_preds.csv')
    
def random_learn_sim(model, choices_count=60, sim_count=1000, prob_cutoff=0.5):
    all_seqs = model[0].key
    
    T50_ref = {}  # quick reference for T50s when plotting. Dead value is nan
    for i, seq in enumerate(all_seqs):
        T50 = model[0].means[i]
        prob = model[0].probs[i]
        if prob >= prob_cutoff:
            T50_ref[seq] = T50
        else:
            T50_ref[seq] = np.nan
    
    runs = []
    for _ in range(sim_count):
        runs.append([])
        seqs = all_seqs.copy()
        for __ in range(choices_count):
            choice = random.choice(seqs)
            seqs.remove(choice)
            runs[-1].append(choice)
            
    iterations = list(range(1, choices_count+1))
    
    plt.figure()
    
    T50_traces = []
    for run in runs:
        T50_traces.append([])
        for iteration in iterations:
            observed = run[0:iteration]
            T50s = [T50_ref[seq] for seq in observed]
            T50_traces[-1].append(max(T50s))
        
        plt.plot(iterations, T50_traces[-1], color='gray', alpha=0.02)   
        
    trace_array = np.array(T50_traces)
    trace_array = trace_array.T
    T50_traces = trace_array.tolist()
    trace_averages_mean = [np.nanmean(trace) for trace in T50_traces]
    trace_averages_median = [np.nanmedian(trace) for trace in T50_traces]
    plt.plot(iterations, trace_averages_mean, linewidth=2, label='mean')
    plt.plot(iterations, trace_averages_median, linewidth=2, label='median')
    plt.xlabel('sequences tested')
    plt.ylabel('max T50 observed')
    plt.legend()
    plt.title(f'simulated random selection, n={sim_count}, prob cutoff={prob_cutoff}')
    
    plt.show()
            
    


if __name__ == "__main__":
    with open('all_data_variants.p', 'rb') as pkl:
        all_data_full, all_data_unobserved, all_data_observed = pickle.load(pkl)
    with open('all_data_variants_unified.pkl', 'rb') as pkl:
        all_data_unified, all_data_full_unified, all_data_observed_unified = pickle.load(
            pkl)
    with open('pearson_trace.pkl', 'rb') as pkl:
        trace = pickle.load(pkl)
    with open('prediction_trace.pkl', 'rb') as pkl:
        pred_trace = pickle.load(pkl)
    with open('probability_trace.pkl', 'rb') as pkl:
        prob_trace = pickle.load(pkl)

    # spread_trace(pred_trace, prob_trace)
    # plot_full_trace(pred_trace)
    # plot_full_prob_trace(prob_trace)
    # plot_matrix_trace(trace)
    
    # compare_at(['Seq_Data_1.csv', 'Seq_Data_2.csv', 'Seq_Data_3.csv',
    #           'Seq_Data_4.csv'], scopes=[17, 12, 7, 11], data_type='mean')
    
    #all_learn_states(['Seq_Data_1.csv', 'Seq_Data_2.csv',
    #                'Seq_Data_3.csv', 'Seq_Data_4.csv'])
   
    mean_prob_plot_development('learn_states', 'all_rounds.pdf')
    
    # pred_master_list()
    
    # random_learn_sim(all_data_full_unified)
    
    #mean_prob_plot(all_data_full, single_plot=False, sort_color=True, save=True)
    
    # colorbar('Wistia', 0, 30)
    # all_data_full.append(all_data_full_unified[0])
    # all_data_observed.append(all_data_observed_unified[0])
    # cross_learn_matrix(all_data_full, all_data_2=all_data_observed, data_type='mean')
    # full_trace_traceback(['Seq_Data_1.csv', 'Seq_Data_2.csv', 'Seq_Data_3.csv', 'Seq_Data_4.csv'], data_type='mean', predict_all=True)
    # all_data_good = learn(['Seq_Data_1.csv', 'Seq_Data_2.csv', 'Seq_Data_3.csv', 'Seq_Data_4.csv'], predict_observed=True, threshold=50)
    
    # mean_prob_plot(all_data_full)
    # mean_prob_plot(all_data_unobserved, all_data_observed)
    # mean_prob_plot(all_data_unobserved, all_data_good)
    # mean_prob_plot(all_data_full)  # , all_data_observed)
    
    # trace = corr_coef_traceback(['Seq_Data_1.csv', 'Seq_Data_2.csv', 'Seq_Data_3.csv', 'Seq_Data_4.csv'], show_scatters=False, data_type='mean', predict_all=True, observed_only=False)
    # plot_matrix_trace(trace)
    #plt.show()
