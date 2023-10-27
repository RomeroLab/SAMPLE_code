import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from collections import namedtuple
import pickle as pkl
from typing import List, Tuple



learn_state = namedtuple('learn_state', ['filename', 'means', 'stds', 'probs', 'key', 'next_seq'])

def unified_model(all_data_full) -> Tuple[List, pd.DataFrame]:
    # load final "unified" model
    T50,CI,prob,name = all_data_full[0][1:5]
    T50 = dict((name[i],T50[i]) for i in range(len(name)))
    prob = dict((name[i],prob[i]) for i in range(len(name)))
    all_seqs = T50.keys()
    y_hat = [T50[s] for s in all_seqs]
    unified_T50_df = pd.DataFrame([(s,T50[s]) for s in all_seqs]).set_index(0)
    return (y_hat, unified_T50_df)


## FIG 3e ## ## FIG 3e ## ## FIG 3e ## ## FIG 3e ## ## FIG 3e ## ## FIG 3e ## ## FIG 3e ## ## FIG 3e ## ## FIG 3e ## 


def pearson_vs_background(T50_df: pd.DataFrame,
                          unified_T50: List[float],
                          save: bool=False):
    '''
    Plot correlation coefficients at each round between all agents
    and the combined model's predictions.
    
    :param T50_df: A dataframe of T50 predictions for each sequence by agent
    and round.
    :param unified_T50: A list of all T50s predicted by the unified model
    in order based on sequence ID.
    :return: A pyplot figure
    '''
    
    all_seqs = T50_df.columns.tolist()[2:]
    correlation_coefficients = []
    for agent in range(1,5):
        t = T50_df[T50_df.agent==agent][all_seqs].T
        t = t.rename(columns = dict((c,i) for i,c in enumerate(t.columns)))
        d = [np.corrcoef(t[cycle],unified_T50)[0,1] for cycle in range(21)]
        correlation_coefficients.append(d)
        
    fig = plt.figure()
    plt.plot(range(21),np.array(correlation_coefficients).T)
    plt.xticks(list(range(0,21,2)))
    plt.axis([0,20,-0.02,0.95])
    plt.xlabel('Learning round')
    plt.ylabel('correlation coefficient')
    plt.title('Pearson correlation coefficients vs final unified model')
    plt.legend(['agent 1','agent 2','agent 3','agent 4'])
    if save:
        plt.savefig('pearson_with_unified.eps')


## FIG 3f ## ## FIG 3f ## ## FIG 3f ## ## FIG 3f ## ## FIG 3f ## ## FIG 3f ## ## FIG 3f ## ## FIG 3f ## ## FIG 3f ## 

def landscape_confidence(std_df: pd.DataFrame,
                         prob_df: pd.DataFrame,
                         unified_T50_df: pd.DataFrame,
                         save: bool=False):


    all_seqs = std_df.columns.tolist()[2:]
    
    # scan a range of temperature windows and find all sequences that fall within range 
    temps = np.linspace(unified_T50_df.min(), unified_T50_df.max(), 100).T[0]
    window = 5
    seqs_in_range = []
    for temp in temps:
        seqs_in_range.append(list(unified_T50_df[(unified_T50_df[1] > (temp - window))
                                      & (unified_T50_df[1] < (temp + window))].index))
    
    # find confidence within window 
    fig = plt.figure()
    cycle = 20
    for agent in range(1,5):
        stds = std_df[prob_df.agent == agent][all_seqs].T
        stds = stds.rename(columns=dict((c, i) for i, c in enumerate(stds.columns)))
        mean = np.array([np.mean(stds[cycle][i]) for i in seqs_in_range])
        stds = np.array([np.std(stds[cycle][i]) for i in seqs_in_range])
        plt.plot(temps, mean)
        # plt.fill_between(temps,mean-stds,mean+stds,alpha=0.2)
    
    plt.axis([22,60,2,11.7])
    plt.xlabel('T50 prediction (C)')
    plt.ylabel('mean uncertainty (C)')
    plt.title('Distribution of T50 uncertainty by agent')
    plt.legend(['agent 1','agent 2','agent 3','agent 4'])
    if save:
        plt.savefig('landscape_confidence.eps')


## FIG 3g ## ## FIG 3g ## ## FIG 3g ## ## FIG 3g ## ## FIG 3g ## ## FIG 3g ## ## FIG 3g ## ## FIG 3g ## ## FIG 3g ## 

def pairwise_pearson(T50_df,
                     save=False):

    all_seqs = T50_df.columns.tolist()[2:]

    # pearson corrlation across agent pairs 
    fig = plt.figure()
    correlation_coefficients = []
    for agent1 in range(1,5):
        for agent2 in range(1,5):
            if agent1<agent2:
                t1 = T50_df[T50_df.agent==agent1][all_seqs].T
                t1 = t1.rename(columns=dict((c, i)
                                              for i, c in enumerate(t1.columns)))
    
                t2 = T50_df[T50_df.agent==agent2][all_seqs].T
                t2 = t2.rename(columns=dict((c, i)
                                              for i, c in enumerate(t2.columns)))
    
                d = [np.corrcoef(t1[cycle], t2[cycle])[0, 1]
                     for cycle in range(21)]
                correlation_coefficients.append(d)
    
    
    plt.plot(range(21),np.array(correlation_coefficients).T)
    plt.xticks(list(range(0,21,2)))
    plt.axis([0,20,-.4,1])
    plt.xlabel('Learning round')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('Pairwise Pearson Correlation between all agents')
    plt.legend(['A%i-A%i'%(a1,a2) for a1 in range(1,5)
                for a2 in range(1,5) if a1<a2])
    if save:
        plt.savefig('correlation_across_agents.eps')


## SUPP FIG ?? ## ## SUPP FIG ?? ## ## SUPP FIG ?? ## ## SUPP FIG ?? ## ## SUPP FIG ?? ## ## SUPP FIG ?? ## ## SUPP FIG ?? ## 

def choice_percentiles(T50_df: pd.DataFrame,
                       std_df: pd.DataFrame,
                       prob_df: pd.DataFrame,
                       save: bool=False):

    all_seqs = T50_df.columns.tolist()[2:]

    # load in trajectory of chosen sequences 
    summary = pd.read_csv('Experiment_Summary.csv')
    chosen_seqs = {}
    for i in  [1, 2, 3, 4]:
        slice_start = 3 * i - 3
        slice_end = 3 * i
        sequence_lists = [ast.literal_eval(seqs) for seqs in summary['Sequences']]
        learn = [seq_list[slice_start:slice_end] for seq_list in sequence_lists]
        chosen_seqs[i] = learn
    
    
    fig, axs = plt.subplots(1,4,sharey=True,sharex=True, figsize=(20, 5))
    
    for agent in range(1,5):
        T50 = T50_df[T50_df.agent == agent][all_seqs].T
        std = std_df[std_df.agent == agent][all_seqs].T
        prob = prob_df[prob_df.agent == agent][all_seqs].T
    
        # various quantities of interest 
        T50 = T50.rename(columns=dict((c, i) for i, c in enumerate(T50.columns)))
        std = std.rename(columns=dict((c, i) for i, c in enumerate(std.columns)))
        prob = prob.rename(columns=dict((c, i) for i, c in enumerate(prob.columns)))
        ucb = T50 + 2 * std  # UCB
        eucb = prob * ((T50 - T50.min(0)) + 2 * std)  # eUCB
    
        # percentile ranks
        T50_rank = T50[list(range(21))].rank(pct=True)
        std_rank = std[list(range(21))].rank(pct=True)
        prob_rank = prob[list(range(21))].rank(pct=True)
        ucb_rank = ucb[list(range(21))].rank(pct=True)
        eucb_rank = eucb[list(range(21))].rank(pct=True)
    
        # deltas
        delta_T50 = T50 - T50.quantile(0.5)
        delta_std = std - std.quantile(0.5)
    
        # zscores
        z_T50 = (T50 - T50.mean()) / T50.std()
        z_std = (std - std.mean()) / std.std()
    
        # use the first seq of batch
        #t_traj = [z_T50.loc[chosen_seqs[agent][cycle][0]][cycle] for cycle in range(20)]
        #s_traj = [z_std.loc[chosen_seqs[agent][cycle][0]][cycle] for cycle in range(20)]
        # p_traj = [prob_rank.loc[chosen_seqs[agent][cycle][0]][cycle] for cycle in range(20)]
        #e_traj = [er.loc[chosen_seqs[agent][cycle][0]][cycle] for cycle in range(20)]
        
        # take mean across batch of three
        t_traj = [np.mean([T50_rank.loc[chosen_seqs[agent][cycle][rep]][cycle] for rep in range(3)]) for cycle in range(20)]
        s_traj = [np.mean([std_rank.loc[chosen_seqs[agent][cycle][rep]][cycle] for rep in range(3)]) for cycle in range(20)]
        p_traj = [np.mean([prob_rank.loc[chosen_seqs[agent][cycle][rep]][cycle] for rep in range(3)]) for cycle in range(20)]
    
        axs[agent-1].plot(t_traj)
        axs[agent-1].plot(s_traj)
        axs[agent-1].plot(p_traj)
        axs[agent-1].legend(['mean','sd','prob'])
    
        plt.xticks(list(range(0,21,2)))
        plt.axis([0,20,0,1])
        
    fig.add_subplot(111, frame_on=False)
    plt.xlabel('round')
    plt.ylabel('ranking of choice')
    plt.title('Percentile rank of each variable for chosen sequences')
    
    if save:    
        plt.savefig('agent_decision_making.eps')

if __name__ == "__main__":
    # load predictions/stds/p(active) for all agents and cycles
    T50_df = pd.read_csv('all_mean_preds.csv',index_col='index')
    std_df = pd.read_csv('all_std_preds.csv',index_col='index')
    prob_df = pd.read_csv('all_prob_preds.csv',index_col='index')

    unobserved,full,observed = pkl.load(open('all_data_variants_unified.pkl', 'rb'))
    y_hat, unified_T50_df = unified_model(full)
    pearson_vs_background(T50_df, unified_T50 = y_hat, save=False)
    landscape_confidence(std_df, prob_df, unified_T50_df, save=False)
    pairwise_pearson(T50_df, save=False)
    choice_percentiles(T50_df, std_df, prob_df, save=False)
    plt.show()