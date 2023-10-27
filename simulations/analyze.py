"""Script for analyzing run data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import load_data
import tools
from tools import DP
from typing import List

run_date = '20221130'
dataset = 'P450reduced'
strategies = ['random_selection', 'ucb', 'pp_ucb', 'expected_ucb']

def get_simulation_data(run_date: str='20221130',
                        dataset: str='P450reduced',
                        strategies: List[str] = ['random_selection',
                                                 'ucb',
                                                 'pp_ucb',
                                                 'expected_ucb']):
    run_datas = []
    for strat in strategies:
        strat_data = pickle.load(open(f'{run_date}_{dataset}_run/{strat}.p','rb'))
        run_datas.append(strat_data)
    
    run_T50s = tools.get_filled_T50s(run_datas,50)
    return run_T50s

def learning_curve(run_T50s,
                   run_date,
                   dataset,
                   strategies):
    
    T50s = np.array([float(s[1]) for s in 
                     load_data.read_data(dataset+'_thermostability.data')[1]])
    
    temp_thresh = (max(T50s)-min(T50s))*0.9 + min(T50s) # 90% of the max
    save_fn=f'Images/{run_date}_{dataset}_lc'
    
    fig = tools.learning_curve(run_T50s,
                               strategies,
                               save_fn=save_fn,
                               ylines=[temp_thresh,max(T50s)],
                               xlims=[-1,60])
    
    plt.title('Mean observed maximum across 10000 trials')
    
def iteration_boxplot(run_T50s,
                      run_date,
                      dataset,
                      strategies):
    
    T50s = np.array([float(s[1]) for s in 
                     load_data.read_data(dataset+'_thermostability.data')[1]])
    
    temp_thresh = (max(T50s)-min(T50s))*0.9 + min(T50s) # 90% of the max
    save_fn=f'Images/{run_date}_{dataset}_bw'
    
    fig = tools.itrs_to_temp_boxplot(run_T50s,
                                     strategies,
                                     temp_thresh,
                                     save_fn=save_fn)
    
    plt.title('Iterations needed to reach 90th percentile')
    
def iteration_histogram(run_T50s,
                        run_date,
                        dataset,
                        strategies):
    
    T50s = np.array([float(s[1]) for s in 
                     load_data.read_data(dataset+'_thermostability.data')[1]])
    
    temp_thresh = (max(T50s)-min(T50s))*0.9 + min(T50s) # 90% of the max
    save_fn=f'Images/{run_date}_{dataset}_hist'
    
    fig = tools.itrs_to_temp_hist(run_T50s,
                                  strategies,
                                  temp_thresh,
                                  save_fn=save_fn)
    
    plt.title('Iterations needed to reach 90th percentile')
    

if __name__ == "__main__":
    run_T50s = get_simulation_data()
    print(type(run_T50s))
    print(type(run_T50s[0]))
    print(type(run_T50s[0][0]))
    print(len(run_T50s[0][250]))
    #data = pd.DataFrame(run_T50s)
    #print(data)
    #data.to_csv('test.csv')
    learning_curve(run_T50s, run_date, dataset, strategies)
    iteration_boxplot(run_T50s, run_date, dataset, strategies)
    iteration_histogram(run_T50s, run_date, dataset, strategies)
    plt.show()

