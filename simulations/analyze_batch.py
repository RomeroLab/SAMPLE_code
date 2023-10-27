"""Script for analyzing run data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import load_data
import tools
from tools import DP


def plot_batch(run_date='20221130',
               dataset='P450reduced',
               strategies=['random_selection',
                           'ucb',
                           'pp_ucb',
                           'expected_ucb'],
               save=False
               ):
    
    T50s = np.array([float(s[1]) for s in load_data.read_data(dataset+'_thermostability.data')[1]])
    temp_thresh = (max(T50s)-min(T50s))*0.9 + min(T50s) # 90% of the max
    #temp_thresh = np.percentile(T50s,95) # 95th percentile
    #temp_thresh = max(T50s) - 1    
    #print('Sequences above thresh %i'%sum(T50s>temp_thresh))  
    run_datas = []
    for strat in strategies:
        rd = []
        for bs in range(1,21):
            strat_data = pickle.load(open(f'{run_date}_{dataset}_run/{strat}_batch_batch{bs}.p','rb'))
            r = []
            for trial in strat_data:
                above = [i for i,s in enumerate(trial) if s.T50>temp_thresh]
                if above!=[]:
                    num_expts = min(above)
                    num_expts = int(np.ceil(num_expts/bs) * bs ) # round up to batch size 
                    r.append(num_expts)
                else:
                    r.append(500)
            rd.append(r)
        run_datas.append(rd)
    
    mean_perf = [[np.mean(bs) for bs in strat] for strat in run_datas]
    
    
    plt.plot(list(range(1,21)),np.array(mean_perf).T,'.-')
    plt.legend(strategies)
    plt.xlabel('Batch size')
    plt.ylabel('Number of trials to reac 90% of max')
    plt.title('Effect of Batch Size on Performance')
    if save:
        plt.savefig('mean_perf.png')


if __name__ == "__main__":
    plot_batch()
    plt.show()