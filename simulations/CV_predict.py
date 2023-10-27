""" Script to run bandit simulations on chimeric protein engineering datasets.
"""

import load_data
import pickle
from sklearn.gaussian_process import kernels
import trial_types
import tools
import numpy as np
import matplotlib.pyplot as plt



#dataset = 'P450'
dataset = 'P450reduced'
DATA, ENCODING = load_data.get_dataset(dataset)

kernel = kernels.DotProduct(1) + kernels.WhiteKernel(1)




indices = list(range(len(DATA)))
from random import shuffle
shuffle(indices)

cvfold = 10
n_seqs_CV = int(np.floor(len(indices)/cvfold))

folds = []
for i in range(cvfold-1):
    folds.append(indices[i*n_seqs_CV:(i+1)*n_seqs_CV])

folds.append(indices[(i+1)*n_seqs_CV:])


Y = [] # if folded has T50, if unfolded has NaN
Yhat = [] 


for n,test_ind in enumerate(folds):
    print(n)
    
    train_ind = [i for i in indices if i not in test_ind]
    test_DATA = [DATA[i] for i in test_ind]
    train_DATA = [DATA[i] for i in train_ind]     

    T50_data = [(d.seq, d.T50) for d in train_DATA if not np.isnan(d.T50)]
    func_data = [(d.seq, 0) if np.isnan(d.T50) else (d.seq, 1) for d in train_DATA]

    T50_pred, r_std = tools.gp_reg(T50_data, test_DATA, kernel)
    p_func = tools.gp_class(func_data, test_DATA, kernel)

    Y.extend([d.T50 for d in test_DATA])

    Yhat.extend([T50_pred[i] if p_func[i]>0.5 else np.nan for i in range(len(test_DATA))])




tp=0
fn=0
fp=0
tn=0
for i in range(len(Y)):
    if not np.isnan(Yhat[i]) and not np.isnan(Y[i]): tp+=1 #  true pos: both predicted to be active 
    if     np.isnan(Yhat[i]) and not np.isnan(Y[i]): fn+=1 # false neg: predicted to be inactive, but active
    if not np.isnan(Yhat[i]) and     np.isnan(Y[i]): fp+=1 # false pos: predicted to be active, but inactive
    if     np.isnan(Yhat[i]) and     np.isnan(Y[i]): tn+=1 #  true neg: both predict inactive 
    
    

CC = np.corrcoef(np.array([(Y[i],Yhat[i]) for i in range(len(Y)) if not np.isnan(Yhat[i]) and not np.isnan(Y[i])]).T)[0,1]


Y_plt = [d if not np.isnan(d) else 33+0.5*np.random.normal() for d in Y]
Yhat_plt = [d if not np.isnan(d) else 33+0.5*np.random.normal() for d in Yhat]

plt.plot(Y_plt,Yhat_plt,'.')
plt.text(50,40,'tp=%0.2f, fp=%0.2f, fn=%0.2f, tn=%0.2f'%(tp/len(Y),fp/len(Y),fn/len(Y),tn/len(Y)))
plt.text(50,38,'CC = %0.2f'%CC)
plt.plot([40,60],[40,60])
plt.axis('square')



plt.savefig('T50_func_scatter.eps')

plt.show()
