import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from Strateos_Scheduler import read_evagreen, protein_activity_analysis, process_data, fit_curve, activity, double_logistic

def relative_sort(target_list: List, key_list: List) -> List:
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


# Parameters here are used if the desired sequence is not in Experiment_Summary.csv
manual_run = False
manual_eva_id = 'r1ggrhy388u8xm'  # Parents in r1gaq7fekq8p5a, r1gb8mc5w2dzw5, r1frpday3md7mm
manual_assay_id = 'r1ggs32xkan5dz'
# Parents in r1gb8wj8bczjsz, r1gb9bvuwg7twu, r1frskpwukg63b
manual_seqs = ['1111', '2222', '3333', '4444', '5555', '6666']
manual_seqs = ['1136', '1361', '6131', '1361', '1316', '1631', '1361', '1316', '1rr', '1636', '1361', '6131']

# Toggles for viewing raw data
print_raw = False
save_raw = False

summary = pd.read_csv('Experiment_Summary.csv', index_col='Index')
last_run = summary.iloc[16]
if not manual_run:
    print(last_run)

seq_ids = last_run['Sequences'].strip('[]').split(',')
seq_ids = [seq.strip(" '") for seq in seq_ids]
if manual_run:
    seq_ids = manual_seqs
evagreen_id = last_run['Evagreen Run ID']
if manual_run:
    evagreen_id = manual_eva_id
evagreen_data = read_evagreen(evagreen_id)
run_id = last_run['Assay Run ID']
if manual_run:
    run_id = manual_assay_id
run_data = protein_activity_analysis(run_id)


filenames = ['Seq_Data_1.csv', 'Seq_Data_2.csv', 'Seq_Data_3.csv', 'Seq_Data_4.csv']
#filenames = ['Seq_Data_Test.csv']
process_data(run_id, seq_ids, filenames, 3, evagreen_data, test=True)
print('done')

#df = pd.DataFrame()
 
#print(run_data)
print(evagreen_data)
for sequence in run_data:
    
    temps, times, products, fluors = zip(*run_data[sequence])
    slopes = activity(*zip(*run_data[sequence]))
    x = np.array(list(slopes.keys()))
    y = np.array(list(slopes.values()))
    if print_raw or seq_ids[sequence] == '6311':
        plt.figure()
        for i, temp in enumerate(temps):
            product = relative_sort(products[i], times[i])          
            product = np.array(product)
            product = product / (fluors[i] / np.mean(fluors))
            time = np.array(sorted(times[i]))
                
            plt.plot(time, product, label=temp)
        plt.legend()
        plt.ylim(0,30000)
        plt.title(seq_ids[sequence])
        if save_raw:
            plt.savefig(f'raw_curves/{sequence}.eps', format='eps')
    plt.figure(sequence)
    plt.scatter(x, y)
    plt.title(f'{seq_ids[sequence]}')
    try:
        fit = fit_curve(x, y)
    except Exception as e:
        print(f'error in sequence {seq_ids[sequence]}')
        print(e)
        continue
    print(seq_ids[sequence])
    print(fit._asdict())
    x_space = np.linspace(min(x), max(x))
    plt.plot(x_space, double_logistic(x_space, fit.k, fit.T50, fit.mag, fit.mag_2,))
plt.show()
