import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

agents = [1,2,3,4]
zero_point = 20  # Plot inactive sequences at this value to avoid compressing y-axis
point_offset = 0.1  # Space points within a batch this far apart

marker_size = 40
dashed_line_width = 1.6

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15, 6))

for agent in agents:

    seq_data_filename = f'Seq_Data_{agent}.csv'
    summary = pd.read_csv('Experiment_Summary.csv')
    data = pd.read_csv(seq_data_filename, index_col='Seq_ID')
    
    sequence_lists = [ast.literal_eval(seqs) for seqs in summary['Sequences']]
    
    # Get only sequences tested by this agent
    slice_start = 3*agent - 3
    slice_end = 3*agent
     
    
    # List all sequences tested by the current agent in chronological order
    learn = [seq_list[slice_start:slice_end] for seq_list in sequence_lists]
    concat = [y for x in learn for y in x]
    
    
    T50s = [data['T50'][seq] for seq in concat]
    parent_T50s = [data['T50'][seq] for seq in ['1111', '2222', '3333', '4444', '5555', '6666']]
    for i, T50 in enumerate(parent_T50s):
        if T50 == 'dead':
            parent_T50s[i] = zero_point
        else:parent_T50s[i] = float(T50)
        
    # Spread parents to avoid excessive overlap
    parent_x = [np.random.normal(0, 0.1) for _ in parent_T50s]
    
    T50s_plot = []
    observed_seqs = []
    for T50, seq in zip(T50s, concat):
        observed_seqs.append(seq)
        if T50 == 'retry':
            T50s_plot.append('r')
        elif T50 == 'dead':
            # If a sequence observed multiple times, only final read is dead, other(s) retry
            if observed_seqs.count(seq) < concat.count(seq):
                T50s_plot.append('r')
            else:
                T50s_plot.append('d')
        elif T50 == '':
            T50s_plot.append(None)
        else:
            # If a sequence observed multiple times, only final read is live, other(s) retry
            if observed_seqs.count(seq) < concat.count(seq):
                T50s_plot.append('r')
            else:
                T50s_plot.append(float(T50))
                
    T50s_plot = [T50 for T50 in T50s_plot if type(T50) in [int, float, str]]
    
    style_values = []
    for T50 in T50s:
        if T50 == 'r':
            style_values.append(2)
        if T50 == 'd':
            style_values.append(1)
        else:
            style_values.append(3)

    plot_data = pd.DataFrame(data={'Seq_ID': concat, 'T50': T50s_plot})
    conditions = [plot_data['T50'] == 'd', plot_data['T50'] == 'r']
    plot_data['T50 (C)'] = pd.to_numeric(plot_data['T50'], errors='coerce').fillna(zero_point).astype('int')
    plot_data['Style'] = np.select(conditions, [1, 2], default=3)
    plot_data['Hue'] = np.select(conditions, ['black', 'gray'], default='red')
    plot_data['Batch'] = plot_data.index // 3 + 1
    noise = [-point_offset, 0, point_offset] * (plot_data['Batch'].size // 3)
    plot_data['x_coord'] = plot_data['Batch'] + noise
    hue_order = ['black', 'gray', 'red']
    markers = {1:'x', 2:'$?$', 3:'o'}


    curr_axs = axs[(agent - 1) // 2, (agent - 1) % 2]
    curr_axs.axhline(y=max(parent_T50s), linestyle='--', color='green', linewidth=dashed_line_width, alpha=0.4, dashes=(5, 5))

    curr_axs.scatter(x=parent_x, y=parent_T50s, color='green')
    curr_axs.scatter(x='x_coord', y='T50 (C)', data=plot_data, color='Hue')

    curr_axs.set_title(f'Agent #{agent}', size=15)
    curr_axs.set_xticks(np.arange(0, plot_data['Batch'].size // 3 + 1, 1))
    curr_axs.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    curr_axs.tick_params(axis='y', which='both', width=2, length=6, labelsize=13)
    [i.set_linewidth(3) for i in curr_axs.spines.values()]

fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.ylabel('T50 (C)', size=20)
plt.xlabel('Learning Round', size=20)
plt.tight_layout()
plt.show()
