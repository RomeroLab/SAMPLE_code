import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.decomposition import PCA



x_names = {'PCA': 'pc1', 'VAE': 'z1', 'MDS': 'MDS_1'}
y_names = {'PCA': 'pc2', 'VAE': 'z2', 'MDS': 'MDS_2'}


df = pd.read_csv('all_seqs.csv')

# Get rid of empty column and rename
df.drop('seq', axis=1, inplace=True)
df.columns = ['seq']

parents = ['1111', '2222', '3333', '4444', '5555', '6666']  # Define the names of the parental enzymes
df['Legend'] = [str(i) if i in parents else 'Chimera' for i in df.index]
color_list = sns.color_palette('tab10', len(parents))


# Hamming distance function
def hamming(seq1, seq2):
    return sum(s1 != s2 for s1, s2 in zip(seq1, seq2))


def xy(type):

    if type == 'PCA':
        for p in parents:
            df[p] = [hamming(df.loc[p, 'seq'], c) for c in df['seq']]
        # Import a list of colors to use to identify the parents      
        
        pca = PCA(n_components=3)
        
        X = df.loc[:, parents]
        
        pcs = pca.fit_transform(X)  # Perform PCA
        
        # Add back to the dataframe
        
        df['pc1'] = pcs[:, 0]
        df['pc2'] = pcs[:, 1]
        df['pc3'] = pcs[:, 2]
        
        return df
    
    elif type == 'VAE':
        df = pd.read_csv('all_seqs_latent.csv', index_col='seqID')
        df['Legend'] = [str(i) if i in parents else 'Chimera' for i in df.index]
        return df
    
    elif type == 'MDS':
        df = pd.read_csv('all_seqs_MDS.csv', index_col='seqID')
        df['Legend'] = [str(i) if i in parents else 'Chimera' for i in df.index]
        return df


def max_trace(xy):

    starting_points = []
    ending_points = []
    colors = ['blue', 'orange', 'red', 'green']
    for i in  [1, 2, 3, 4]:
        summary = pd.read_csv('Experiment_Summary.csv')
        seq_data = pd.read_csv(f'Seq_Data_{i}.csv', index_col='Seq_ID')
        slice_start = 3 * i - 3
        slice_end = 3 * i
        sequence_lists = [ast.literal_eval(seqs) for seqs in summary['Sequences']]
        learn = [seq_list[slice_start:slice_end] for seq_list in sequence_lists]
        concat = [y for x in learn for y in x]  # Concatenate all sequences in the learn
        plot_coords = []
        max_T50 = 0
        for seq in concat:
            T50 = seq_data['T50'][seq]
            if T50 == 'retry' or T50 == 'dead':
                continue
            T50 = float(T50)
            if float(seq_data['T50'][seq]) > max_T50:
                print(seq, T50)
                plot_coords.append([df[x_name][seq], df[y_name][seq]])
                max_T50 = T50
        x = [entry[0] for entry in plot_coords]
        y = [entry[1] for entry in plot_coords]
        starting_points.append([x[0], y[0]])
        ending_points.append([x[-1], y[-1]])
    
        plt.plot(x, y, label=f'learn #{i}', linewidth=3, color=colors[i - 1])
    
    for i, pair in enumerate(ending_points):
        plt.scatter([pair[0]], [pair[1]], s=50, color=colors[i])
        
        
def color_T50(xy):
    for i in  [1, 2, 3, 4]:
        summary = pd.read_csv('Experiment_Summary.csv')
        seq_data = pd.read_csv(f'Seq_Data_{i}.csv', index_col='Seq_ID')
        slice_start = 3 * i - 3
        slice_end = 3 * i
        sequence_lists = [ast.literal_eval(seqs) for seqs in summary['Sequences']]
        learn = [seq_list[slice_start:slice_end] for seq_list in sequence_lists]
        concat = [y for x in learn for y in x]  # Concatenate all sequences in the learn
        

plot_type = 'VAE' # Supported types: VAE, PCA, MDS
        
df = xy(plot_type)    
x_name = x_names[plot_type]
y_name = y_names[plot_type]


sns.scatterplot(data=df, x=x_name, y=y_name, color='0.5')
sns.scatterplot(data=df.loc[parents], x=x_name, y=y_name, hue='Legend', s=200, palette='tab10', edgecolor='k')


max_trace(df)
plt.legend(bbox_to_anchor=(1, 1))
plt.savefig('Chimera_Landscape_VAE_2d.svg')
plt.show()
