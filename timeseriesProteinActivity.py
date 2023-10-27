# This code provided by Eriberto Lopez at Strateos

from io import StringIO
import pandas as pd
import re
from pathlib import Path
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import os

from transcriptic import Connection, Project, Run, Container
# Strateos methods
test = False
api = Connection.from_file("~/.transcriptic")


_ANALYSIS_TOOL = "TimeSeriesProteinActivity"
__version__ = "0.0.2"

project = Project('p1fjjtwb2mnnh6')


def protein_activity_analysis(run_id) -> pd.DataFrame:
    """
    This function aggregates containers and datasets from a protein activity Run.
    It generates a directory with the aggregated CSVs and plotted data, then
    zips it up and uploads the archive to the Run that generated the data.

    Parameters
    ----------
    run: transcriptic.jupyter.Run
        Is the Run object of the current Run that triggered the program
    """

    run = Run(run_id=run_id)
    merged_results_df, analysis_root_dir = protein_activity_aggregated(run)
    #visualize_timeseries(merged_results_df, analysis_root_dir, save_fig=False)
    return report_data(merged_results_df)


def protein_activity_aggregated(run):
    analysis_root_dir = f'{run.id}'
    Path(analysis_root_dir).mkdir(parents=True, exist_ok=True)
    # No idea why this works but it doesn't load containers df with out calling .containers() first
    try:
        df = run.containers()
    except Exception as e:
        print(e)
        df = run._containers.copy()
        
        
    # Containers df Add column prefix
    containers_df = rename_columns(df=df, prefix='Container')
    container_frames = []
    for i, row in containers_df.iterrows():
        aliquot_df = row['Containers'].aliquots.copy().reset_index()
        container_row_df = pd.DataFrame(
            [row] * len(aliquot_df.index)).reset_index().drop('index', axis=1)
        frame = pd.concat([container_row_df, aliquot_df],
                          axis=1).reset_index().drop('index', axis=1)
        container_frames.append(frame)
    container_result = pd.concat(container_frames).reset_index().drop('index',
                                                                      axis=1)
    # Change column name to merge dataframes
    container_result.rename(columns={'Id': 'Aliquot ID'}, inplace=True)
    # Write to CSV - container_result.csv
    container_result.to_csv(f'Strateos_outputs/{analysis_root_dir}/container_result.csv')

    # Data df Add column prefix
    data_dfs = rename_columns(df=run.data.copy(), prefix='Data')
    dataset_frames = []
    run.attributes['instructions']
    for i, row in data_dfs.iterrows():
        # Get CSV dataset and read into dataframe        
        try:
            do = [do for do in row.Datasets.data_objects() if
                  do.name.split('.')[-1] == 'csv'][0]
        except IndexError: 
            print('skipped data object')
            print([do.name for do in row.Datasets.data_objects()])
            continue
        datacsv = StringIO(str(do.data, 'utf-8'))
        df = pd.read_csv(datacsv)

        # pop last column off and expand as a column of all of the measurements
        gain_row = pop(df, len(df.index) - 1, axis=0).iloc[0]
        # GAIN is named under the Well column and the numerical value is under Fluorescence
        df[str(gain_row.Well)] = [gain_row.Fluorescence] * len(df.index)

        # After removing gain row, add the additional columns from the row we are iterrating on
        dataset_row_df = pd.DataFrame([row] * len(df.index)).reset_index().drop(
            'index', axis=1)
        frame = pd.concat([dataset_row_df, df], axis=1).reset_index().drop(
            'index', axis=1)
        dataset_frames.append(frame)
    datasets_result = pd.concat(dataset_frames).reset_index().drop('index',
                                                                   axis=1)

    # Gather additional information from Run.attributes to accurately get timepoint data
    instructions = [
        i for i in run.attributes['instructions'] if i['operation']['op'] == 'fluorescence'
    ]
    datasets = run.attributes['datasets']
    for ds in datasets:
        try:
            i = [i for i in instructions if i['id'] == ds['instruction_id']].pop()
        except IndexError: # One datapoint corresponding to a zip file instead of csv creates empty list
            print('skipped unpoppable list')
            print(ds['instruction_id'])
            continue
        # Augment ds dict with relevent info from the instruction
        ds['DataName'] = i['data_name']
        ds['DataTimepoint'] = i['completed_at']
    dataset_instr_df = rename_columns(df=pd.DataFrame(datasets), prefix='Data')
    similar_cols = [
        col for col in dataset_instr_df.columns if col in datasets_result.columns
    ]
    datasets_result = datasets_result.merge(dataset_instr_df, how='inner',
                                            on=similar_cols)
    datasets_result = pd.concat([
        datasets_result,
        datasets_result['DataName'].apply(
            lambda s: pd.Series({
                'read_num': get_value_from_dataname(s, value='Read'),
                'excitation_nm': get_value_from_dataname(s, value='ex'),
                'emission_nm': get_value_from_dataname(s, value='em')
            })
        )],
        axis=1
    )
    datasets_result.to_csv(f'Strateos_outputs/{analysis_root_dir}/datasets_result.csv')

    # Merge container_results and datasets_results
    similar_cols = [
        col for col in container_result.columns if col in datasets_result.columns
    ]
    merged_results_df = container_result.merge(
        datasets_result, how='inner', on=similar_cols
    )
    merged_results_df = apply_relevent_timepoint(merged_results_df)
    merged_results_df.to_csv(f'Strateos_outputs/{analysis_root_dir}/merged_results.csv')

    return merged_results_df, analysis_root_dir

def read_evagreen(run_id):
    run = Run(run_id=run_id)
    analysis_root_dir = f'{run.id}'
    Path(analysis_root_dir).mkdir(parents=True, exist_ok=True)
    # No idea why this works but it doesn't load containers df with out calling .containers() first
    try:
        df = run.containers()
    except Exception as e:
        print(e)
        df = run._containers.copy()
        
        
    # Containers df Add column prefix
    containers_df = rename_columns(df=df, prefix='Container')
    container_frames = []
    for i, row in containers_df.iterrows():
        aliquot_df = row['Containers'].aliquots.copy().reset_index()
        container_row_df = pd.DataFrame(
            [row] * len(aliquot_df.index)).reset_index().drop('index', axis=1)
        frame = pd.concat([container_row_df, aliquot_df],
                          axis=1).reset_index().drop('index', axis=1)
        container_frames.append(frame)
    container_result = pd.concat(container_frames).reset_index().drop('index',
                                                                      axis=1)
    # Change column name to merge dataframes
    container_result.rename(columns={'Id': 'Aliquot ID'}, inplace=True)
    # Write to CSV - container_result.csv
    if not os.path.exists(f'Strateos_outputs/{analysis_root_dir}'):
        os.makedirs(f'Strateos_outputs/{analysis_root_dir}')
    container_result.to_csv(f'Strateos_outputs/{analysis_root_dir}/evagreen_result.csv')

    # Data df Add column prefix
    data_dfs = rename_columns(df=run.data.copy(), prefix='Data')
    dataset_frames = []
    run.attributes['instructions']
    for i, row in data_dfs.iterrows():
        # Get CSV dataset and read into dataframe        
        try:
            do = [do for do in row.Datasets.data_objects() if
                  do.name.split('.')[-1] == 'csv'][0]
        except IndexError: 
            print('skipped data object')
            print([do.name for do in row.Datasets.data_objects()])
            continue
        datacsv = StringIO(str(do.data, 'utf-8'))
        df = pd.read_csv(datacsv)

        # pop last column off and expand as a column of all of the measurements
        gain_row = pop(df, len(df.index) - 1, axis=0).iloc[0]
        # GAIN is named under the Well column and the numerical value is under Fluorescence
        df[str(gain_row.Well)] = [gain_row.Fluorescence] * len(df.index)

        # After removing gain row, add the additional columns from the row we are iterrating on
        dataset_row_df = pd.DataFrame([row] * len(df.index)).reset_index().drop(
            'index', axis=1)
        frame = pd.concat([dataset_row_df, df], axis=1).reset_index().drop(
            'index', axis=1)
        dataset_frames.append(frame)
    datasets_result = pd.concat(dataset_frames).reset_index().drop('index',
                                                                   axis=1)

    # Gather additional information from Run.attributes to accurately get timepoint data
    instructions = [
        i for i in run.attributes['instructions'] if i['operation']['op'] == 'fluorescence'
    ]
    datasets = run.attributes['datasets']
    for ds in datasets:
        try:
            i = [i for i in instructions if i['id'] == ds['instruction_id']].pop()
        except IndexError: # One datapoint corresponding to a zip file instead of csv creates empty list
            print('skipped unpoppable list')
            print(ds['instruction_id'])
            continue
        # Augment ds dict with relevent info from the instruction
        ds['DataName'] = i['data_name']
        ds['DataTimepoint'] = i['completed_at']
    dataset_instr_df = rename_columns(df=pd.DataFrame(datasets), prefix='Data')
    similar_cols = [
        col for col in dataset_instr_df.columns if col in datasets_result.columns
    ]
    datasets_result = datasets_result.merge(dataset_instr_df, how='inner',
                                            on=similar_cols)
    datasets_result = pd.concat([
        datasets_result,
        datasets_result['DataName'].apply(
            lambda s: pd.Series({
                'read_num': get_value_from_dataname(s, value='Read'),
                'excitation_nm': get_value_from_dataname(s, value='ex'),
                'emission_nm': get_value_from_dataname(s, value='em')
            })
        )],
        axis=1
    )
    datasets_result.to_csv(f'Strateos_outputs/{analysis_root_dir}/evagreen_result.csv')

    # Merge container_results and datasets_results
    similar_cols = [
        col for col in container_result.columns if col in datasets_result.columns
    ]
    merged_results_df = container_result.merge(
        datasets_result, how='inner', on=similar_cols
    )
    
    merged_results_df = merged_results_df[merged_results_df['EVA-Green:DNADye'].notnull()]
    merged_results_df.to_csv(f'Strateos_outputs/{analysis_root_dir}/evagreen_result.csv')

    return list(merged_results_df['Fluorescence'])


def apply_relevent_timepoint(merged_results_df):
    frames = []
    filtered_df = merged_results_df[merged_results_df['source'].notnull()]
    filtered_df = filtered_df[filtered_df['incubation_temp'].notnull()]
    sorted_temps = list(set(filtered_df['incubation_temp']))
    sorted_temps.sort()
    for src in set(filtered_df['source']):
        for i, temp in enumerate(sorted_temps):
            temp_df = filtered_df[filtered_df['source'] == src]
            temp_df = temp_df[temp_df['incubation_temp'] == temp]
            #temp_df = temp_df[temp_df['read_num'] > -1]  # Exclude fluorocein measurement
            temp_df = pd.concat([
                temp_df,
                temp_df['DataTimepoint'].apply(
                    lambda s: pd.Series({
                        'timestamp': get_timestamp(s)
                    })
                )],
                axis=1
            )
            min_timestamp = min(temp_df['timestamp'])
            temp_df = pd.concat([
                temp_df,
                temp_df['timestamp'].apply(
                    lambda s: pd.Series({
                        'minutes': get_timepoint(s, min_timestamp)
                    })
                )],
                axis=1
            )
            frames.append(temp_df)
    return pd.concat(frames).reset_index().drop('index', axis=1)


def visualize_timeseries(merged_results_df, analysis_root_dir, save_fig=False):
    temp_colors = ['blue', 'blueviolet', 'mediumvioletred', 'crimson', 'red',
                   'orangered', 'gold']

    filtered_df = merged_results_df[merged_results_df['source'].notnull()]
    filtered_df = filtered_df[filtered_df['incubation_temp'].notnull()]
    sorted_temps = list(set(filtered_df['incubation_temp']))
    sorted_temps.sort()
    for src in set(filtered_df['source']):
        fig, ax = plt.subplots(figsize=(12.5, 7.5))
        for i, temp in enumerate(sorted_temps):
            temp_df = filtered_df[filtered_df['source'] == src]
            temp_df = temp_df[temp_df['incubation_temp'] == temp]
            temp_df = temp_df[
                temp_df['read_num'] > -1]  # Exclude fluorocein measurement
            ax = temp_df.plot(ax=ax, x='minutes', y='Fluorescence',
                              kind='scatter', c=temp_colors[i], label=f'{temp}')
        plt.title("Source:" + src)
        plt.legend(loc='best')
        if save_fig:
            plt.savefig(f'Strateos_outputs/{analysis_root_dir}/{src}-timeseries.png')
            plt.close()
        #=======================================================================
        # else:
        #     fig.show()
        #=======================================================================
    plt.show()

def report_data(merged_results_df):
    filtered_df = merged_results_df[merged_results_df['source'].notnull()]
    filtered_df.to_csv('temp_df_fluor.csv')
    filtered_df = filtered_df[filtered_df['incubation_temp'].notnull()]
    sorted_temps = list(set(filtered_df['incubation_temp']))
    sorted_temps.sort()
    output_dict = {}
    for src in set(filtered_df['source']):
        index = int(src.split(':')[1])
        output_dict[index] = []
        for temp in sorted_temps:
            temp_df = filtered_df[filtered_df['source'] == src]
            #temp_df.to_csv('temp_df_fluor.csv')
            temp_df = temp_df[temp_df['incubation_temp'] == temp]
            
            fluor = temp_df[
                temp_df['read_num'] == -1]
            temp_df = temp_df[
                temp_df['read_num'] > -1]  # Exclude fluorocein measurement
            temp = float(temp.split(':')[0])
            minutes = temp_df['minutes'].tolist()
            products = temp_df['Fluorescence'].tolist()
            fluor = fluor['Fluorescence'].tolist()[0]
            temp_xy_fluor = [temp, minutes, products, fluor]
            output_dict[index].append(temp_xy_fluor)

    return output_dict

def get_timestamp(s):
    return pd.to_datetime(s)


def get_timepoint(s, min_timestamp):
    timepoint = pd.to_datetime(s) - min_timestamp
    return int(timepoint.total_seconds() // 60)


def pop(df, values, axis=1):
    if axis == 0:
        if isinstance(values, (list, tuple)):
            popped_rows = df.loc[values]
            df.drop(values, axis=0, inplace=True)
            return popped_rows
        elif isinstance(values, (int)):
            popped_row = df.loc[values].to_frame().T
            df.drop(values, axis=0, inplace=True)
            return popped_row
        else:
            print('values parameter needs to be a list, tuple or int.')
    elif axis == 1:
        # current df.pop(values) logic here
        return df.pop(values)


def get_value_from_dataname(s, value: str):
    try:
        m = re.search(
            r'\d+',
            [tok for tok in s.split('-') if value in tok][0]
        )
        return int(m[0])
    except:  # Background Fluorocein measurement does not have Read in name
        return -1


def rename_columns(df, prefix):
    df.columns = [
        prefix + name if prefix not in name else name for name in df.columns
    ]
    return df

