import transcriptic.commands as txc
from transcriptic import Connection, Project, Container
from contextlib import redirect_stdout
import json
import io
import glob
import random
import time
from datetime import date
import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from scipy.optimize import curve_fit
from typing import List,Dict,Tuple,Optional,TextIO
from scipy import stats
from collections import namedtuple

from timeseriesProteinActivity import protein_activity_analysis, read_evagreen

# Run Constants
max_stderr_ratio = np.inf # Set to 0.3 to exclude bubbles, empirically determined
fluor_cutoff = 0.2 # Fraction of average fluorescein below which points are excluded
kest = 0.5 # Starting point for curve_fit gradient descent 
T50est = 55 # Starting point for curve_fit gradient descent
mag_mult = 2  # magnitude of top curve must be >= mag_mult * bottom magnitude, else print warning
min_bg = 0
max_bg = 6
min_activity = 3
min_bg_ratio = 2
evagreen_minimum = 10000

test = False

# Strateos initialization
api = Connection.from_file("~/.transcriptic")

protocol_ids = {'GG': 'pr1fp4f9897rc74',
                'PCR': 'pr1fp4ezrgaa2x3'}

project_ids = {'biovalidation': 'p1fdh5g6xarwhr',
               'protocol launch testing': 'p1fjjtwb2mnnh6',}


# Strateos output types
tx_run_data = namedtuple('tx_run_data', ['run_id', 'date', 'status'])
tx_inventory_entry = namedtuple('tx_inventory_entry', ['label', 'id', 'barcode', 'container', 'storage', 'created_date', 'volume'])

def tx_runs(project_name: str):
    '''
    Get data for all runs in a project.
    
    Output format is [Title, run ID, date, status]. Runs are
    output in reverse chronological order, with output[0] always
    corresponding to the most recent run.
            
    :param project_name: The project ID or its nickname in project_ids
    :return: A list of tx_run_data objects in reverse chronological order
    '''  
    
    if project_name[0:2] != 'p1':
                project_name = project_ids[project_name]
    route = api.url(f'/api/runs?filter[project_id]={project_name}&fields[runs]=created_at,accepted_at,started_at,completed_at,test_mode,status,friendly_status,progress')
    
                
    while True:
        try:
            response = api.get(route)
            #print(json.dumps(response, indent=2))
            output = []
            for run in response['data'][0:3]:
                run_id = run['id']
                date = run['attributes']['created_at'] 
                status = run['attributes']['status'] 
                run_data = tx_run_data(run_id, date, status)
                output.append(run_data)  
            return output
        except Exception as e:
            print('tx_runs call failed. Retrying')
            print(e)
            time.sleep(20)
            continue

def tx_inventory(query: str,
                 include_aliquots: bool=False,
                 show_status: bool=False,
                 retrieve_all: bool=False) -> tx_inventory_entry:
    '''
    Find all containers that fit a certain label (not case sensitive)
    
    Output is a list of lists, format for each inner list is
    [label, container ID, barcode (if used), container type, storage, creation date]
    
    :param query: The label to seek
    :return: A list of lists in the format listed above
    '''
    
    read = io.StringIO()
    with redirect_stdout(read):  # txc.inventory prints rather than return data
        txc.inventory(api, include_aliquots, show_status, retrieve_all, query)
    
    # Remove headers and closing empty line
    out = read.getvalue().split('\n')[3:-1]
    # Remove spacer lines
    out = [entry for entry in out if entry[0]!='-'] 
    # String to list for each entry
    out = [entry.split('|') for entry in out] 
    # Remove unnecessary spaces
    out = [[substring.strip() for substring in entry] for entry in out] 
    out = [tx_inventory_entry(*entry, _volume_in_tube(entry[1])) 
           for entry in out]
    return out

def tx_launch(params: TextIO,
              protocol: str='goldengate',
              project: str=project_ids['protocol launch testing'],
              title=None,
              save_input=False,
              local=False,
              accept_quote=True,
              pm=None,
              test: bool=test,
              pkg=None,):
    '''
    Launch a run with Strateos based on a formatted JSON file
    
    :param params: A JSON file contianing the run parameters.
    :param protocol: The protocol to start.
    :param project: The project within which to launch the run.
    :param test: If True, Strateos will not perform experiments.
    I don't know what the others do but the above are the default values.
    '''

    txc.launch(api,
               protocol,
               project,
               title,
               save_input,
               local,
               accept_quote,
               params,
               pm,
               test,
               pkg)

    
def _volume_in_tube(container_id: str):
    '''Report the volume in a chosen container as a float'''
    tube = Container(container_id)
    aliquot = tube.attributes['aliquots'][0]    
    volume = aliquot['volume_ul']
    return float(volume)
    
    
def logistic(x: npt.ArrayLike,
             k0: float=1,
             x0: float=50,
             mag: float=1):
    '''
    For an input value x, calculate the output from a logistic function
    with parameters specified according to y = mag/(1 + np.exp(k0*(x-x0)))
    :param x: Input variable. Can be a scalar value or a 1D numpy array
    :param k0: Slope-defining coefficient
    :param x0: Midpoint of logistic curve
    :param mag: Amplitude of nonzero side of curve
    '''
    y = mag / (1 + np.exp(k0 * (x - x0)))
    return y


def double_logistic(x: npt.ArrayLike,
                    k1: float=0.5,
                    x1: float=50,
                    mag1: float=5,
                    mag2: float=0,
                    x2: float=62.5,
                    k2: float=0.466):
    '''
    For an input value x, calculate the output from a double logistic function
    with parameters specified according to y = mag/(1 + np.exp(k0*(x-x0)))
    :param x: Input variable
    :param k1: Slope-defining coefficient of upper logistic
    :param x1: Midpoint of logistic curve of upper logistic
    :param mag1: Amplitude of nonzero side of of upper logistic
    :param k2: Slope-defining coefficient of lower logistic
    :param x2: Midpoint of logistic curve of lower logistic
    :param mag2: Amplitude of nonzero side of curve of lower logistic
    '''
    y = logistic(x, k1, x1, mag1) + logistic(x, k2, x2, mag2)
    return y

fitted_curve = namedtuple('fitted_curve', 
                          ['k',
                           'T50',
                           'mag',
                           'mag_2',
                           'kerr',
                           'T50err',
                           'magerr',
                           'magerr_2'],
                          defaults=[None,
                                    'retry',
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None])

def fit_curve(temps: List[float],
              slopes: List[float]):
    '''
    Fit a double logistic curve to input data
    
    :param temps: Temperatures tested in list or array form
    :param slopes: Kinetic slopes corresponding to each input temperature
    :return: A fitted_curve object
    '''
    popt, pcov = curve_fit(double_logistic, temps, slopes,
                           p0=[kest, T50est, slopes[2], min_bg],
                           bounds=((0, -100, 0, min_bg),
                                   (np.inf, np.inf, np.inf, max_bg)))

    k, T50, mag, mag_2 = popt
    kerr, T50err, magerr, magerr_2 = np.sqrt(np.diag(pcov))
    return fitted_curve(k, T50, mag, mag_2, kerr, T50err, magerr, magerr_2)
    
def activity(temps: List[float],
             times_all: List[List[float]],
             products_all: List[List[float]],
             fluoresceins: List[float]):
    """
    Takes a filename (*.xlsx) and a list of wells to scan, the first of
    which must be the blank, and returns an average slope across the
    wells tested.
    :param times: The temperatures at which measurements were recorded
    :param times_all: A list of lists of time points, one list per temperature
    :param products_all: A list of lists of fluorescense readings, one list
    per temperature
    :param fluoresceins: The fluorescein standard readings for each temperature
    :return: A dictionary with temperatures as keys and fluorescein-normalized
    slopes as values
    """

    output = {}
    
    avg_fluor = np.average(fluoresceins)
    fluor_floor = avg_fluor * fluor_cutoff
    

    for temp, times, products, fluorescein in zip(temps,
                                                  times_all,
                                                  products_all,
                                                  fluoresceins):
        
        regression = stats.linregress(x=times, y=products)
        
        # If fluorescein too low or kinetic data too noisy, discard the point
        if (abs(regression.stderr/regression.slope) < max_stderr_ratio
            and fluorescein > fluor_floor):
            fluor_norm = fluorescein / avg_fluor
            # Include the point, normalized
            output[temp] = regression.slope / fluor_norm 

    return output

def process_data(run_id, seq_ids: List[str],
                 filenames: List[str] = ['Seq_Data.csv'],
                 batch_size=3,
                 evagreen_data: List[float] = None,
                 test: bool = False):
    '''Process recieved data into a format for internalization
    
    Take data from strateos and update each learning file with its
    corresponding data
    :param run_id: The ID of the run to be processed
    :param seq_ids: A list of sequence IDs
    :param filenames: A list of learning file names, one for each agent
    :param batch_size: The number of sequences selected for each agent
    :param evagreen_data: The EvaGreen signal for each sequence in seq_ids
    :param test: If True, do not write output to the files in filenames
    '''
    
    if evagreen_data is None:  # Give default value and tag for error checking
        evagreen_data = [-1 for _ in range(batch_size) for __ in filenames]
    
    # Expand filenames to match indices with seq_ids
    filenames_indexed = []
    for filename in filenames:
        for x in range(batch_size):
            filenames_indexed.append(filename)
    
    run_data = protein_activity_analysis(run_id)
    for i, sequence in enumerate(run_data):
        print(' ')
        slopes = activity(*zip(*run_data[sequence]))
        x = np.array(list(slopes.keys()))
        y = np.array(list(slopes.values()))
        try:
            fit = fit_curve(x, y)
        except Exception as e:  
            # If curve cannot be fit, mark sequence as retry
            print(f'error in sequence {seq_ids[sequence]}')
            print(e)
            fit = fitted_curve(T50='retry')
            if test:
                print(seq_ids[sequence], fit)
            else:
                updateT50Data(filename=filenames_indexed[sequence],
                              seq_id=seq_ids[sequence],
                              data=fit,
                              run_id=run_id,
                              evagreen=evagreen_data[i])
            continue
        if evagreen_data[i] < evagreen_minimum:  
            # If EvaGreen too low, mark sequence as retry
            print(f'bad eva: {evagreen_data[i]}')
            fit = fitted_curve(T50='retry')
            if test:
                print(seq_ids[sequence], fit)
                print('eva too low')
            else:
                updateT50Data(filename=filenames_indexed[sequence],
                              seq_id=seq_ids[sequence],
                              data=fit,
                              run_id=run_id,
                              evagreen=evagreen_data[i])
            continue
        prior_data = pd.read_csv(filenames_indexed[sequence], index_col='Seq_ID')
        if test:
            print(f"prior data = {prior_data['T50'][seq_ids[sequence]]}")
            print(f'evagreen = {evagreen_data[i]}')
            
        # if activity is too low to label sequence live, and
        # sequence not labeled retry for fit error or low evagreen signal
        if (fit.mag < min_activity or fit.mag < min_bg_ratio * fit.mag_2) and fit.T50 != 'retry': 
            if prior_data['T50'][seq_ids[sequence]] == 'retry': # Label inactive sequences dead on second dead reading only
                fit = fitted_curve(T50='dead')
                print(f'max(y)={max(y)}, min_activity={min_activity}')
                if max(y) >= min_activity:
                    try:
                        print(f'fit.mag={fit.mag}, min_bg_ratio * fit.mag_2 = {min_bg_ratio * fit.mag_2}')
                    except:
                        print('print failed')
            else:  # First time inactive gets labeled for retry
                fit = fitted_curve(T50='retry')
            if test:
                print(seq_ids[sequence], fit) 
                continue
                
        if test:
            print('good data')
            print(seq_ids[sequence], fit)
            
        #  By here, sequence must have good Evagreen, no fit error, and signal greater than dead threshold    
        else:
            updateT50Data(filename=filenames_indexed[sequence],
                          seq_id=seq_ids[sequence],
                          data=fit,
                          run_id=run_id,
                          evagreen=evagreen_data[i])


def strateos_run(sequences: List[str],
                 filename: str='Seq_Data.csv'):
    '''Perform a run and record data
    
    :param sequences: A list of sequence IDs to test
    :param filename: A .csv file containing columns "Seq_ID" and "Fragments"
    '''
    dataframe = pd.read_csv(filename, dtype={"Seq_ID": str})
    dataframe.set_index("Seq_ID", inplace=True)
    
    all_frags = {}  # All fragments used and the volumes required of each
    assemblies = []
    
    for seq in sequences:
        frags = dataframe["Fragments"][seq].split()
        for frag in frags:
            if frag in all_frags:
                all_frags[frag] += 5
            else:
                all_frags[frag] = 15
                
    for seq in sequences:
        frags = dataframe["Fragments"][seq].split()
        json_input = []
        frag_found = False
        for frag in frags:
            frag_found = False
            inventory = tx_inventory(frag)
            for entry in inventory:
                if entry.volume > all_frags[frag]+5 and entry.label.lower() == frag.lower():
                    json_input.append({'containerId': entry.id, 'wellIndex': 0})
                    frag_found = True
                    break       
            if not frag_found:
                print(f'frag not found: {frag}')
                raise Exception(f'frag not found: {frag}')
        assemblies.append({'fragments': json_input})
    
    run_json = {
                  "parameters": {
                    "ptype": {
                      "value": "linear_assembly",
                      "inputs": {
                        "linear_assembly": {
                          "stoichiometry": {
                            "value": "equivolume",
                            "inputs": {
                              "equivolume": {
                                "volume": 5
                              }
                            }
                          },
                        "assemblies": assemblies
                        }
                      }
                    }
                  }
                }    

    filename=''
    for i in range(1000):
        num = str(i).zfill(3)
        filename = f'param_files/params{num}.json'
        glb = glob.glob(filename)
        if len(glb) == 0:
            break
    with open(filename, 'w') as params:
        json.dump(run_json, params)  # save run format before submitting
        print('wrote')
    with open(filename, 'r') as params:
        tx_launch(params)  # submit run request
    print(run_json)
    
project = Project('p1fjjtwb2mnnh6')
 

def parseT50Data(filename: str) -> Tuple[Dict[str,Tuple[str,int]],
                                         Dict[str,Tuple[str,float]],
                                         Dict[str,str]]:
    '''
    Read T50 data from a file into a form usable in learning 
    :param filename: Name of file from which to read
    :return: Three dictionaries: func_seqs, thermo_seqs, and unexplored_seqs.
    func_seqs has sequence IDs as keys and tuples of format (one-hot encoded
    sequence, [1 for active or 0 for inactive]) as values.
    thermo_seqs has sequence IDs as keys and tuples of format (one-hot encoded
    sequence, T50) as values.
    unexplored_seqs has sequence IDs as keys and one-hot encoded sequences
    as values.
    '''
    dataFrame = pd.read_csv(filename, dtype={"Seq_ID": str}) 
    dataFrame.set_index("Seq_ID", inplace=True)
    func_seqs: Dict[str,Tuple[str,int]] = {}
    thermo_seqs: Dict[str,Tuple[str,float]] = {}
    unexplored_seqs: Dict[str,str] = {}

    for seq_ID in dataFrame.index:
        row = dataFrame.loc[seq_ID]
        bin_seq = list([int(res) for res in row["Sequence"]])
        if str(row["T50"]) == "nan": 
            unexplored_seqs.update({seq_ID: bin_seq}) #Sequences not yet evaluated go in unexplored
        elif row["T50"] == "dead":
            func_seqs.update({seq_ID: (bin_seq, 0)}) #Inactive sequences go in functional as a 0
        elif row["T50"] == "retry":
            unexplored_seqs.update({seq_ID: bin_seq}) #Confusing sequences also go in unexplored
        else:
            func_seqs.update({seq_ID: (bin_seq, 1)}) #Live sequences go in functional as a 1 and in thermo_seqs as their T50s
            thermo_seqs.update({seq_ID: (bin_seq, float(row["T50"]))})
    return func_seqs, thermo_seqs, unexplored_seqs #Return the three dictionaries


def updateT50Data(filename: str='',
                  seq_id: str='',
                  data: fitted_curve=fitted_curve(),
                  run_id: str='',
                  evagreen: float=0.0):
    '''
    Add newly acquired data to working data structures and save it to file
    :param filename: Name of file where new data are saved
    :param seq_id: The ID of the sequence being written out
    '''
    dataFrame = pd.read_csv(filename, dtype={"Seq_ID": str})
    dataFrame.set_index("Seq_ID", inplace=True)
    dataFrame["T50"] = dataFrame["T50"].astype(object)
    dataFrame["T50err"] = dataFrame["T50err"].astype(object)
    dataFrame["k"] = dataFrame["k"].astype(object)
    dataFrame["kerr"] = dataFrame["kerr"].astype(object)
    dataFrame["run_id"] = dataFrame["run_id"].astype(object)
    
    if data.T50 == 'retry' or data.T50 == 'dead': #Data is either dead or retry
        dataFrame.at[seq_id,"T50"] = str(data.T50)
        dataFrame.at[seq_id,"run_id"] = str(run_id)

    else: #Data is complete, sequence is live
        dataFrame.at[seq_id,"T50"] = str(data.T50)
        dataFrame.at[seq_id,"T50err"] = str(data.T50err)
        dataFrame.at[seq_id,"k"] = str(data.k)
        dataFrame.at[seq_id,"kerr"] = str(data.kerr)
        dataFrame.at[seq_id,"mag"] = str(data.mag)
        dataFrame.at[seq_id,"magerr"] = str(data.magerr)
        dataFrame.at[seq_id,"evagreen"] = str(evagreen)
        dataFrame.at[seq_id,"run_id"] = str(run_id)
    dataFrame.to_csv(filename) #Save the updated data frame to the original file

def _choose_seq(func_seqs: Dict[str, Tuple[str, int]],
                thermo_seqs: Dict[str, Tuple[str, float]],
                unexplored_seqs: Dict[str, str],
                kernel) -> Tuple[str, float]:
    '''
    Select a sequence to test based on maximum eUCB.
    
    :param func_seqs: A dictionary with sequence IDs as keys and tuples of
    (one_hot_sequence, binary functionality [0 for inactive, 1 for active])
    as values
    :param thermo_seqs: A dictionary with sequence IDs as keys and tuples of
    (one_hot_sequence, T50) as values
    :param unexplored_seqs: A dictionary with sequence IDs as keys and one-hot
    encoded sequences as values
    :return: A Tuple of format (chosen sequence ID, predicted T50)
    '''
    
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

        # Subtract minimum mean from all means to force range to start at zero
        min_mean = min(y_means.values())
        y_means_zeroed = {ID: y_mean - min_mean for ID, y_mean
                          in y_means.items()}

        upper_bounds = {}
        for ID in y_means_zeroed:
            u = y_means_zeroed[ID]
            d = y_stds[ID]
            p = probs[ID]
            ub = (u + 2*d) * p
            upper_bounds.update({ID: ub})

        # Choose seq with largest upper bound
        chosen_seq_ID = max(upper_bounds, key=lambda x: upper_bounds[x])
        pred_T50 = y_means[chosen_seq_ID]

        return chosen_seq_ID, pred_T50

def learn(iterations: int,
          filenames: List[str],
          sim: bool=False,
          batch_size: int=1,
          white_kernel: float=1,
          dot_product: float=1):
    """
    Performs the experiment a pre-defined number of times with a pre-defined batch size
    :param iterations: Number of cycles of experimentation and learning to run
    :param filenames: List of file names housing prior data and where new data will be saved
    :param sim: boolean: True if testing code output, False if real.
    :param batch_size: Number of sequences to test per file in filenames
    :param white_kernel: Noise level of the white kernel in the GPR, default=1
    :param dot_product: Inhomogeneity of the dot product kernel in the GPR, default=1
    """
    
    #Type declarations
    chosen_bin_seq: str
    predT50: float

    kernel = DotProduct(dot_product) + WhiteKernel(white_kernel)

    for _ in range(iterations):
        
        all_run_seqs = [] # Store all sequences to run at once here
        for filename in filenames:
            
            func_seqs, thermo_seqs, unexplored_seqs = parseT50Data(
                filename)
    
            if not unexplored_seqs: #Stop running if no more unexplored seqs
                print("run finished")
                break
    
            chosen_seqs = []
            predT50s = []
    
            for _ in range(batch_size):
                if not unexplored_seqs:  # If no seqs available, pick nothing
                    break
                
                chosen_seq_ID, predT50 = _choose_seq(func_seqs,
                                                          thermo_seqs,
                                                          unexplored_seqs,
                                                          kernel)
                chosen_seqs.append(chosen_seq_ID)
                predT50s.append(predT50)
                chosen_bin_seq = unexplored_seqs.pop(chosen_seq_ID)
                func_seqs.update({chosen_seq_ID: (chosen_bin_seq, 1)})
                thermo_seqs.update({chosen_seq_ID: (chosen_bin_seq, predT50)})
    
    
            # If not, would break assay and waste time/resources
            assert(len(chosen_seqs) == len(predT50s))
    
            all_run_seqs += chosen_seqs
        
        print(all_run_seqs)
        if sim:
            return

        summary = pd.read_csv('Experiment_Summary.csv', index_col='Index')
        now = date.today()
        datestring = f'{str(now.year).zfill(4)}{str(now.month).zfill(2)}{str(now.day).zfill(2)}'
        summary.loc[len(summary.index)] = [datestring,all_run_seqs,None,None]
        summary.to_csv('Experiment_Summary.csv')

        previous_run = tx_runs('protocol launch testing')[0].run_id

        while True:  # Check run status periodically until submitted run starts
            try:
                strateos_run(all_run_seqs)
                if tx_runs('protocol launch testing')[0].run_id != previous_run:
                    break
            except:  # Sometimes strateos_run throws errors that don't affect this
                print('error in strateos_run')
                if tx_runs('protocol launch testing')[0].run_id != previous_run:
                    break
            time.sleep(30)
        
        print('run started')
        run_ids = set()
        evagreen_id = None
        
        while True:
            latest_run = tx_runs('protocol launch testing')[0]
            run_ids.add(latest_run.run_id)
            if (latest_run.run_id != previous_run # First check should be unnecessary
                and len(run_ids) == 5):  # Fifth run is the assay
                break
            time.sleep(60)
            if len(run_ids) == 3 and evagreen_id is None:  # Third run is EvaGreen
                evagreen_id = latest_run.run_id
            print(latest_run)
        run_id = tx_runs('protocol launch testing')[0].run_id
        
        summary = pd.read_csv('Experiment_Summary.csv', index_col='Index')
        last_row = summary.iloc[-1]
        summary.loc[len(summary.index)-1] = [last_row['Date'], last_row['Sequences'], evagreen_id, run_id]
        summary.to_csv('Experiment_Summary.csv')
        
        while True:  # Wait for assay step to complete
            assay_run = tx_runs('protocol launch testing')[0]
            if assay_run.status == 'complete':
                break
            time.sleep(60)  # Wait for run to be complete
            print('waiting for assay completion')
            
        evagreen_data = read_evagreen(evagreen_id)
        process_data(run_id, all_run_seqs, filenames, batch_size, evagreen_data)


if __name__ == "__main__":
    # Execute one round of learning. This takes roughly 2 days. Computer must stay running for the duration.
    learn(1,
          ['Seq_Data_1.csv',
           'Seq_Data_2.csv',
           'Seq_Data_3.csv',
           'Seq_Data_4.csv'],
          sim=False,
          batch_size=3)
