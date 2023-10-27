import argparse
import re
import sys
import tkinter as tk
import tkinter.filedialog as fd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from scipy.optimize import curve_fit

line = lambda x, m, b: m * x + b
logistic = lambda x, L, k, x0, dy: L / (1 + np.exp(-k * (x - x0))) + dy
michaelis_menten = lambda x, Vmax, KM: (Vmax * x) / (KM + x)

def double_logistic(x, L0, k0, x0_0, L1, k1, x0_1, dy):
    top_curve = L0 / (1 + np.exp(-k0 * (x - x0_0)))
    bottom_curve = L1 / (1 + np.exp(-k1 * (x - x0_1)))
    return top_curve + bottom_curve + dy

linregress_output = namedtuple('linear_regression_fit', ['slope', 'intercept'])
def linear_fit(x, y):   
    fit = curve_fit(line, x, y)
    output = linregress_output(fit[0][0], fit[0][1])
    return output

logistic_output = namedtuple('logistic_regression_fit', ['magnitude', 'steepness', 'midpoint', 'y_adjust'])
def logistic_fit(x, y): 
    # determine if L is positive or negative
    low_x_y = y[x.index(min(x))]  # y_value at low x
    high_x_y = y[x.index(max(x))]  # y_value at high x
    L_estimated = high_x_y-low_x_y  # May not be right value, should have right sign
    fit, covariance = curve_fit(logistic, x, y, p0 = [L_estimated, 0.5, np.mean(x), 1])
    print(np.diag(covariance))
    output = logistic_output(*fit)
    return output

double_logistic_output = namedtuple('double_logistic_regression_fit',
                                ['magnitude_0', 'steepness_0', 'midpoint_0',
                                'magnitude_1', 'steepness_1', 'midpoint_1',
                                'y_adjust'])
def double_logistic_fit(x, y):
    low_x_y = y[x.index(min(x))]  # y_value at low x
    high_x_y = y[x.index(max(x))]  # y_value at high x
    L_estimated = high_x_y-low_x_y  # May not be right value, should have right sign
    fit, covariance = curve_fit(double_logistic, x, y,
                                p0 = [L_estimated, 0.5, np.mean(x), -0.25, 0.5, 60, 1],
                                bounds = ([-np.inf,      0, 25, -0.3,      0,   55, -np.inf], # Lower bounds
                                          [      0, np.inf, 80,    0, np.inf,   65,  np.inf]))  # Upper bounds
    print(np.diag(covariance))
    output = double_logistic_output(*[fit])
    return output

michaelis_menten_output = namedtuple('michaelis_menten_fit',
                                     ['Vmax', 'KM'])

def michaelis_menten_fit(x, y):
    fit = curve_fit(michaelis_menten, x, y)
    return michaelis_menten_output(*fit[0])

def _check_duplicate(data, header):
    max_decimals = max([head.count('.') for head in data.columns])
    if header.count('.') == max_decimals:
        return True
    return False

def _dict_to_tuples(input):
    return [(i,x) for i in input for x in input[i]]

def _truncate_over_reads(data):
    last_index = data.index[-1]
    for header in data.columns:
        over_indices = data.index[data[header] == 'OVER']
        if len(over_indices) > 0:
            index = min(over_indices)
            if index < last_index:
                last_index = index
    return data.truncate(0, after=last_index-1)


float_regex = '([0-9]+([.][0-9]*)?|[.][0-9]+)'
# Detect a positive float followed by a capital C
temperature_regex = '(([0-9]+([.][0-9]*)?|[.][0-9]+))C'
# Detect a positive float followed by a unit in [M, nM, uM, mM]
concentration_regex = '(([0-9]+([.][0-9]*)?|[.][0-9]+))n?u?m?M'

def melt_curve(data, args):
    
    delimiter = args.delimiter
    curve = args.curve

    fit_type = {'double_logistic': double_logistic_fit,
                'logistic': logistic_fit}
    plot_type = {'double_logistic': double_logistic,
                 'logistic': logistic}
    
    data = _truncate_over_reads(data)

    times = data.index.to_numpy()
    melt_curves = {}
    for header in data.columns:
        # Regex magic to extract the temperature
        try:
            temp_str = re.search(temperature_regex, header).group()
        except:
            continue  # temp not present in header
        temp = float(temp_str[0:-1]) # Remove the C from the end
        header_parts = header.split(delimiter)
        
        for part in header_parts:  # The label is the substring without the temp
            if temp_str not in part:
                label = part

        if args.split_replicates:
            if _check_duplicate(data, header):
                label = f'{label}-{header[-1]}'
            else:
                label = f'{label}-0'      
        if label not in melt_curves:
            melt_curves[label] = {}

        reads_fit = data[header].to_numpy()
        # Ensure x and y are same length
        times_fit = times
        if len(times_fit) != len(reads_fit):
            times_fit = times_fit[0:len(reads_fit)]
        fit = linear_fit(times_fit, reads_fit)
        if temp not in melt_curves[label]:
            melt_curves[label][temp] = []
        melt_curves[label][temp].append(fit.slope)
    for label, fits in melt_curves.items():
        # fits form is dict of {temp: [slope, slope...]}
        # Convert to form [(temp: slope), (temp, slope)...]
        fits_list = _dict_to_tuples(fits)  
        x = [coord[0] for coord in fits_list]
        y = [coord[1] for coord in fits_list]
        plt.scatter(x, y, label=label)
        try:
            fit = fit_type[curve](x, y)
            x_space = np.linspace(min(x), max(x))
            plt.plot(x_space, plot_type[curve](x_space, *fit))
            print(f'{label}: {fit}')
        except Exception as e:
            print(label)
            print(e)
            try:
                fit = logistic_fit(x, y)
                x_space = np.linspace(min(x), max(x))
                plt.plot(x_space, logistic(x_space, *fit))
                print(f'{label}: {fit}')
            except Exception as e:
                print(label)
                print(e)

    plt.ylabel('Acivity (RFU/s)')
    plt.xlabel('Temperature (C)')
    if args.title:
        plt.title(args.title)

def Michaelis_Menten_plot(data, args):

    delimiter = args.delimiter
    window = args.window

    if window:
        data = data.truncate(before=window[0], after=window[1])

    data = _truncate_over_reads(data)

    times = data.index.to_numpy()
    raw_curves = {}
    for header in data.columns:
        # Regex magic to extract the concentration
        try:
            conc_str = re.search(concentration_regex, header).group()
        except:
            continue  # concentration not present in header
        conc = float(re.search(float_regex, conc_str).group()) # Remove the C from the end
        header_parts = header.split(delimiter)
        for part in header_parts:  # The label is the substring without the conc
            if conc_str not in part:
                label = part
        if label not in raw_curves:
            raw_curves[label] = {}
        reads_fit = data[header].astype('float64').to_numpy()
        # Ensure x and y are same length
        times_fit = times
        if len(times_fit) != len(reads_fit):
            times_fit = times_fit[0:len(reads_fit)]

        # Scale RFUs to molarity if requested
        if args.y_scale:
            reads_fit = reads_fit / float(args.y_scale)

        fit = linear_fit(times_fit, reads_fit)
        if conc not in raw_curves[label]:
            raw_curves[label][conc] = []
        raw_curves[label][conc].append(fit.slope)
    for label, fits in raw_curves.items():
        # fits form is dict of {conc: [slope, slope...]}
        # Convert to form [(conc: slope), (conc, slope)...]
        fits_list = _dict_to_tuples(fits)  
        x = [coord[0] for coord in fits_list]
        y = [coord[1] for coord in fits_list]
        plt.scatter(x, y, label=label)
        try:
            fit = michaelis_menten_fit(x, y)
            x_space = np.linspace(min(x), max(x))
            plt.plot(x_space, michaelis_menten(x_space, *fit))
            print(f'{label}: {fit}')
        except Exception as e:
            print(label)
            print(e)
            
    if args.y_scale:
        plt.ylabel('Activity (M product/s)')
    else:
        plt.ylabel('Acivity (RFU/s)')
    plt.xlabel('Substrate Concentration (uM)')
    if args.title:
        plt.title(args.title)

methods = {'melt_curve': melt_curve,
           'michaelis_menten': Michaelis_Menten_plot}

if __name__ == '__main__':
    root = tk.Tk()  # These lines hide the Root tkinter window from view
    root.withdraw()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-f', '--filename',
                            required=False)
    arg_parser.add_argument('-t', '--title',
                            required=False)
    arg_parser.add_argument('-s', '--save',
                            action='store_true',
                            required=False)
    arg_parser.add_argument('-m', '--method',
                            required=True,
                            choices=['melt_curve', 'michaelis_menten'])
    arg_parser.add_argument('-d', '--delimiter', default='-')
    arg_parser.add_argument('-c', '--curve',
                            choices=['logistic', 'double_logistic'],
                            default = 'logistic')
    arg_parser.add_argument('-w', '--window',
                            nargs=2,
                            default=None,
                            help='Specify a start time and end time for kinetic consideration.')
    arg_parser.add_argument('-ys', '--y_scale',
                            default=None,
                            help='The fluorescence of a 1M solution of product in RFUs')
    arg_parser.add_argument('-sr', '--split_replicates',
                            action='store_true')
    args = arg_parser.parse_args()

    filenames = [args.filename]  # Pack in list to match with tkinter output
    if filenames == [None]:
        filenames = fd.askopenfilenames(initialdir='./Processed_CSV', title='Select formatted CSV to plot')
        if len(filenames) == 0:
            raise(Exception('No filename(s) provided'))
        
    method = methods[args.method]
    for filename in filenames:
        data = pd.read_csv(filename, index_col='Time [s]')
        method(data, args)

    plt.legend()    
    if args.save:
        filename = fd.asksaveasfilename()
        if filename:
            format = filename.split('.')[-1]
            plt.save_fig(filename, format=format)
    plt.show()