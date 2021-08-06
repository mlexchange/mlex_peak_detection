from scipy import signal
import scipy
import numpy as np
import pandas as pd
from astropy.modeling import models, fitting
import argparse
import glob
import pathlib
import math
import h5py
from packages.hitp import bayesian_block_finder

def peak_helper(x_data, y_data):
    num_peaks = 1
    flag_list = []
    total_p = signal.find_peaks_cwt(y_data, 1)
    total_img = signal.cwt(y_data, scipy.signal.ricker, list(range(1, 10)))
    total_img = np.log(total_img+1)
    if len(total_p) == 0:
        return [], [], []
    temp_list = []
    return_p = []
    if len(total_p > num_peaks):
        for i in total_p:
            temp_list.append((y_data[i], i))
        temp_list.sort()
        temp_list = temp_list[-num_peaks:]
        for i in temp_list:
            return_p.append(i[1])
        return_p.sort()
    g_unfit = None
    g_fit = None
    difference = x_data[1] - x_data[0]
    for i in return_p:
        largest_width = 0
        for i_img in range(len(total_img)):
            if total_img[i_img][i] > largest_width:
                largest_width = total_img[i_img][i]
        stddev = (largest_width*difference)/(2*math.sqrt(2*math.log(2)))
        g_init = models.Gaussian1D(
                amplitude=y_data[i],
                mean=x_data[i],
                stddev=stddev)
        g_init.mean.min = float(x_data[0])
        g_init.mean.max = float(x_data[-1])
        g_init.amplitude.min = 0
        if g_unfit is None:
            g_unfit = g_init
        else:
            g_unfit = g_unfit+g_init
    fit_g = fitting.SimplexLSQFitter()
    if len(return_p) == 1:
        fit_g = fitting.LevMarLSQFitter()
    g_fit = fit_g(g_unfit, x_data, y_data)
    FWHM_list = []
    if len(return_p) == 1:
        FWHM_list.append(g_fit.stddev.value)
    else:
        for i in range(len(return_p)):
            FWHM_list.append(getattr(g_fit, f"stddev_{i}"))
            FWHM_list[-1] = FWHM_list[-1].value
    residual = 0
    fit_total = 0
    y_total = 0
    for i in range(len(x_data)):
        fit_total += g_fit(x_data[i])
        y_total += y_data[i]
    residual = fit_total/y_total
    residual = abs(1-residual)
    if residual > 0.15:
        for i in return_p:
            flag_list.append(1)
    else:
        for i in return_p:
            flag_list.append(0)
    return return_p, FWHM_list, flag_list

def save_local_file(data_path, tags_table, filename):
    # getting rid of .csv from file name to add _tags.csv to end
    tags_file = filename[16:-4]+'_tags.csv'
    f = open(str(data_path)+tags_file, 'w')
    for i in tags_table:
        x1 = i['Peak'].split()[0][:-1]
        x2 = i['Peak'].split()[1]
        f.write(i['Tag']+','+x1+','+x2+','+i['FWHM']+','+i['flag']+'\n')
    f.close()
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='input filepath')
    parser.add_argument('results_dir', help='output filepath')
    args = parser.parse_args()
    OUTPUT_DIR = pathlib.Path(args.results_dir)
    data_path = pathlib.Path(args.data_dir)
    for filename in glob.glob(str(data_path)+'/*'):
        print(str(filename))
        try:
            # Different if statments to hopefully handel the files types needed
            # when graphing 1D data
            if filename.endswith('.csv'):
                # The user uploaded a CSV file
                df = pd.read_csv(filename)
                # Can't handle anything other than 3 columns for graphing
                if len(df.columns) != 2:
                    raise
            if filename.endswith('.xdi'):
                # The user uploaded a XDI file
                df = parseXDI(filename)
            if filename.endswith('.npy'):
                # The user uploaded a numpy file
                npyArr = np.load(filename)
                df = pd.DataFrame({'Column1': npyArr})
        except Exception as e:
            print("There was an error processing this file: " + str(e))
        data = pd.DataFrame.to_numpy(df)
        x_data = data[:, 0]
        if len(df.columns) == 2:
            y_data = data[:, 1]
        else:
            y_data = []
        baseline = None
        block = None
        base_model = None
        g_unfit = None
        g_fit = None
        # Linear Model from data on the left wall of the window to data on the
        # right wall of the window
        if baseline:
            slope = (y_data[-1] - y_data[0])/(x_data[-1] - x_data[0])
            intercept = y_data[0] - (slope * x_data[0])
            base_model = models.Linear1D(slope=slope, intercept=intercept)
            for i in range(len(y_data)):
                y_data[i] = y_data[i] - base_model(x_data[i])
        FWHM_list = []
        peak_list = []
        flag_list = []
        boundaries = bayesian_block_finder(np.array(x_data), np.array(y_data))
        for bound_i in range(len(boundaries)):
            lower = int(boundaries[bound_i])
            if bound_i == (len(boundaries)-1):
                upper = len(x_data)
            else:
                upper = int(boundaries[bound_i+1])
            temp_x = x_data[lower:upper]
            temp_y = y_data[lower:upper]
            temp_peak, temp_FWHM, temp_flag = peak_helper(temp_x,temp_y)
            temp_peak = [i+lower for i in temp_peak]
            flag_list.extend(temp_flag)
            FWHM_list.extend(temp_FWHM)
            peak_list.extend(temp_peak)
        return_list = []
        for i in range(len(peak_list)):
            diction = {}
            index = peak_list[i]
            x, y = x_data[index], y_data[index]
            diction['Tag'] = 'auto'
            diction['Peak'] = str(x)+', '+str(y)
            diction['FWHM'] = str(FWHM_list[i])
            diction['flag'] = str(flag_list[i])
            return_list.append(diction)
        save_local_file(OUTPUT_DIR, return_list, filename)
        print('peaks found for: {}'.format(filename))
