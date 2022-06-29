import math
import os
import numpy as np
from scipy import signal
from astropy.modeling import models, fitting

# Imported code by Robert Tang-Kong
from packages.hitp import bayesian_block_finder


def peak_helper(x_data, y_data, num_peaks, peak_shape):
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    c = 2*np.sqrt(2*np.log(2))
    ind_peaks = signal.find_peaks_cwt(y_data, 1)
    ref = signal.cwt(y_data, signal.ricker, list(range(1, 10)))
    ref = np.log(ref+1)
    if len(ind_peaks) == 0:
        return [], [], [], None, None
    init_xpeaks = x_data[ind_peaks]
    init_ypeaks = y_data[ind_peaks]
    sorted_ind_peaks = init_ypeaks.argsort()
    sorted_ind_peaks = ind_peaks[sorted_ind_peaks]
    if len(sorted_ind_peaks) > num_peaks:
        init_xpeaks = x_data[sorted_ind_peaks[-num_peaks:]]
        init_ypeaks = y_data[sorted_ind_peaks[-num_peaks:]]
        ind_peaks = sorted_ind_peaks[-num_peaks:]
    g_unfit = None
    for ind, (xpeak, ypeak) in enumerate(zip(np.flip(init_xpeaks), np.flip(init_ypeaks))):
        i = ind_peaks[ind]
        largest_width = 0
        for i_img in range(len(ref)):
            if ref[i_img][i] > largest_width:
                largest_width = ref[i_img][i]
        if peak_shape == 'Voigt':
            g_init = models.Voigt1D(x_0=xpeak,
                                    amplitude_L=ypeak,
                                    fwhm_L=largest_width * (x_data[1] - x_data[0]),
                                    fwhm_G=largest_width * (x_data[1] - x_data[0]))
        else:
            sigma = (largest_width * (x_data[1] - x_data[0])) / c
            g_init = models.Gaussian1D(amplitude=ypeak,
                                       mean=xpeak,
                                       stddev=sigma)
        if g_unfit is None:
            g_unfit = g_init
        else:
            g_unfit = g_unfit + g_init
    fit_g = fitting.SimplexLSQFitter()
    if init_xpeaks.shape[0] == 1:
        fit_g = fitting.LevMarLSQFitter()
    g_fit = fit_g(g_unfit, x_data, y_data)
    residual = np.abs(g_fit(x_data) - y_data)
    if np.mean(residual/y_data) > 0.10:
        flag_list = list(np.ones(num_peaks))
    else:
        flag_list = list(np.zeros(num_peaks))
    FWHM_list = []
    if len(init_ypeaks) == 1:
        if peak_shape == 'Voigt':
            fG = g_fit.fwhm_G.value
            fL = g_fit.fwhm_L.value
            FWHM_list.append((0.5346 * fL + np.sqrt(0.2166*(fL ** 2) + (fG ** 2))))
        else:
            FWHM_list.append(c*g_fit.stddev.value)
    else:
        for i in range(len(init_ypeaks)):
            if peak_shape == 'Voigt':
                fG = getattr(g_fit, f"fwhm_G_{i}")
                fL = getattr(g_fit, f"fwhm_L_{i}")
                FWHM_list.append((0.5346*fL+np.sqrt(0.2166*(fL**2) + (fG**2))))
            else:
                FWHM_list.append(c*getattr(g_fit, f"stddev_{i}"))
    return ind_peaks, FWHM_list, flag_list, g_unfit, g_fit


# The logic on fitting peaks to data split up by blocks, or all the data
# together.  If data is split by blocks, we attempt to fit 3 peaks to the
# section
def get_peaks(x_data, y_data, num_peaks, peak_shape, baseline=None, block=None):
    base_list = None
    unfit_list = [[], []]
    fit_list = [[], []]
    residual = [[], []]
    base_model = None
    g_unfit = None
    g_fit = None
    # Linear Model from data on the left wall of the window to data on the
    # right wall of the window
    if baseline:
        base_list = [[], []]
        slope = (y_data[-1] - y_data[0])/(x_data[-1] - x_data[0])
        intercept = y_data[0] - (slope * x_data[0])
        base_model = models.Linear1D(slope=slope, intercept=intercept)
        base = list(base_model(np.array(x_data)))
        y_data = list(np.array(y_data) - base_model(np.array(x_data)))
        base_list[0] = x_data
        base_list[1] = base

    FWHM_list = []
    peak_list = []
    flag_list = []

    if block:
        num_peaks = 1   # 1 peak per block
        boundaries = bayesian_block_finder(np.array(x_data), np.array(y_data))
        for bound_i in range(len(boundaries)):
            lower = int(boundaries[bound_i])
            if bound_i == (len(boundaries)-1):
                upper = len(x_data)
            else:
                upper = int(boundaries[bound_i+1])
            temp_x = x_data[lower:upper]
            temp_y = y_data[lower:upper]
            temp_peak, temp_FWHM, temp_flag, unfit, fit = peak_helper(
                    temp_x,
                    temp_y,
                    num_peaks,
                    peak_shape)
            temp_peak = [i+lower for i in temp_peak]
            flag_list.extend(temp_flag)
            FWHM_list.extend(temp_FWHM)
            peak_list.extend(temp_peak)
            unfit_list[0].extend(temp_x)
            fit_list[0].extend(temp_x)
            for i in temp_x:
                if unfit is None:
                    unfit_list[1].append(0)
                    fit_list[1].append(0)
                else:
                    unfit_list[1].append(unfit(i))
                    fit_list[1].append(fit(i))

    else:
        peak_list, FWHM_list, flag_list, g_unfit, g_fit = peak_helper(
                        x_data,
                        y_data,
                        num_peaks,
                        peak_shape)
        unfit_list[0].extend(x_data)
        fit_list[0].extend(x_data)
        for i in x_data:
            if g_unfit is not None:
                unfit_list[1].append(g_unfit(i))
                fit_list[1].append(g_fit(i))
            else:
                unfit_list[1].append(0)
                fit_list[1].append(0)

    return_list = []
    for i in range(len(peak_list)):
        diction = {}
        diction['index'] = peak_list[i]
        diction['FWHM'] = FWHM_list[i]
        diction['flag'] = flag_list[i]
        return_list.append(diction)

    residual[0].extend(fit_list[0])
    temp_fit = np.array(fit_list[1])
    temp_y = np.array(y_data)
    resid = temp_y-temp_fit
    residual[1].extend(resid)

    return return_list, unfit_list, fit_list, residual, base_list


def add_paths_from_dir(dir_path, supported_formats, list_file_path):
    '''
    Args:
        dir_path, str:            full path of a directory
        supported_formats, list:  supported formats, e.g., ['tiff', 'tif', 'jpg', 'jpeg', 'png']
        list_file_path, [str]:     list of absolute file paths

    Returns:
        Adding unique file paths to list_file_path, [str]
    '''
    root_path, list_dirs, filenames = next(os.walk(dir_path))
    for filename in filenames:
        exts = filename.split('.')
        if exts[-1] in supported_formats and exts[0] != '':
            file_path = root_path + '/' + filename
            if file_path not in list_file_path:
                list_file_path.append(file_path)

    for dirname in list_dirs:
        new_dir_path = dir_path + '/' + dirname
        list_file_path = add_paths_from_dir(new_dir_path, supported_formats, list_file_path)

    return list_file_path