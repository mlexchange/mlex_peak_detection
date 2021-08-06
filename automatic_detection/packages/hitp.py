"""
Code grabbed from https://github.com/tangkong/dataproc/blob/master/dataproc/operations/hitp.py
by Robert Tang-Kong
Operations for diffraction image reduction
Includes:
"""
import os
from pathlib import Path
import numpy as np
from numpy.polynomial.chebyshev import chebval

import scipy
import scipy.io
from scipy.optimize import curve_fit, basinhopping
from scipy import integrate, signal
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad
from scipy.special import wofz

import pandas as pd
import fabio
import re

import matplotlib.pyplot as plt


def bayesian_block_finder(x: np.ndarray = np.ones(5, ), y: np.ndarray = np.ones(5, ), ):
    """bayesian_block_finder performs Bayesian Block analysis on x, y data.
    see Jeffrey Scargle's papers at doi: 10.1088/0004-637X/764/2/167
    :param x: array of x-coordinates
    :type x: numpy.ndarray
    :param y: array of y-values with same length as x
    :type y: numpy.ndarray
    """
    data_mode = 3
    numPts = len(x)
    if len(x) != len(y):
        raise ValueError('x and y are not of equal length')

    tt = np.arange(numPts)
    nnVec = []

    sigmaGuess = np.std(y[y <= np.median(y)])
    cellData = sigmaGuess * np.ones(len(x))

    ncp_prior = 0.5

    ## To implement: trimming/choosing of where to start/stop
    ## To implement: timing functions

    cp = []
    cntVec = []

    iterCnt = 0
    iterMax = 10

    while iterCnt <= iterMax:
        best = []
        last = []

        for r in range(numPts):
            # y-data background subtracted
            sumX1 = np.cumsum(y[r::-1])
            sumX0 = np.cumsum(cellData[r::-1])  # sigma guess
            fitVec = (np.power(sumX1[r::-1], 2) / (4 * sumX0[r::-1]))

            paddedBest = np.insert(best, 0, 0)
            best.append(np.amax(paddedBest + fitVec - ncp_prior))
            last.append(np.argmax(paddedBest + fitVec - ncp_prior))

            # print('Best = {0},  Last = {1}'.format(best[r], last[r]))

        # Find change points by peeling off last block iteratively
        index = last[numPts - 1]

        while index > 0:
            cp = np.concatenate(([index], cp))
            index = last[index - 1]

        # Iterate if desired, to implement later
        iterCnt += 1
        break

    numCP = len(cp)
    numBlocks = numCP + 1

    rateVec = np.zeros(numBlocks)
    numVec = np.zeros(numBlocks)

    cptUse = np.insert(cp, 0, 0)

    # print('cptUse start: {0}, end: {1}, len: {2}'.format(cptUse[0],
    # cptUse[-1], len(cptUse)))
    # print('lenCP: {0}'.format(len(cp)))

    # what does this stuff do I don't know... ( good one man )
    print('numBlocks: {0}, dataPts/Block: {1}'.format(numBlocks, len(x) / numBlocks))
    for idBlock in range(numBlocks):
        # set ii1, ii2 as indexes.  Fix edge case at end of blocks
        ii1 = int(cptUse[idBlock])
        if idBlock < (numBlocks - 1):
            ii2 = int(cptUse[idBlock + 1] - 1)
        else:
            ii2 = int(numPts)

        subset = y[ii1:ii2]
        weight = cellData[ii1:ii2]
        if ii1 == ii2:
            subset = y[ii1]
            weight = cellData[ii1]
        rateVec[idBlock] = np.dot(weight, subset) / np.sum(weight)

        if np.sum(weight) == 0:
            raise ValueError('error, divide by zero at index: {0}'.format(idBlock))
            print('-------ii1: {0}, ii2: {1}'.format(ii1, ii2))

    # Simple hill climbing for merging blocks
    cpUse = np.concatenate(([1], cp, [len(y)]))
    cp = cpUse

    numCP = len(cpUse) - 1
    idLeftVec = np.zeros(numCP)
    idRightVec = np.zeros(numCP)

    for i in range(numCP):
        idLeftVec[i] = cpUse[i]
        idRightVec[i] = cpUse[i + 1]

    # Find maxima defining watersheds, scan for
    # highest neighbor of each block

    idMax = np.zeros(numBlocks)
    idMax = np.zeros(numBlocks)
    for j in range(numBlocks):
        jL = (j - 1) * (j > 0) + 0 * (j <= 0)  # prevent out of bounds
        jR = (j + 1) * (j < (numBlocks - 1)) + (numBlocks - 1) * (j >= (numBlocks - 1))

        rateL = rateVec[jL]
        rateC = rateVec[j]
        rateR = rateVec[jR]
        rateList = [rateL, rateC, rateR]

        jMax = np.argmax(rateList)  # get direction [0, 1, 2]
        idMax[j] = j + jMax - 1  # convert direction to delta

    idMax[idMax > numBlocks] = numBlocks
    idMax[idMax < 0] = 0

    # Implement hill climbing (HOP algorithm)

    hopIndex = np.array(range(numBlocks))  # init: all blocks point to self
    hopIndex = hopIndex.astype(int)  # cast all as int
    ctr = 0
    # point each block to its max block
    while ctr <= len(x):
        newIndex = idMax[hopIndex]  # Point each to highest neighbor

        if np.array_equal(newIndex, hopIndex):
            break
        else:
            hopIndex = newIndex.astype(int)

        ctr += 1
        if ctr == len(x):
            print('Hill climbing did not converge...?')

    idMax = np.unique(hopIndex)
    numMax = len(idMax)

    # Convert to simple list of block boundaries
    boundaries = [0]
    for k in range(numMax):
        currVec = np.where(hopIndex == idMax[k])[0]

        rightDatum = idRightVec[currVec[-1]] - 1  # stupid leftover matlab index
        boundaries.append(rightDatum)

    return np.array(boundaries)
