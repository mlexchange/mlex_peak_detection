from packages.hitp import bayesian_block_finder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('TiNiSn_500C_Y20190218_14x14_t60_0174_bkgdSub_1D.csv')
data = pd.DataFrame.to_numpy(df)
print(data.shape)

a = bayesian_block_finder(data[:,0], data[:,1])

print(a)
print(data[a.astype(int),0])