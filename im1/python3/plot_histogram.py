import sys
if not hasattr(sys, 'argv'):
    sys.argv  = ['']

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



intensity = pd.read_csv('data.csv', sep=',', header=None)
I = np.array(intensity)[:,0]
f = np.array(intensity)[:,1]
del intensity

# plt.bar(I,f,width=0.5)
# plt.xlabel('Intensity')
# plt.ylabel('No. of pixels')
