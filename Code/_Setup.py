"""

@author: AL
"""

#%% imports

import os, sys

import pyarrow.feather as feather
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold


# paths
codePath = os.path.abspath('.')


projPath = os.path.dirname(codePath)

dataPath = os.path.join(projPath, "Data")

rawDataPath = os.path.join(dataPath, "rawData")




rawTrainFile = os.path.join(rawDataPath, "train.csv")
rawTestFile = os.path.join(rawDataPath, "test.csv")