"""

@author: AL
@author: TP
"""

#%% imports

import os, sys

from missforest.missforest import MissForest
from matplotlib import pyplot as plt
import tensorflow as tf
import ydf
import seaborn as sns
import numpy as np
import pandas as pd
import random
import pickle

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE


# paths
codePath = os.path.abspath('.')


projPath = os.path.dirname(codePath)

dataPath = os.path.join(projPath, "Data")

rawDataPath = os.path.join(dataPath, "rawData")

imputedDataPath = os.path.join(dataPath, "imputedData")


rawTrainFile = os.path.join(rawDataPath, "train.csv")
rawTestFile = os.path.join(rawDataPath, "test.csv")
rawSubmissionFile = os.path.join(rawDataPath, "sample_submission.csv")

rawTrainFilePickle = os.path.join(rawDataPath, "train.pkl")
rawTestFilePickle = os.path.join(rawDataPath, "test.pkl")

tensorDecompTrainFile = os.path.join(imputedDataPath, "tensor_decomp_pre_knn.csv")
imputedTrainFile = os.path.join(imputedDataPath, "iterative_imputed.csv")
kmeansTrainFile = os.path.join(imputedDataPath, "kmeans_imputed.csv")


tensorDecompTrainFilePickle = os.path.join(imputedDataPath, "tensor_decomp_pre_knn.pkl")

# Unimportant/SKlearn generated
imputedTrainFilePickle = os.path.join(imputedDataPath, "iterative_imputed.pkl")

# IMPORTANT TEAM-AGREED SOLUTION

validationFile = os.path.join(imputedDataPath, "imputed_validate.csv")

missForestTrainFile = os.path.join(imputedDataPath, "missforest_imputed_train.csv")

missForestTestFile = os.path.join(imputedDataPath, "missforest_imputed_test.csv")
