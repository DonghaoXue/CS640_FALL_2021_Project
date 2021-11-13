# File for preprocessing our data
# Ben: I'm familiar with sklearn and pandas, but if someone wants to use PyTorch or something else, go ahead
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def readData(path):
    data = pd.read_csv(path)

def pca(data, num_components):


# Here is also where we would want to do vectorization of text
# We could us https://www.tensorflow.org/tutorials/text/word2vec or something similar