import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
# [:, ] means all the lines
# :-1] means we take all the columns except the last one
# .values just means the values, python syntax
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# when chaining methods use \ to break new line
from sklearn.preprocessing import Imputer
X[:,1:3] = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) \
			.fit(X[:,1:3]).transform(X[:,1:3])
# We are interested in columns indexed at 1 and 2, but we put 3 because that's the exclusive upper bound.

print(X)
