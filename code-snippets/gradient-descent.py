import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np
from AdalineGD import AdalineGD
from plot_decision_regions import plot_decision_regions

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values

# Subtract sample mean from every training sample and divide it by its standard deviation to normalize this dataset

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

# After standardization, we will train the Adaline again and see that it now converges using a learning rate eta = 0.01

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std,y)


plot_decision_regions(X_std,y,classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('speal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1,len(ada.cost_) +1),ada.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()