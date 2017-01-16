import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# File writer
import csv

## test bed
a = np.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
a.tofile('output/foo.csv',sep=',',format='%10.5f')
x = np.arange(20).reshape((4,5))
np.savetxt('output/test.txt', x)
## test bed end

