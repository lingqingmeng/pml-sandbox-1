# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# File writer
import csv

dataset = pd.read_csv('data/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

import re
import nltk

nltk.download('stopwords');

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#input what we dont want to review
#param 2 the space is what it gets replaced into

corpus = []
for i in range(0, 1000):
	review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
	review = review.lower()
	review = review.split()
	ps = PorterStemmer()
	review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]# get rid of a, an the, this, etc words that have no meaning
	review = ' '.join(review); #bring them together as a single string that represents the review
	corpus.append(review)


# key data structure used is a bag of words. it's a matrix where the rows are the reviews (whole text) and the columns are individual words that we used
# then the cell intersecting them represents the frequency count of how many times that word appeared in the review
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)#max_features param remove irrelevant words (keep 1500 most relevant) that also wont appear in other reviews like rick steve


# object need to be fitted first to corpus. then do transform which means make the sparse matrix and putting diff words in own column
X = cv.fit_transform(corpus).toarray(); # matrix of independent variables X
y = dataset.iloc[:,1].values # create dependent variable vector (whether the tone was positive or negative)

## finally 
np.savetxt('output/matrix.txt',X, fmt='%i')
print(y);

### pause point go to naive bayes and understand the code there https://www.udemy.com/machinelearning/learn/v4/t/lecture/6067282
