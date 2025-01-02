# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts
import re
import nltk

nltk.download('stopwords')  # Library with all the unimportant words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    # Select only words, the other characters are replaced by a space
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()  # Convert the sentence in lower case
    review = review.split()
    ps = PorterStemmer()
    # Remove the irrelevant words and simplify all similar words (Ex: loving, loved to love )
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Accuracy is not enough, so you should also look at other performance metrics like Precision (measuring exactness),
# Recall (measuring completeness) and the F1 Score (compromise between Precision and Recall).
# Please find below these metrics formulas.
# TP = # True Positives, TN = # True Negatives, FP = # False Positives, FN = # False Negatives:
#
# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
# F1 Score = 2 * Precision * Recall / (Precision + Recall)
