# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
dataset.shape[0]


'''count=0
for i in range(dataset.shape[0]):
    if(dataset.values[i,1]==1):
        count += 1'''

# Cleaning the text
import re
corpus=[]
for i in range(dataset.shape[0]):
# step 1 removing everything other than texts
    review = re.sub('[^a-zA-Z]' , ' ' , dataset['Review'][i])
#step 2 changing to lower case
    review = review.lower()
#step 3 removing non significant words(articles,prepositions etc)
    review = review.split()    
    for word in review:
        if word in set(stopwords.words('english')):
            review.remove(word)
# review 
#step 4 stemming keeping only the root word    
    ps=PorterStemmer()
# review[0]
    for i in range(len(review)):
        review[i]=ps.stem(review[i])
# review
#step 5 remapping to string
    review=' '.join(review)
    corpus.append(review)


#creating the bag of words by tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features = 1500)
X=cv.fit_transform(corpus).toarray()
Y=dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

#K NN
'''from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,
                                metric='minkowski',
                                p=2)'''
# Fitting Naive Bayes to the Training set
'''from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()'''

# decision tree
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train, Y_train)
# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
