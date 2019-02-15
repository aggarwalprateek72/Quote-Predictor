#Quote Type Predictor

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset= pd.read_csv('quotes_all.csv', sep=';', header=None)
dataset.columns=['Quotes','Author','Category']

#Cleaning the texts
corpus=[]
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
for i in range(0,75936):
    quote= re.sub('[^a-zA-Z]',' ', quote['Quotes'][i])
    quote=quote.lower().split()
    ps=PorterStemmer()
    quote= [ps.stem(word) for word in quote if not quote in set(stopwords.word('English'))]
    quote= ' '.join(quote)
    corpus.append(quote)

#Creating the bag of words model    
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=5000)
X= cv.fit_transform(corpus).toarray()
y= dataset.iloc[:,2].values

#Splitting the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.25, random_state=0)

#Fitting the model
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(X_train,y_train)

classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)



    