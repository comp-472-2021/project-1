from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy

BBC_data=load_files('BBC\BBC',encoding='latin1')
X=BBC_data.data
y=BBC_data.target
X_train_counts=CountVectorizer().fit_transform(X)



X_train, X_test, y_train, y_test =train_test_split(X_train_counts,y,test_size=0.2, random_state=None)
classifier=MultinomialNB().fit(X_train, y_train)



#some basic test
y_predict=classifier.predict(X_test)
print(numpy.mean(y_predict==y_test))