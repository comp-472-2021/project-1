from numpy.random import multinomial
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report,f1_score,accuracy_score


import numpy
# Question 3
# might need to modify the path depending on ide 
BBC_data = load_files('BBC/BBC', encoding='latin1')
X = BBC_data.data
y = BBC_data.target
target_name=BBC_data.target_names
# Question 4
X_train_counts = CountVectorizer().fit_transform(X)
# Question 5
X_train, X_test, y_train, y_test = train_test_split(X_train_counts, y, test_size=0.2, random_state=None)

# might need to modify the path depending on ide 
output=open("BBC/bbc-performance.txt","w")
# Question 6 will be put in an function with Q7 later for reuse purpose
multinomialNB=MultinomialNB()
classifier = multinomialNB.fit(X_train, y_train)
# Question 7
description="MultinomialNB default values, try 1"
output.write("(a) ***************  "+description+"  ***************\n")

y_predict = classifier.predict(X_test)
confusion_matrix=confusion_matrix(y_test,y_predict)
output.write("(b) confusion_matrix:\n")
for column in confusion_matrix:
    for value in column:
        output.write(str(value)+" "*(10-len(str(value))))
    output.write("\n")

output.write("(c) classification report:\n")
output.write(classification_report(y_test,y_predict,target_names=target_name))  
output.write("(d) More detialed accuracy: "+str(accuracy_score(y_test,y_predict))+"\n")
output.write("More detialed macro-average F1: "+str(f1_score(y_test,y_predict,average="micro"))+"\n")
output.write("More detialed weighted-average F1: "+str(f1_score(y_test,y_predict,average="weighted"))+"\n")


# some basic test

print(numpy.mean(y_predict == y_test))
