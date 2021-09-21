from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from functions import plot_instances
import numpy
import os

# Question 2

business_files_count = len(os.listdir("BBC\\business"))
entertainment_files_count = len(os.listdir("BBC\\entertainment"))
politics_files_count = len(os.listdir("BBC\\politics"))
sport_files_count = len(os.listdir("BBC\\sport"))
tech_files_count = len(os.listdir("BBC\\tech"))

news_names = ['business', 'entertainment', 'politics', 'sport', 'tech']
news_values = [business_files_count, entertainment_files_count, politics_files_count, sport_files_count, tech_files_count]
news_pdf = 'BBC-distribution.pdf'
plot_instances(news_pdf, news_names, news_values)

# Question 3

BBC_data = load_files('BBC', encoding='latin1')
X = BBC_data.data
y = BBC_data.target
X_train_counts = CountVectorizer().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_train_counts, y, test_size=0.2, random_state=None)
classifier = MultinomialNB().fit(X_train, y_train)

# some basic test
y_predict = classifier.predict(X_test)
print(numpy.mean(y_predict == y_test))
