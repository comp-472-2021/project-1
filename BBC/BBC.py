from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy
import matplotlib.pyplot as plt
import os

# Question 2

business_files_count = len(os.listdir("BBC\\business"))
entertainment_files_count = len(os.listdir("BBC\\entertainment"))
politics_files_count = len(os.listdir("BBC\\politics"))
sport_files_count = len(os.listdir("BBC\\sport"))
tech_files_count = len(os.listdir("BBC\\tech"))

names = ['business', 'entertainment', 'politics', 'sport', 'tech']
values = [business_files_count, entertainment_files_count, politics_files_count, sport_files_count, tech_files_count]

fig = plt.figure(figsize=(7, 7))
x_location = [i + 1 for i in range(0, 5)]
for i, v in enumerate(values):
    plt.text(x_location[i] - 1, v + 2, str(v))
plt.bar(names, values)
fig.savefig('BBC-distribution.pdf', dpi=fig.dpi)

# Question 3

BBC_data = load_files('BBC\BBC', encoding='latin1')
X = BBC_data.data
y = BBC_data.target
X_train_counts = CountVectorizer().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_train_counts, y, test_size=0.2, random_state=None)
classifier = MultinomialNB().fit(X_train, y_train)

# some basic test
y_predict = classifier.predict(X_test)
print(numpy.mean(y_predict == y_test))
