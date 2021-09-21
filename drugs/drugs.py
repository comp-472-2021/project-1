import pandas as panda
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from constants import DRUGS_FEATURES
from test import print_prediction
import matplotlib.pyplot as plt

drugs_csv_model = panda.read_csv('drug200.csv')

#Question 3

drugA_count = drugs_csv_model.Drug.value_counts().drugA
drugB_count = drugs_csv_model.Drug.value_counts().drugB
drugC_count = drugs_csv_model.Drug.value_counts().drugC
drugX_count = drugs_csv_model.Drug.value_counts().drugX
drugY_count = drugs_csv_model.Drug.value_counts().drugY

names = sorted(drugs_csv_model['Drug'].unique())
values = [drugA_count, drugB_count, drugC_count, drugX_count, drugY_count]

fig = plt.figure(figsize=(7, 7))
x_location = [i + 1 for i in range(0, 5)]
for i, v in enumerate(values):
    plt.text(x_location[i] - 1, v + 2, str(v))
plt.bar(names, values)
fig.savefig('drug-distribution.pdf', dpi=fig.dpi)

# Question 4

def fetch_drugs_features_data(drugs_csv_model):
    return drugs_csv_model.drop(DRUGS_FEATURES["DRUG"], axis=1).head()


def fetch_drugs_target_data(drugs_csv_model):
    return drugs_csv_model[DRUGS_FEATURES["DRUG"]].head()


def get_naive_bayes_classifier(drugs_csv_model, features_vectors):
    initial_naive_bayes_classifier = MultinomialNB(alpha=.01)
    drugs_target_data = fetch_drugs_target_data(drugs_csv_model)
    initial_naive_bayes_classifier.fit(features_vectors, drugs_target_data)
    return initial_naive_bayes_classifier


drugs_csv_model = panda.read_csv('drug200.csv')
vectorizer = TfidfVectorizer()

drugs_features = fetch_drugs_features_data(drugs_csv_model)
features_vectors = vectorizer.fit_transform(drugs_features)

naive_bayes_classifier = get_naive_bayes_classifier(drugs_csv_model, features_vectors)

# testing
print_prediction(naive_bayes_classifier, vectorizer)
