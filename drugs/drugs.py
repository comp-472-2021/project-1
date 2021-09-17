import pandas as panda
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from constants import DRUGS_FEATURES


def fetch_drugs_data(drugs_csv_model):
    return drugs_csv_model.drop(DRUGS_FEATURES["DRUG"], axis=1).head()


def fetch_drugs_test_data(drugs_csv_model):
    return drugs_csv_model.drop(DRUGS_FEATURES["DRUG"], axis=1).tail()


def fetch_drugs_target(drugs_csv_model):
    return drugs_csv_model[DRUGS_FEATURES["DRUG"]].head()


drugs_csv_model = panda.read_csv('drug200.csv')
vectorizer = TfidfVectorizer()

drugs_data = fetch_drugs_data(drugs_csv_model)
vectors = vectorizer.fit_transform(drugs_data)

drugs_test_data = fetch_drugs_test_data(drugs_csv_model)
vectors_test = vectorizer.transform(drugs_test_data)

clf = MultinomialNB(alpha=.01)
clf.fit(vectors, fetch_drugs_target(drugs_csv_model))
