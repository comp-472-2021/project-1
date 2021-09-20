import pandas as panda

from constants import DRUGS_FEATURES


def fetch_drugs_test_data(drugs_csv_model):
    return drugs_csv_model.drop(DRUGS_FEATURES["DRUG"], axis=1).tail()


def print_prediction(naive_bayes_classifier, vectorizer):
    drugs_csv_model = panda.read_csv('drug200.csv')
    test_vectors = vectorizer.transform(fetch_drugs_test_data(drugs_csv_model))
    sample_prediction = naive_bayes_classifier.predict(test_vectors)
    print(sample_prediction)
