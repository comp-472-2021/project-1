import pandas as panda
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from drugs.constants import DRUGS_FEATURES
from drugs.test import print_prediction


def select_numerical_sex_values_from_sex(sex):
    if sex == "M":
        return 0
    else:
        return 1


def select_numerical_BP_values_from_BP(bp):
    if bp == "LOW":
        return 0
    elif bp == "NORMAL":
        return 1
    else:
        return 2


def select_numerical_cholesterol_values_from_cholesterol(cholesterol):
    if cholesterol == "LOW":
        return 0
    elif cholesterol == "NORMAL":
        return 1
    else:
        return 2


def select_formatted_drugs_features(drugs_csv_model):
    drugs_csv_model[DRUGS_FEATURES["SEX"]] = drugs_csv_model[DRUGS_FEATURES["SEX"]].map(
        select_numerical_sex_values_from_sex)
    drugs_csv_model[DRUGS_FEATURES["BP"]] = drugs_csv_model[DRUGS_FEATURES["BP"]].map(
        select_numerical_BP_values_from_BP)
    drugs_csv_model[DRUGS_FEATURES["CHOLESTEROL"]] = drugs_csv_model[
        DRUGS_FEATURES["CHOLESTEROL"]].map(
        select_numerical_cholesterol_values_from_cholesterol)
    return drugs_csv_model


def fetch_drugs_features_data():
    formatted_drugs_csv_model = select_formatted_drugs_features(panda.read_csv('../drug200.csv'))
    return formatted_drugs_csv_model.drop(DRUGS_FEATURES["DRUG"], axis=1)


def fetch_drugs_target_data():
    return panda.read_csv('../drug200.csv')[DRUGS_FEATURES["DRUG"]]


def fetch_drugs_data():
    return train_test_split(fetch_drugs_target_data(), fetch_drugs_features_data())


def get_naive_bayes_classifier(features_vectors):
    initial_naive_bayes_classifier = MultinomialNB(alpha=.01)
    drugs_target_data = fetch_drugs_target_data()
    initial_naive_bayes_classifier.fit(features_vectors, drugs_target_data)
    return initial_naive_bayes_classifier


def question4():
    vectorizer = TfidfVectorizer()

    features_train_set, features_test_set, target_train_set, target_test_set = fetch_drugs_data()
    features_vectors = vectorizer.fit_transform(features_train_set)

    naive_bayes_classifier = get_naive_bayes_classifier(features_vectors)

    # testing
    print_prediction(naive_bayes_classifier, vectorizer)
