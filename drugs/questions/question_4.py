import pandas as panda
from sklearn.feature_extraction.text import TfidfVectorizer
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
    formatted_drugs_csv_model = formatted_drugs_csv_model.drop(DRUGS_FEATURES["DRUG"], axis=1)
    return formatted_drugs_csv_model.head()


def fetch_drugs_target_data(drugs_csv_model):
    return drugs_csv_model[DRUGS_FEATURES["DRUG"]].head()


def get_naive_bayes_classifier(drugs_csv_model, features_vectors):
    initial_naive_bayes_classifier = MultinomialNB(alpha=.01)
    drugs_target_data = fetch_drugs_target_data(drugs_csv_model)
    initial_naive_bayes_classifier.fit(features_vectors, drugs_target_data)
    return initial_naive_bayes_classifier


def question4():
    drugs_csv_model = panda.read_csv('../drug200.csv')
    vectorizer = TfidfVectorizer()

    drugs_features = fetch_drugs_features_data()
    features_vectors = vectorizer.fit_transform(drugs_features)

    naive_bayes_classifier = get_naive_bayes_classifier(drugs_csv_model, features_vectors)

    # testing
    print_prediction(naive_bayes_classifier, vectorizer)


fetch_drugs_features_data()
