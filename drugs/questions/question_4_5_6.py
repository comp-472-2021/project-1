import pandas as panda
from sklearn.model_selection import train_test_split

from common import clear_prediction_results, set_cwd
from drugs.constants import DRUGS_FEATURES
from drugs.questions.question_6 import NB_classifier, decision_tree, grid_search_tree, \
    grid_search_perceptron, multi_layered_perceptron, perceptron


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
    set_cwd()
    formatted_drugs_csv_model = select_formatted_drugs_features(panda.read_csv('drugs/drug200.csv'))
    return formatted_drugs_csv_model.drop(DRUGS_FEATURES["DRUG"], axis=1)


def fetch_drugs_target_data():
    set_cwd()
    drug_targets = panda.read_csv('drugs/drug200.csv')[DRUGS_FEATURES["DRUG"]]
    return panda.Categorical(drug_targets).codes


def fetch_drugs_data():
    return train_test_split(fetch_drugs_features_data(), fetch_drugs_target_data())


def question4_5_6():
    clear_prediction_results()

    features_train_set, features_test_set, target_train_set, target_test_set = fetch_drugs_data()

    # 6 a)
    NB_classifier(features_train_set, target_train_set, features_test_set, target_test_set)

    # 6 b)
    decision_tree(features_train_set, target_train_set, features_test_set, target_test_set)

    # 6 c)
    grid_search_tree(features_train_set, target_train_set, features_test_set, target_test_set)

    # 6 d)
    perceptron(features_train_set, target_train_set, features_test_set, target_test_set)

    # 6 e)
    multi_layered_perceptron(features_train_set, target_train_set, features_test_set,
                             target_test_set)

    # 6 f)
    grid_search_perceptron(features_train_set, target_train_set, features_test_set, target_test_set)


question4_5_6()
