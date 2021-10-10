from drugs.questions.question_4_5_6 import fetch_drugs_data
from drugs.questions.question_6_7 import NB_classifier, decision_tree, grid_search_tree, perceptron, \
    multi_layered_perceptron, grid_search_perceptron


def question_8(file):
    file.write("\n\n --------- question 8 --------- \n")
    features_train_set, features_test_set, target_train_set, target_test_set = fetch_drugs_data()
    NB_classifier(features_train_set, target_train_set, features_test_set, target_test_set, file)
    decision_tree(features_train_set, target_train_set, features_test_set, target_test_set, file)
    grid_search_tree(features_train_set, target_train_set, features_test_set, target_test_set, file)
    perceptron(features_train_set, target_train_set, features_test_set, target_test_set, file)
    multi_layered_perceptron(features_train_set, target_train_set, features_test_set,
                             target_test_set, file)
    grid_search_perceptron(features_train_set, target_train_set, features_test_set, target_test_set,
                           file)
