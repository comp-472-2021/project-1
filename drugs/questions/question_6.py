from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from common import output_prediction_results, output_hyper_parameters


def NB_classifier(features_train_set, target_train_set, features_test_set, target_test_set):
    nb_classifier = GaussianNB()
    nb_classifier.fit(features_train_set, target_train_set)

    prediction_results = nb_classifier.predict(features_test_set)
    output_prediction_results(target_test_set, prediction_results, "A) NB Classifier")


def decision_tree(features_train_set, target_train_set, features_test_set, target_test_set):
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(features_train_set, target_train_set)

    prediction_results = decision_tree_classifier.predict(features_test_set)
    output_prediction_results(target_test_set, prediction_results, "B) Decision Tree")


def grid_search_tree(features_train_set, target_train_set, features_test_set, target_test_set):
    param_grid = {"criterion": ["gini", "entropy"],
                  'max_depth': list(range(1, 15)),
                  'min_samples_split': list(range(5, 15))
                  }
    grid_search_tree_classifier = GridSearchCV(DecisionTreeClassifier(), param_grid)
    grid_search_tree_classifier.fit(features_train_set, target_train_set)

    output_hyper_parameters(str(grid_search_tree_classifier.best_params_), "Search tree",
                            "C) Search tree GridSearch")


def perceptron(features_train_set, target_train_set, features_test_set, target_test_set):
    perceptron_classifier = Perceptron()
    perceptron_classifier.fit(features_train_set, target_train_set)

    prediction_results = perceptron_classifier.predict(features_test_set)
    output_prediction_results(target_test_set, prediction_results, "D) Perceptron")


def multi_layered_perceptron(features_train_set, target_train_set, features_test_set,
                             target_test_set):
    multi_layered_perceptron_classifier = MLPClassifier(hidden_layer_sizes=(100,),
                                                        activation="logistic",
                                                        solver="sgd")
    multi_layered_perceptron_classifier.fit(features_train_set, target_train_set)

    prediction_results = multi_layered_perceptron_classifier.predict(features_test_set)
    output_prediction_results(target_test_set, prediction_results, "E) Multi-level perceptron",
                              "hidden_layer_sizes=(100,),activation = logistic,solver = sgd")


def grid_search_perceptron(features_train_set, target_train_set, features_test_set,
                           target_test_set):
    param_grid = {
        "activation": ["tanh", "relu", "and", "identity"],
        "hidden_layer_sizes": [(100,), (10, 10, 10), (30, 50)],
        "solver": ["lbfgs", "sgd", "adam"]
    }
    grid_mlp_classifier = GridSearchCV(MLPClassifier(hidden_layer_sizes=(100,),
                                                     activation="logistic",
                                                     solver="sgd"), param_grid)
    grid_mlp_classifier.fit(features_train_set, target_train_set)

    output_hyper_parameters(str(grid_mlp_classifier.best_params_), "Perceptron",
                            "F) Perceptron GridSearch")
