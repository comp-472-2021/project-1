from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def NB_classifier(features_train_set, target_train_set, features_test_set, target_test_set):
    nb_classifier = GaussianNB()
    nb_classifier.fit(features_train_set, target_train_set)

    # testing
    prediction_results = nb_classifier.predict(features_test_set)
    print(metrics.accuracy_score(target_test_set, prediction_results))


def decision_tree(features_train_set, target_train_set, features_test_set, target_test_set):
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(features_train_set, target_train_set)

    # testing
    prediction_results = decision_tree_classifier.predict(features_test_set)
    print(metrics.accuracy_score(target_test_set, prediction_results))


def grid_search_tree(features_train_set, target_train_set, features_test_set, target_test_set):
    param_grid = {"criterion": ["gini"],
                  'max_depth': list(range(1, 15)),
                  'min_samples_split': list(range(5, 15))
                  }
    grid_search_tree_classifier = GridSearchCV(DecisionTreeClassifier(), param_grid)
    grid_search_tree_classifier.fit(features_train_set, target_train_set)

    # testing
    print(grid_search_tree_classifier.best_params_)


def perceptron(features_vectors, target_train_set, vectorizer, features_test_set):
    return ""


def multi_layered_perceptron(features_vectors, target_train_set, vectorizer, features_test_set):
    return ""


def grid_search_perceptron(features_vectors, target_train_set, vectorizer, features_test_set):
    return ""
