from sklearn.naive_bayes import GaussianNB


# top level methods
def NB_classifier(features_train_set, target_train_set, vectorizer, features_test_set):
    nb_classifier = GaussianNB()
    nb_classifier.fit(features_train_set, target_train_set)

    # testing
    test_vectors = vectorizer.transform(features_test_set)
    sample_prediction = nb_classifier.predict(test_vectors)


def decision_tree(features_vectors, target_train_set, vectorizer, features_test_set):
    return ""


def grid_search_tree(features_vectors, target_train_set, vectorizer, features_test_set):
    return ""


def perceptron(features_vectors, target_train_set, vectorizer, features_test_set):
    return ""


def multi_layered_perceptron(features_vectors, target_train_set, vectorizer, features_test_set):
    return ""


def grid_search_perceptron(features_vectors, target_train_set, vectorizer, features_test_set):
    return ""
