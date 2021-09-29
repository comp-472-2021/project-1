import numpy
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix as cm, classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from common import set_cwd


def main():
    question_3_4_5_6()


def train_classifier(X_train, y_train, X_test, y_test, output,target_name, description, smoothing_value):
    # Question 6 will be put in an function with Q7 later for reuse purpose
    multinomialNB = MultinomialNB(alpha=smoothing_value)
    classifier = multinomialNB.fit(X_train, y_train)

    # Question 7

    output.write("(a) ***************  " + description + "  ***************\n")

    y_predict = classifier.predict(X_test)
    confusion_matrix = cm(y_test, y_predict)
    output.write("(b) confusion_matrix:\n")
    for column in confusion_matrix:
        for value in column:
            output.write(str(value) + " " * (10 - len(str(value))))
        output.write("\n")

    output.write("(c) classification report:\n")
    output.write(classification_report(y_test, y_predict, target_names=target_name))
    output.write("(d) More detailed accuracy: " + str(accuracy_score(y_test, y_predict)) + "\n")
    output.write(
        "More detailed macro-average F1: " + str(
            f1_score(y_test, y_predict, average="micro")) + "\n")
    output.write("More detailed weighted-average F1: " + str(
        f1_score(y_test, y_predict, average="weighted")) + "\n")


    output.write("\n")
    print(numpy.mean(y_predict == y_test))


def question_3_4_5_6():
    set_cwd()
    # Question 3
    BBC_data = load_files('BBC/BBC', encoding='latin1')
    X = BBC_data.data
    y = BBC_data.target
    target_name = BBC_data.target_names

    # Question 4
    X_train_counts = CountVectorizer().fit_transform(X)

    # Question 5
    X_train, X_test, y_train, y_test = train_test_split(X_train_counts, y, test_size=0.2,
                                                        random_state=None)

    output = open("BBC/bbc-performance.txt", "w")

    train_classifier(X_train, y_train, X_test, y_test, output, target_name, "MultinomialNB default values, try 1", 1)
    train_classifier(X_train, y_train, X_test, y_test, output, target_name, "MultinomialNB default values, try 2", 1)
    train_classifier(X_train, y_train, X_test, y_test, output, target_name, "MultinomialNB smoothing value 0.0001, "
                                                                            "try 1", 0.0001)
    train_classifier(X_train, y_train, X_test, y_test, output, target_name, "MultinomialNB smoothing value 0.9, "
                                                                            "try 1", 0.9)

    # some basic test
