import numpy
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix as cm, classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from common import set_cwd


def main():
    question_3_4_5_6()


def train_classifier(X_train, y_train, X_test, y_test, output, target_name, description, smoothing_value):
    # Question 6 will be put in an function with Q7 later for reuse purpose
    multinomialNB = MultinomialNB(alpha=smoothing_value)
    classifier = multinomialNB.fit(X_train, y_train)

    # Question 7

    number_of_classes = len(target_name)
    output.write("(a) ***************  " + description + "  ***************\n")

    y_predict = classifier.predict(X_test)
    confusion_matrix = cm(y_test, y_predict)

    output.write("\n(b) Confusion_matrix:\n")
    for column in confusion_matrix:
        for value in column:
            output.write(str(value) + " " * (10 - len(str(value))))
        output.write("\n")

    output.write("\n(c) Classification report:\n")
    output.write(classification_report(y_test, y_predict, target_names=target_name))

    output.write("\n(d) More detailed accuracy: " + str(accuracy_score(y_test, y_predict)) + "\n")
    output.write(
        "More detailed macro-average F1: " + str(
            f1_score(y_test, y_predict, average="micro")) + "\n")
    output.write("More detailed weighted-average F1: " + str(
        f1_score(y_test, y_predict, average="weighted")) + "\n")

    output.write("\n(e) Prior probability of each class:\n")
    y_list = y_train.tolist()
    for x in range(0, number_of_classes):
        y_list_class = y_list.count(x)
        prior_y_list = y_list_class / len(y_list)
        output.write(f"Class {target_name[x]}: {prior_y_list}\n")

    output.write("\n(f) Size of the vocabulary:\n")
    number_of_different_words = X_train.shape[1]
    output.write(f"There are {number_of_different_words} different words.\n")

    output.write("\n(g) Number of word-tokens by class:\n")
    number_of_words_each_class = multinomialNB.feature_count_.sum(axis=1)
    for x in range(0, number_of_classes):
        output.write(f"Class {target_name[x]}: {number_of_words_each_class[x]} words in total.\n")

    output.write("\n(h) Number of word-tokens in the entire corpus:\n")
    output.write(f"There are {number_of_words_each_class.sum(axis=0)} words in total in the entire corpus.\n")

    output.write("\n(i) Number and percentage of words with a frequency of zero in each class:\n")
    for x in range(0, number_of_classes):
        number_of_zeroes = numpy.count_nonzero(multinomialNB.feature_count_[x] == 0)
        output.write(f"Class {target_name[x]}: {number_of_zeroes} words with "
                     f"frequency of zero. This is {number_of_zeroes*100/number_of_different_words}% of the words.\n")

    output.write("\n(j) Number and percentage of words with a frequency of one in the entire corpus:\n")
    frequency_of_words_in_corpus = multinomialNB.feature_count_.sum(axis=0)
    number_of_ones = numpy.count_nonzero(frequency_of_words_in_corpus == 1)
    output.write(f"There are {number_of_ones} words that appear only once in the entire corpus. This is {number_of_ones*100/number_of_different_words}% of the words.\n")

    output.write("\n(k) Two favorite words and their log-prob:\n")
    # Word 1: apple (indice: 2639), word 2: parked (indice: 19453) found with count_vectorizer.vocabulary_
    output.write("Log probabilities for the word apple: \n")
    for x in range(0, number_of_classes):
        feature_log_prob = multinomialNB.feature_log_prob_[x,2639]
        output.write(f"Class {target_name[x]}: {feature_log_prob}\n")
    output.write("\nLog probabilities for the word parked: \n")
    for x in range(0, number_of_classes):
        feature_log_prob = multinomialNB.feature_log_prob_[x, 19453]
        output.write(f"Class {target_name[x]}: {feature_log_prob}\n")




    output.write("\n")
    # print(numpy.mean(y_predict == y_test))


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
