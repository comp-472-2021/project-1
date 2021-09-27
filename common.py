import os

import matplotlib.pyplot as plt
from sklearn import metrics


def plot_instances(pdf_name, names, values):
    fig = plt.figure(figsize=(7, 7))
    x_location = [i + 1 for i in range(0, 5)]
    for i, v in enumerate(values):
        plt.text(x_location[i] - 1, v + 2, str(v))
    plt.bar(names, values)
    fig.savefig("outputs/" + pdf_name, dpi=fig.dpi)


def clear_prediction_results():
    os.remove("../outputs/drug-performance.txt")


def output_prediction_results(target_test_set, prediction_results, header, parameters="default"):
    file = open("../outputs/drug-performance.txt", 'a+')
    performance_content = "\n \n***** " + header + " *****"
    performance_content += "\nParameters: " + parameters
    performance_content += "\nPrecision: " + str(
        metrics.precision_score(target_test_set, prediction_results, average="weighted"))
    performance_content += "\nRecall: " + str(
        metrics.accuracy_score(target_test_set, prediction_results))
    performance_content += "\nF1 score: " + str(
        metrics.f1_score(target_test_set, prediction_results, average="weighted"))
    print(performance_content)
    file.write(performance_content)
