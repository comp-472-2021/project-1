import os

import matplotlib.pyplot as plt
from sklearn import metrics

path_to_drug_performance = "drugs/outputs/drug-performance.txt"


def set_cwd():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)


def plot_instances(pdf_name, names, values):
    set_cwd()
    fig = plt.figure(figsize=(7, 7))
    x_location = [i + 1 for i in range(0, 5)]
    for i, v in enumerate(values):
        plt.text(x_location[i] - 1, v + 2, str(v))
    plt.bar(names, values)
    fig.savefig(pdf_name, dpi=fig.dpi)


def clear_prediction_results():
    set_cwd()
    if os.path.exists(path_to_drug_performance):
        os.remove(path_to_drug_performance)


def output_prediction_results(target_test_set, prediction_results, header, file,
                              parameters="default"):
    set_cwd()
    performance_content = "\n \n***** " + header + " *****"

    performance_content += "\nParameters: " + parameters

    performance_content += "\nConfusion matrix: " + str(metrics.confusion_matrix(target_test_set,
                                                                                 prediction_results))
    performance_content += "\nPrecision: " + str(
        metrics.precision_score(target_test_set, prediction_results, average="weighted"))

    performance_content += "\nRecall: " + str(
        metrics.accuracy_score(target_test_set, prediction_results))

    performance_content += "\nF1 score: " + str(
        metrics.f1_score(target_test_set, prediction_results, average=None))

    performance_content += "\nAccuracy: " + str(
        metrics.accuracy_score(target_test_set, prediction_results))

    performance_content += "\nMacro-average F1 score: " + str(
        metrics.f1_score(target_test_set, prediction_results, average="macro"))

    performance_content += "\nWeighted-average F1 score: " + str(
        metrics.f1_score(target_test_set, prediction_results, average="weighted"))

    print(performance_content)

    file.write(performance_content)


def output_hyper_parameters(hyper_parameters, model_name, header, file):
    set_cwd()
    content = "\n \n***** " + header + " *****"
    content += "\n" + model_name + " Hyper Parameters: " + hyper_parameters
    print(content)
    file.write(content)
