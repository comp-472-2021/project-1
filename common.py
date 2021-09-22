import matplotlib.pyplot as plt

def plot_instances(pdf_name, names, values):
    fig = plt.figure(figsize=(7, 7))
    x_location = [i + 1 for i in range(0, 5)]
    for i, v in enumerate(values):
        plt.text(x_location[i] - 1, v + 2, str(v))
    plt.bar(names, values)
    fig.savefig(pdf_name, dpi=fig.dpi)