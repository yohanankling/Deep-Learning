import matplotlib.pyplot as plt


def plot_results2(res1, res2, label1, label2, x_label, y_label):
    plt.plot(res1, label=label1)
    plt.plot(res2, label=label2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def plot_results3(res1, res2, res3, label1, label2, label3, x_label, y_label):
    plt.plot(res1, label=label1)
    plt.plot(res2, label=label2)
    plt.plot(res3, label=label3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

