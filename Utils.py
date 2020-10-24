import matplotlib.pyplot as plt


def smape(predicted, real):
    assert len(predicted) == len(real)
    try:
        return 1 / len(predicted) * \
               (sum([abs(predicted[k] - real[k]) / (abs(predicted[k]) + abs(real[k])) for k in
                     range(0, len(predicted))]))
    except ZeroDivisionError:
        print("Division by zero")


def rmsd(dataList):
    return 0


def printGraph(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()
    fig.savefig('function graph')
