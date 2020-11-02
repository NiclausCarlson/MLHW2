import matplotlib.pyplot as plt


def getTrainingData():
    training = open('data/training.txt', 'r')
    numberOfFeatures = int(training.readline())
    numberOfObjects = int(training.readline())
    objects = []

    for line in training:
        objects.append(list(map(int, line.split())))

    training.close()
    return [objects, numberOfFeatures, numberOfObjects]


def normalize(data):
    minimums = [min(row) for row in zip(*data)]
    maximums = [max(row) for row in zip(*data)]

    for i in range(len(data)):
        for j in range(len(data[i])):
            if maximums[j] != minimums[j]:
                data[i][j] = (data[i][j] - minimums[j]) / (maximums[j] - minimums[j])
            else:
                data[i][j] = 0


def smape(predicted, real):
    assert len(predicted) == len(real)
    try:
        return 1 / len(predicted) * \
               (sum([abs(predicted[k] - real[k]) / (abs(predicted[k]) + abs(real[k])) for k in
                     range(0, len(predicted))]))
    except ZeroDivisionError:
        print("Division by zero")


def grad(data, maxIters):
    return 0


def printGraph(x, y, name):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()
    fig.savefig(name)
