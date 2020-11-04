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


def getNormalizedTrainingDataAndClasses():
    data = getTrainingData()
    trainingObjects = data[0]
    normalize(trainingObjects)

    # вектор меток
    y = [trainingObjects[i][data[1]] for i in range(data[2])]

    # матрица признаков
    for i in range(data[2]):
        del trainingObjects[i][-1]

    return [trainingObjects, y, data[1], data[2]]


def getNormalizedTestData():
    test = open('data/test.txt')
    quantity = int(test.readline())
    objects = []

    for line in test:
        objects.append(list(map(int, line.split())))
    test.close()
    normalize(objects)

    y = [objects[i][len(objects[i]) - 1] for i in range(quantity)]
    for i in range(quantity):
        del objects[i][-1]

    return [objects, y]


def smape(predicted, real):
    assert len(predicted) == len(real)
    try:
        return 1 / len(predicted) * \
               (sum([abs(predicted[k] - real[k]) / (abs(predicted[k]) + abs(real[k])) for k in
                     range(0, len(predicted))]))
    except ZeroDivisionError:
        print("Division by zero")


def printGraph(x, y, name, color):
    param = ""
    if color == "blue":
        param = "b"
    elif color == "red":
        param = "r"
    fig, ax = plt.subplots()
    ax.plot(x, y, param)
    ax.grid(True)
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(u'Зависимость SMAPE от числа итераций (' + name + ')')
    plt.show()
    fig.savefig(name)
