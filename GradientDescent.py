import random as random
import numpy as numpy
import math as math
import Utils as Utils

data = Utils.getNormalizedDataAndClasses()
trainingObjects = numpy.array(data[0])
y = data[1]
SAMPLE_SIZE = data[3]
BATCH_SIZE = 3


# функция ошибки 1/l * sum_{i=1}^n (<w, x_i> - y_i)^2, где x - выборка
# производная 2/l * sum_{i=1}^n (<w, x_i> - y_i)*x_i
# тогда j-й элемент градиента(производная по w_j): 2/l * (<w, x_j> - y_j) * x_j
# если j == 0, то 2/l * (<w, x_j> - y_0)
def computeGrad(i, w):  # номер переменной, значения весов
    if i == 0:
        return (2 / SAMPLE_SIZE) * (w.dot(trainingObjects[i]) - y[
            i])  # что-то странное, в ожном случае возвращается число, в другом вектор
    return (2 / SAMPLE_SIZE) * (w.dot(trainingObjects[i]) - y[i]) * trainingObjects[i]


def getBatchGrad(w):
    variablesIndexes = [random.randint(0, len(w)) for i in range(BATCH_SIZE)]
    batchGrad = numpy.array([0 for i in range(len(w))])
    for i in variablesIndexes:
        batchGrad[i] = computeGrad(i, w)
    return batchGrad


EPSILON = 0.000001
weights = numpy.array([1 / (2 * random.randint(1, i)) for i in range(1, 10)])  # веса из (0, 1/2^n]

# пакетный градиентный спуск - берём какой-то набор часнтных происводных и на них обновляем  веса
for i in range(2000):
    coef = 1 / (i + 1)
    newWeights = weights - coef * getBatchGrad(weights)
    if math.sqrt(sum([(x[0] + x[1]) ** 2 for x in zip(weights, newWeights)])) < EPSILON:
        break
    weights = newWeights

