import random as random
import numpy as numpy
import math as math
import Utils as Utils

data = Utils.getNormalizedTrainingDataAndClasses()
trainingObjects = numpy.array(data[0])
y = data[1]
SAMPLE_SIZE = data[3]
BATCH_SIZE = 3
EPSILON = 0.000001

testData = Utils.getNormalizedTestData()
testObjects = numpy.array(testData[0])
testY = testData[1]


# функция ошибки 1/l * sum_{i=1}^n (<w, x_i> - y_i)^2, где x - выборка
# производная 2/l * sum_{i=1}^n (<w, x_i> - y_i)*x_i
# тогда j-й элемент градиента(производная по w_j): 2/l * (<w, x_j> - y_j) * x_j
# если j == 0, то 2/l * (<w, x_j> - y_0)
def computeGrad(i, w):  # номер переменной, значения весов
    return (2 / SAMPLE_SIZE) * (w.dot(trainingObjects[i]) - y[i]) * trainingObjects[i]


def getBatchGrad(w):
    variablesIndexes = [random.randint(0, len(w) - 1) for i in range(BATCH_SIZE)]
    batchGrad = [0 for i in range(len(w))]
    for i in variablesIndexes:
        batchGrad += computeGrad(i, w)
    return numpy.array(batchGrad)


def computeSMAPE(weight, y_real, sings):
    y_predicted = []
    for i in range(len(sings)):
        y_predicted.append(sings[i].dot(weight))
    return Utils.smape(y_predicted, y_real)


weights = numpy.array([1 / (2 * random.randint(1, i)) for i in range(1, 10)])  # веса из (0, 1/2^n]

# используем параметр регуляриции, найденный из метода наименьших квадратов
param = 6.38378239159465e-16

smapeOnTrainingSet = []
smapeOnTestSet = []

# пакетный градиентный спуск - берём какой-то набор частных производных и на них обновляем  веса
for i in range(2000):
    coef = 1 / (i + 1)
    newWeights = weights * (1 - coef * param) - coef * getBatchGrad(weights)
    if math.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(weights, newWeights)])) < EPSILON:
        break
    weights = newWeights
    # псчитаю SMAPE
    smapeOnTrainingSet.append(computeSMAPE(weights, y, trainingObjects))
    smapeOnTestSet.append(computeSMAPE(weights, testY, testObjects))

f = open("Gradient.txt", "w")
f.write("Weight:\n")
f.write(numpy.array_str(weights) + "\n")
f.close()

Utils.printGraph([i for i in range(len(smapeOnTrainingSet))], smapeOnTrainingSet, "trainingSet")
Utils.printGraph([i for i in range(len(smapeOnTestSet))], smapeOnTestSet, "testSet")
