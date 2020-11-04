import Utils as Utils
import numpy as numpy
import math as math


def computeError(real, predicted):
    return math.sqrt(sum([(zipped[0] - zipped[1]) ** 2 for zipped in zip(predicted, real)]))


data = Utils.getNormalizedTrainingDataAndClasses()
trainingObjects = data[0]
y = data[1]

# добавил единицы как первый столбец
ones = numpy.ones((len(trainingObjects), 1))
trainingObjects = numpy.array(trainingObjects)
trainingObjects = numpy.hstack((ones, trainingObjects))

singular = numpy.linalg.svd(trainingObjects, full_matrices=False, compute_uv=True)

diagSigma = numpy.diag(singular[1])
invF = singular[2].transpose() @ numpy.linalg.pinv(diagSigma) @ singular[0].transpose()
theta = invF @ y
yPredicted = trainingObjects.dot(theta)
error = computeError(y, yPredicted)
f = open("leastSquares.txt", "w")

f.write("Without regularization:\n")
f.write(numpy.array_str(theta) + "\n\n")
f.write("Error: " + numpy.str(error) + "\n\n")

param = -2
bestParam = 0
minError = 100000000000000000000
idMatrix = numpy.diag(numpy.ones(len(trainingObjects)))
minTheta = []

while param <= 2:
    theta = singular[2].transpose() @ numpy.linalg.inv(
        diagSigma @ diagSigma + param * idMatrix) @ diagSigma @ singular[0].transpose() @ y
    yPredicted = trainingObjects.dot(theta)
    error = computeError(y, yPredicted)
    if error < minError:
        minError = error
        minTheta = theta
        bestParam = param
    param += 0.1

f.write("With regularization:\n")
f.write(numpy.array_str(minTheta) + "\n\n")
f.write("Error: " + numpy.str(minError) + "\n")
f.write("Regularization param: " + str(bestParam))
f.close()
