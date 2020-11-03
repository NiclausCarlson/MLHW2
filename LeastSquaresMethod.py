import Utils as Utils
import numpy as numpy


def computeError(real, predicted):
    return sum([(zipped[0] - zipped[1]) ** 2 for zipped in zip(predicted, real)])


data = Utils.getNormalizedDataAndClasses()
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

step = -2
minError = 100000000000000000000
idMatrix = numpy.diag(numpy.ones(len(trainingObjects) + 1))
minTheta = []

while step <= 2:
    theta = numpy.linalg.inv(
        trainingObjects.transpose() @ trainingObjects + step * idMatrix) @ trainingObjects.transpose() @ y
    yPredicted = trainingObjects.dot(theta)
    error = computeError(y, yPredicted)
    if error < minError:
        minError = error
        minTheta = theta
    step += 0.1

f.write("With regularization:\n")
f.write(numpy.array_str(minTheta) + "\n\n")
f.write("Error: " + numpy.str(minError) + "\n")

f.close()
