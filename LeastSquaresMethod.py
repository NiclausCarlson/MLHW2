import Utils as Utils
import numpy as numpy

data = Utils.getTrainingData()
trainingObjects = data[0]
Utils.normalize(trainingObjects)

# вектор меток
y = [trainingObjects[i][data[1]] for i in range(data[2])]

# матрица признаков
for i in range(data[2]):
    del trainingObjects[i][-1]

# добавил единицы как первый столбец ?ЗАЧЕМ?
ones = numpy.ones((len(trainingObjects), 1))
trainingObjects = numpy.array(trainingObjects)
trainingObjects = numpy.hstack((ones, trainingObjects))

singular = numpy.linalg.svd(trainingObjects)
# возвращает диагональную матрицу как вектор, лол
print(singular[2])

# не сходятся размерности матриц
diagSigma = numpy.diag(singular[1])
invF = singular[2] @ numpy.linalg.pinv(diagSigma) @ singular[0].transpose()
theta = invF @ y
# без регуляризации
print(theta)

step = -2
minError = 100000000000000000000
idMatrix = numpy.diag(numpy.ones((len(trainingObjects), 1)))
transposeDiagSigma = numpy.transpose(diagSigma)

while step != 2:
    theta = numpy.linalg.inv(transposeDiagSigma @ diagSigma + step * idMatrix) @ transposeDiagSigma @ y
    yPredicted = trainingObjects @ theta
    error = sum([(zipped[0] - zipped[1]) ** 2 for zipped in zip(yPredicted, y)])
    if error < minError:
        minError = error
        minTheta = theta
    step += 0.1

print(minTheta)
